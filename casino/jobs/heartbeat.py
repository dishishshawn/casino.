"""Daily heartbeat — one Discord embed summarizing today's state.

Designed to run after the EOD ingest, reconcile_eod, and tsmom_clock_check
tasks have completed (so the daily_pnl row is fresh, drift is checked,
DuckDB is current, and any kill_event has been persisted).

Pre-clock (before the operator runs ``casino.execution.tsmom_runner`` for
the first time on a month-end), the heartbeat shows broker equity, drift
state, and ingest freshness. Post-clock it additionally includes days
elapsed in the 30-day cap, drawdown vs starting NAV, rebal count, and
kill-event count.

Severity rules:

* ``critical`` if any new ``kill_event`` row has been persisted today
  (the kill itself already fired its own critical alert; the heartbeat
  acknowledges it).
* ``warning`` if any of: reconcile drift > 0.5% NAV, drawdown > 5% from
  starting NAV (post-clock only), DuckDB OHLCV gap > 5 calendar days
  (catches pipeline breakage past normal weekend + holiday window).
* ``info`` otherwise.

CLI:
    uv run python -m casino.jobs.heartbeat
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
from loguru import logger

from casino.config import get_config
from casino.execution import book, paper_clock, reconcile
from casino.execution.alpaca_broker import AlpacaBroker, build_default_broker
from casino.monitoring import alerts

# OHLCV gap warning threshold (calendar days). 5 covers a long weekend +
# Monday holiday + Tuesday-morning data lag without false-warning.
OHLCV_STALE_DAYS: int = 5

# Drawdown threshold for promoting heartbeat to warning. Below the 10%
# kill threshold (which fires its own critical alert), but high enough
# to be worth flagging in the daily summary.
HEARTBEAT_DRAWDOWN_WARN: Decimal = Decimal("0.05")

# Reconcile drift threshold for warning (matches the verdict gate).
HEARTBEAT_DRIFT_WARN: Decimal = Decimal("0.005")

# TSMOM universe used to assess DuckDB OHLCV freshness.
TSMOM_UNIVERSE: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "GLD",
    "DBC",
    "USO",
)


@dataclass(frozen=True)
class HeartbeatSummary:
    """Snapshot computed by ``run_heartbeat``. Returned for tests."""

    severity: alerts.Severity
    title: str
    fields: dict[str, str]
    message: str
    has_clock: bool
    days_elapsed: int | None
    drift_fraction: Decimal
    drawdown_from_start: Decimal | None
    ohlcv_gap_days: int | None
    n_kill_events_today: int


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _latest_ohlcv_date(duckdb_path: Path) -> date | None:
    """Return the latest OHLCV UTC date across the TSMOM universe, or None.

    DuckDB's ``MAX(ts)::DATE`` resolves the cast in the session timezone,
    which on a non-UTC host shifts the date by one. We pull the raw
    timestamp and project to a UTC date in Python instead.
    """
    if not duckdb_path.exists():
        return None
    placeholders = ",".join(["?"] * len(TSMOM_UNIVERSE))
    sql = f"SELECT MAX(ts) FROM ohlcv WHERE ticker IN ({placeholders})"
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        row = con.execute(sql, list(TSMOM_UNIVERSE)).fetchone()
    finally:
        con.close()
    if not row or row[0] is None:
        return None
    ts = row[0]
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        return ts.astimezone(UTC).date()
    if hasattr(ts, "date"):
        return ts.date()
    return None


def _drift_fraction(*, broker: AlpacaBroker, db_path: Path | None) -> Decimal:
    """Sum |broker - book| as a fraction of NAV."""
    rec = reconcile.reconcile(broker=broker, db_path=db_path)
    nav = broker.get_account().equity
    if nav <= Decimal("0"):
        return Decimal("0")
    drift_dollars = sum(
        (abs(d.broker_notional - d.book_notional) for d in rec.drift),
        start=Decimal("0"),
    )
    return drift_dollars / nav


def _today_kill_events(
    *,
    run_id: str,
    db_path: Path | None,
    as_of: datetime,
) -> int:
    """Count kill_event rows fired on ``as_of``'s UTC date."""
    rows = paper_clock.fetch_kill_events(run_id=run_id, db_path=db_path)
    today_str = as_of.strftime("%Y-%m-%d")
    return sum(1 for r in rows if r.fired_at_utc.strftime("%Y-%m-%d") == today_str)


def _decide_severity(
    *,
    n_kill_today: int,
    drift_fraction: Decimal,
    drawdown_from_start: Decimal | None,
    ohlcv_gap_days: int | None,
) -> alerts.Severity:
    if n_kill_today > 0:
        return "critical"
    if drift_fraction > HEARTBEAT_DRIFT_WARN:
        return "warning"
    if drawdown_from_start is not None and drawdown_from_start > HEARTBEAT_DRAWDOWN_WARN:
        return "warning"
    if ohlcv_gap_days is not None and ohlcv_gap_days > OHLCV_STALE_DAYS:
        return "warning"
    return "info"


def build_summary(
    *,
    broker: AlpacaBroker,
    db_path: Path | None = None,
    duckdb_path: Path | None = None,
    run_id: str = paper_clock.DEFAULT_RUN_ID,
    as_of: datetime | None = None,
) -> HeartbeatSummary:
    """Compute the heartbeat summary without dispatching to Discord.

    Pure (no I/O beyond the broker, DuckDB, and SQLite reads). Tests can
    pass an in-memory broker fake.
    """
    cfg = get_config()
    as_of = as_of if as_of is not None else _utc_now()

    account = broker.get_account()
    equity = account.equity
    drift = _drift_fraction(broker=broker, db_path=db_path)

    history = book.fetch_daily_pnl(limit=1, db_path=db_path)
    today_pnl = history[0] if history else None

    duckdb_path_resolved = duckdb_path if duckdb_path is not None else cfg.duckdb_path
    latest_bar = _latest_ohlcv_date(duckdb_path_resolved)
    today_local = as_of.date()
    gap = (today_local - latest_bar).days if latest_bar is not None else None

    clock = paper_clock.fetch_paper_clock(run_id=run_id, db_path=db_path)
    has_clock = clock is not None
    days = paper_clock.days_elapsed(run_id=run_id, db_path=db_path) if has_clock else None
    rebals = paper_clock.fetch_rebal_events(run_id=run_id, db_path=db_path) if has_clock else []
    kills = paper_clock.fetch_kill_events(run_id=run_id, db_path=db_path) if has_clock else []
    n_kill_today = _today_kill_events(run_id=run_id, db_path=db_path, as_of=as_of)

    drawdown_from_start: Decimal | None = None
    if has_clock and clock is not None and clock.start_nav > Decimal("0"):
        drawdown_from_start = (clock.start_nav - equity) / clock.start_nav
        if drawdown_from_start < Decimal("0"):
            drawdown_from_start = Decimal("0")

    severity = _decide_severity(
        n_kill_today=n_kill_today,
        drift_fraction=drift,
        drawdown_from_start=drawdown_from_start,
        ohlcv_gap_days=gap,
    )

    # Field labels are deliberately plain so the embed is scannable on
    # mobile. Numbers are rounded — operators don't need 4 decimal places
    # of precision in a daily summary.
    def _money(v: Decimal) -> str:
        try:
            return f"${float(v):,.2f}"
        except (TypeError, ValueError):
            return f"${v}"

    fields: dict[str, str] = {
        "Equity": _money(equity),
        "Drift": f"{drift:.2%}",
        "Latest market data": latest_bar.isoformat() if latest_bar else "n/a",
    }
    if today_pnl is not None:
        total_pl = today_pnl.realized_pl + today_pnl.unrealized_pl
        sign = "+" if total_pl >= Decimal("0") else "-"
        fields["Today P&L"] = f"{sign}{_money(abs(total_pl))}"
        fields["Positions"] = str(today_pnl.n_positions)
    if has_clock and clock is not None:
        fields["Paper-run day"] = f"{days} of {clock.cap_days}"
        fields["Rebalances so far"] = str(len(rebals))
        kill_label = (
            f"{len(kills)} total — {n_kill_today} today" if n_kill_today else f"{len(kills)} total"
        )
        fields["Safety halts"] = kill_label
        if drawdown_from_start is not None:
            fields["Drawdown from start"] = f"{drawdown_from_start:.2%}"
    else:
        fields["30-day paper run"] = (
            "not started yet (first rebal is the last business day of the month)"
        )

    title_suffix = (
        f" — paper-run day {days} of {clock.cap_days}"
        if has_clock and clock is not None
        else " — paper run not started yet"
    )
    title = f"Daily status{title_suffix}"

    parts: list[str] = []
    if severity == "critical":
        plural = "" if n_kill_today == 1 else "s"
        parts.append(f"{n_kill_today} safety check{plural} halted trading today.")
    if has_clock and clock is not None and len(kills) == 0:
        parts.append(f"30-day paper run on track. Day {days} of {clock.cap_days}, no safety halts.")
    if drift > HEARTBEAT_DRIFT_WARN:
        parts.append(
            f"Positions are {drift:.2%} out of sync with the broker — fix before next trading day."
        )
    if gap is not None and gap > OHLCV_STALE_DAYS:
        parts.append(f"Market data is {gap} calendar days stale — the ingest job may be broken.")
    if not parts:
        parts.append("All systems nominal.")
    message = " ".join(parts)

    return HeartbeatSummary(
        severity=severity,
        title=title,
        fields=fields,
        message=message,
        has_clock=has_clock,
        days_elapsed=days,
        drift_fraction=drift,
        drawdown_from_start=drawdown_from_start,
        ohlcv_gap_days=gap,
        n_kill_events_today=n_kill_today,
    )


def run_heartbeat(
    *,
    broker: AlpacaBroker | None = None,
    db_path: Path | None = None,
    duckdb_path: Path | None = None,
    run_id: str = paper_clock.DEFAULT_RUN_ID,
    transport: alerts.WebhookTransport | None = None,
    as_of: datetime | None = None,
) -> tuple[HeartbeatSummary, alerts.AlertResult]:
    """Build the summary and dispatch one Discord embed."""
    actual_broker = broker if broker is not None else build_default_broker()
    summary = build_summary(
        broker=actual_broker,
        db_path=db_path,
        duckdb_path=duckdb_path,
        run_id=run_id,
        as_of=as_of,
    )
    result = alerts.fire(
        title=summary.title,
        message=summary.message,
        severity=summary.severity,
        fields=summary.fields,
        transport=transport,
    )
    return summary, result


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.jobs.heartbeat",
        description="Post a daily heartbeat embed to Discord summarizing today's state.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the summary and log it, but do NOT POST to Discord.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        broker = build_default_broker()
        summary = build_summary(broker=broker)
        logger.info(
            "heartbeat (dry-run): severity={} title={} message={} fields={}",
            summary.severity,
            summary.title,
            summary.message,
            summary.fields,
        )
        return 0

    summary, alert_result = run_heartbeat()
    logger.info(
        "heartbeat: severity={} sent={} reason={} title={}",
        summary.severity,
        alert_result.sent,
        alert_result.reason,
        summary.title,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
