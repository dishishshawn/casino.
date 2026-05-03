"""End-of-day cron entry: reconcile and write daily P&L to the audit DB.

PRD §11 Phase 3 EOD flow:

1. Reconcile broker positions vs internal book; alert critically on any
   non-price drift.
2. Read account equity (open + close), compute realized + unrealized P&L,
   and persist a `daily_pnl` row in `state.sqlite`.
3. Compute drawdown vs the running high-water mark and alert if breached.
4. Optionally re-evaluate the regime gate so the dashboard reflects EOD
   risk-on/off (no order side-effects).

Run: ``uv run python -m casino.jobs.reconcile_eod``
Cron: once daily after the close (RUNBOOK).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from loguru import logger

from casino.execution import book, reconcile
from casino.execution.alpaca_broker import AlpacaBroker, build_default_broker
from casino.monitoring import alerts
from casino.signals import regime

# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class JobResult:
    as_of: datetime
    date: str
    equity_open: Decimal
    equity_close: Decimal
    realized_pl: Decimal
    unrealized_pl: Decimal
    n_positions: int
    n_orders: int
    drift_alerts: int
    drawdown: Decimal
    high_water_mark: Decimal
    risk_on: bool


# ---------------------------------------------------------------------------- helpers


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def compute_drawdown(
    *,
    today_equity: Decimal,
    history: list[book.DailyPnLRow],
) -> tuple[Decimal, Decimal]:
    """Return ``(drawdown_fraction, high_water_mark)``.

    Drawdown is computed against the running max of `equity_close` across
    all stored history (and today's equity). Returns 0 when the high-water
    mark is non-positive.
    """
    closes: list[Decimal] = [today_equity] + [h.equity_close for h in history]
    hwm = max(closes)
    if hwm <= Decimal("0"):
        return Decimal("0"), hwm
    dd = (hwm - today_equity) / hwm
    if dd < Decimal("0"):
        dd = Decimal("0")
    return dd, hwm


def run_reconcile_eod(
    *,
    broker: AlpacaBroker | None = None,
    as_of: datetime | None = None,
    db_path: Path | None = None,
    duckdb_path: Path | None = None,
    drawdown_alert_threshold: Decimal = Decimal("0.10"),
) -> JobResult:
    """Run the EOD reconcile + P&L + drawdown sweep.

    Returns a structured result so the test suite can assert on each
    output field. Side-effects: writes one `daily_pnl` row, may write
    alerts, never submits orders.
    """
    as_of = as_of if as_of is not None else _utc_now()
    if broker is None:
        broker = build_default_broker()
    try:
        # 1) reconcile
        recon = reconcile.reconcile(broker=broker, db_path=db_path)
        critical = reconcile.critical_drift(recon)
        if critical:
            alerts.alert_reconciliation_drift(
                n_drift=len(critical),
                summary="; ".join(d.detail for d in critical),
            )

        # 2) account snapshot
        account = broker.get_account()
        positions = broker.get_positions()

        # 3) regime evaluation (informational; no orders)
        risk_on = regime.is_risk_on(as_of=as_of, db_path=duckdb_path)

        # 4) P&L row
        equity_open = account.last_equity
        equity_close = account.equity
        realized_pl = equity_close - equity_open
        unrealized_pl = sum(
            (p.unrealized_pl for p in positions),
            start=Decimal("0"),
        )
        date_str = as_of.strftime("%Y-%m-%d")
        n_orders = _count_orders_today(db_path=db_path, as_of=as_of)

        history = book.fetch_daily_pnl(limit=2000, db_path=db_path)
        drawdown, hwm = compute_drawdown(today_equity=equity_close, history=history)

        notes = "regime=on" if risk_on else "regime=off"
        if recon.has_drift:
            notes += f"; drift={len(recon.drift)}"
        book.upsert_daily_pnl(
            book.DailyPnLRow(
                date=date_str,
                equity_open=equity_open,
                equity_close=equity_close,
                realized_pl=realized_pl,
                unrealized_pl=unrealized_pl,
                n_positions=len(positions),
                n_orders=n_orders,
                notes=notes,
            ),
            db_path=db_path,
        )

        # 5) drawdown alert
        if drawdown >= drawdown_alert_threshold:
            alerts.alert_drawdown_breach(
                drawdown_pct=float(drawdown),
                high_water_mark=hwm,
                current_equity=equity_close,
                threshold_pct=float(drawdown_alert_threshold),
            )

        return JobResult(
            as_of=as_of,
            date=date_str,
            equity_open=equity_open,
            equity_close=equity_close,
            realized_pl=realized_pl,
            unrealized_pl=unrealized_pl,
            n_positions=len(positions),
            n_orders=n_orders,
            drift_alerts=len(critical),
            drawdown=drawdown,
            high_water_mark=hwm,
            risk_on=risk_on,
        )
    except Exception as e:  # noqa: BLE001
        alerts.alert_unhandled_exception(
            job="reconcile_eod",
            exc_type=type(e).__name__,
            detail=str(e),
        )
        logger.exception("reconcile_eod: unhandled exception")
        raise


def _count_orders_today(
    *,
    as_of: datetime,
    db_path: Path | None,
) -> int:
    """Return the count of orders submitted on `as_of`'s UTC date."""
    book.init_schema(db_path)
    day = as_of.strftime("%Y-%m-%d")
    sql = """
        SELECT COUNT(*) FROM orders
        WHERE substr(submitted_at_utc, 1, 10) = ?
    """
    with book.get_book_conn(db_path) as conn:
        row = conn.execute(sql, (day,)).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.jobs.reconcile_eod",
        description="Run end-of-day reconcile + write daily P&L row.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="ISO timestamp for the EOD run (defaults to UTC now).",
    )
    args = parser.parse_args(argv)
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else _utc_now()
    result = run_reconcile_eod(as_of=as_of)
    logger.info("reconcile_eod result: {}", result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
