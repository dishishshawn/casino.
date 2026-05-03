"""Pre-close cron entry: score today's reports, submit orders for tomorrow.

PRD §11 Phase 3 daily flow:

1. Load today's earnings tickers (and the prior day's after-hours releases).
2. For each ticker, fetch the transcript and score it via the combined
   SUE × LLM signal in `casino.signals.llm_earnings`.
3. Quintile-select: long the top quintile (positive composites), short the
   bottom quintile (negative composites). Drop the middle.
4. Check the regime gate (`signals.regime.is_risk_on`) — risk-off ⇒ no
   new positions today.
5. For each name in the basket, call `risk.size_position` then
   `risk.submit_order` (broker-side stops, all caps enforced).
6. EOD reconcile (`reconcile.reconcile`) and alert on drift.

Resilience: the entire body is wrapped in a try/except that fires a
critical Discord alert on unhandled exceptions (PRD §10).

Run: ``uv run python -m casino.jobs.earnings_daily``
Cron: see docs/RUNBOOK.md.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from loguru import logger

from casino.config import get_config
from casino.data import store
from casino.execution import reconcile
from casino.execution.alpaca_broker import AlpacaBroker, build_default_broker
from casino.execution.risk import (
    PortfolioState,
    RiskRejection,
    TradingDisabledError,
    snapshot_portfolio_from_broker,
    submit_order,
)
from casino.llm.client import LLMClient
from casino.llm.prompts.earnings_score import TranscriptParts
from casino.monitoring import alerts
from casino.signals import regime
from casino.signals.llm_earnings import (
    DEFAULT_LLM_THRESHOLD,
    DEFAULT_SUE_THRESHOLD,
    CombinedSignal,
    combined_earnings_signal,
)

# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class CandidateRow:
    """One name considered for trading today."""

    ticker: str
    actual_eps: Decimal
    consensus_eps: Decimal | None
    transcript_parts: TranscriptParts
    company_aliases: tuple[str, ...]
    last_close: Decimal


@dataclass(frozen=True)
class JobResult:
    """Summary returned by `run_earnings_daily` (also used by tests)."""

    as_of: datetime
    n_candidates: int
    n_scored: int
    n_long: int
    n_short: int
    n_submitted: int
    n_rejected: int
    risk_on: bool
    drift_alerts: int
    skipped_reason: str | None


# ---------------------------------------------------------------------------- data fetch


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def fetch_todays_earnings(
    *,
    as_of: datetime,
    db_path: Path | None = None,
) -> list[dict[str, object]]:
    """Return earnings rows reported on or in the 24 hours before `as_of`.

    Daily job: catches both the after-hours releases from yesterday and
    today's morning releases. Filters to rows that have an `actual_eps`
    set (i.e. the report has actually printed).
    """
    sql = """
        SELECT ticker, report_date, period_end, actual_eps, consensus_eps
        FROM earnings
        WHERE report_date BETWEEN ? AND ?
          AND actual_eps IS NOT NULL
        ORDER BY ticker
    """
    from datetime import timedelta  # local import keeps module top short

    start = as_of - timedelta(days=1)
    with store.get_duckdb_conn(db_path, read_only=False) as conn:
        rows = conn.execute(sql, [start, as_of]).fetchall()
    return [
        {
            "ticker": str(r[0]),
            "report_date": r[1],
            "period_end": r[2],
            "actual_eps": Decimal(str(r[3])),
            "consensus_eps": Decimal(str(r[4])) if r[4] is not None else None,
        }
        for r in rows
    ]


def fetch_transcript(
    ticker: str,
    *,
    as_of: datetime,
    db_path: Path | None = None,
) -> TranscriptParts | None:
    """Load the most recent transcript on or before `as_of` for `ticker`.

    Returns None if no transcript is available — callers skip the name.
    For Phase 3 the body is split heuristically; the full prepared/Q&A
    split happens in `casino.data.ingest_transcripts`.
    """
    sql = """
        SELECT transcript_text
        FROM transcripts
        WHERE ticker = ?
          AND event_date <= ?
        ORDER BY event_date DESC
        LIMIT 1
    """
    with store.get_duckdb_conn(db_path, read_only=False) as conn:
        rows = conn.execute(sql, [ticker.upper(), as_of]).fetchall()
    if not rows:
        return None
    text = str(rows[0][0] or "")
    if not text.strip():
        return None
    # Heuristic split on the first "Q&A" marker; falls back to a 60/40 cut.
    lower = text.lower()
    for marker in ("question-and-answer", "q&a", "questions and answers", "operator"):
        idx = lower.find(marker)
        if idx > 200:
            return TranscriptParts(prepared_remarks=text[:idx], qa_session=text[idx:])
    cut = int(len(text) * 0.6)
    return TranscriptParts(prepared_remarks=text[:cut], qa_session=text[cut:])


def fetch_last_close(
    ticker: str,
    *,
    as_of: datetime,
    db_path: Path | None = None,
) -> Decimal | None:
    """Return the most recent adj_close (or close) on or before `as_of`."""
    sql = """
        SELECT COALESCE(adj_close, close)
        FROM ohlcv
        WHERE ticker = ? AND ts <= ?
        ORDER BY ts DESC
        LIMIT 1
    """
    with store.get_duckdb_conn(db_path, read_only=False) as conn:
        rows = conn.execute(sql, [ticker.upper(), as_of]).fetchall()
    if not rows or rows[0][0] is None:
        return None
    return Decimal(str(rows[0][0]))


# ---------------------------------------------------------------------------- selection


def quintile_select(
    signals: Sequence[CombinedSignal],
    *,
    fraction: float = 0.2,
) -> tuple[list[CombinedSignal], list[CombinedSignal]]:
    """Return (longs, shorts) by quintile of `combined`.

    Only `traded=True` names are considered. Top quintile (highest combined)
    becomes the long basket; bottom quintile becomes the short basket. The
    middle 60% is dropped (PRD §5.3 — trade only the tails).
    """
    eligible = [s for s in signals if s.traded]
    if not eligible:
        return [], []
    eligible_sorted = sorted(eligible, key=lambda s: s.combined)
    n = len(eligible_sorted)
    k = max(1, int(n * fraction))
    shorts = [s for s in eligible_sorted[:k] if s.combined < 0]
    longs = [s for s in eligible_sorted[-k:] if s.combined > 0]
    return longs, shorts


def stop_price_for(
    *,
    side: str,
    entry: Decimal,
    stop_pct: Decimal = Decimal("0.05"),
) -> Decimal:
    """Compute a 5%-from-entry stop. Long: stop below; short: stop above.

    Phase 3 uses a flat 5% stop; Phase 4 will swap in ATR-based stops.
    """
    if side == "buy":
        return entry * (Decimal("1") - stop_pct)
    return entry * (Decimal("1") + stop_pct)


# ---------------------------------------------------------------------------- main flow


def _build_candidates(
    earnings: list[dict[str, object]],
    *,
    as_of: datetime,
    db_path: Path | None,
) -> list[CandidateRow]:
    out: list[CandidateRow] = []
    for row in earnings:
        ticker_obj = row["ticker"]
        if not isinstance(ticker_obj, str):
            continue
        ticker = ticker_obj
        actual_eps = row["actual_eps"]
        consensus = row["consensus_eps"]
        if not isinstance(actual_eps, Decimal):
            continue
        consensus_eps = consensus if isinstance(consensus, Decimal) else None
        parts = fetch_transcript(ticker, as_of=as_of, db_path=db_path)
        if parts is None:
            logger.debug("earnings_daily: no transcript for {}; skipping", ticker)
            continue
        last_close = fetch_last_close(ticker, as_of=as_of, db_path=db_path)
        if last_close is None or last_close <= Decimal("0"):
            logger.debug("earnings_daily: no close for {}; skipping", ticker)
            continue
        out.append(
            CandidateRow(
                ticker=ticker,
                actual_eps=actual_eps,
                consensus_eps=consensus_eps,
                transcript_parts=parts,
                company_aliases=(),
                last_close=last_close,
            )
        )
    return out


def _score_candidates(
    candidates: list[CandidateRow],
    *,
    client: LLMClient,
    as_of: datetime,
    db_path: Path | None,
    sue_threshold: float,
    llm_threshold: float,
) -> list[CombinedSignal]:
    out: list[CombinedSignal] = []
    for c in candidates:
        try:
            sig = combined_earnings_signal(
                client=client,
                ticker=c.ticker,
                actual_eps=c.actual_eps,
                consensus_eps=c.consensus_eps,
                transcript_parts=c.transcript_parts,
                as_of_date=as_of,
                company_aliases=c.company_aliases,
                sue_threshold=sue_threshold,
                llm_threshold=llm_threshold,
                db_path=db_path,
            )
        except Exception as e:  # noqa: BLE001 — one bad row must not kill the basket
            logger.warning("earnings_daily: scoring failed for {}: {}", c.ticker, e)
            continue
        out.append(sig)
    return out


def _submit_basket(
    *,
    longs: Sequence[CombinedSignal],
    shorts: Sequence[CombinedSignal],
    candidates_by_ticker: dict[str, CandidateRow],
    broker: AlpacaBroker,
    portfolio: PortfolioState,
    db_path: Path | None,
) -> tuple[int, int]:
    """Submit each basket name. Returns (n_submitted, n_rejected)."""
    submitted = 0
    rejected = 0
    for sig in longs:
        c = candidates_by_ticker.get(sig.ticker)
        if c is None:
            continue
        entry = c.last_close
        try:
            order = submit_order(
                broker=broker,
                symbol=sig.ticker,
                side="buy",
                entry_price=entry,
                stop_price=stop_price_for(side="buy", entry=entry),
                portfolio=portfolio,
                db_path=db_path,
            )
            submitted += 1
            alerts.alert_order_fill(
                symbol=sig.ticker,
                side="buy",
                qty=order.qty,
                price=order.filled_avg_price or entry,
                order_id=order.id,
            )
        except (RiskRejection, TradingDisabledError, ValueError) as e:
            logger.warning("earnings_daily: long {} rejected: {}", sig.ticker, e)
            rejected += 1

    for sig in shorts:
        c = candidates_by_ticker.get(sig.ticker)
        if c is None:
            continue
        entry = c.last_close
        try:
            order = submit_order(
                broker=broker,
                symbol=sig.ticker,
                side="sell",
                entry_price=entry,
                stop_price=stop_price_for(side="sell", entry=entry),
                portfolio=portfolio,
                db_path=db_path,
            )
            submitted += 1
            alerts.alert_order_fill(
                symbol=sig.ticker,
                side="sell",
                qty=order.qty,
                price=order.filled_avg_price or entry,
                order_id=order.id,
            )
        except (RiskRejection, TradingDisabledError, ValueError) as e:
            logger.warning("earnings_daily: short {} rejected: {}", sig.ticker, e)
            rejected += 1
    return submitted, rejected


def run_earnings_daily(
    *,
    client: LLMClient | None = None,
    broker: AlpacaBroker | None = None,
    as_of: datetime | None = None,
    db_path: Path | None = None,
    state_path: Path | None = None,
    sue_threshold: float = DEFAULT_SUE_THRESHOLD,
    llm_threshold: float = DEFAULT_LLM_THRESHOLD,
) -> JobResult:
    """End-to-end daily earnings run. Returns a structured `JobResult`.

    `db_path` is the DuckDB path (market data); `state_path` is the
    SQLite path (orders, book, audit). Both default to the values in
    `casino.config`.

    Always uses `LLMClient(mode="live", ...)` — the daily job is never a
    backtest. PRD §6 / CLAUDE.md: anonymization in live mode is a no-op
    on entities; the LLMClient still scrubs per-call dates.
    """
    as_of = as_of if as_of is not None else _utc_now()
    cfg = get_config()
    if state_path is None:
        state_path = cfg.state_sqlite_path
    if client is None:
        client = LLMClient(
            mode="live",
            audit_db_path=state_path,
        )
    if broker is None:
        broker = build_default_broker()

    try:
        # 1) candidates
        earnings = fetch_todays_earnings(as_of=as_of, db_path=db_path)
        if not earnings:
            return JobResult(
                as_of=as_of,
                n_candidates=0,
                n_scored=0,
                n_long=0,
                n_short=0,
                n_submitted=0,
                n_rejected=0,
                risk_on=False,
                drift_alerts=0,
                skipped_reason="no earnings rows for window",
            )
        candidates = _build_candidates(earnings, as_of=as_of, db_path=db_path)
        candidates_by_ticker = {c.ticker: c for c in candidates}

        # 2) regime gate
        risk_on = regime.is_risk_on(as_of=as_of, db_path=db_path)
        if not risk_on:
            logger.warning("earnings_daily: regime risk-off; no new orders")
            return JobResult(
                as_of=as_of,
                n_candidates=len(candidates),
                n_scored=0,
                n_long=0,
                n_short=0,
                n_submitted=0,
                n_rejected=0,
                risk_on=False,
                drift_alerts=0,
                skipped_reason="regime risk-off",
            )

        # 3) score + select
        signals = _score_candidates(
            candidates,
            client=client,
            as_of=as_of,
            db_path=db_path,
            sue_threshold=sue_threshold,
            llm_threshold=llm_threshold,
        )
        longs, shorts = quintile_select(signals)

        # 4) submit
        portfolio = snapshot_portfolio_from_broker(broker)
        submitted, rejected = _submit_basket(
            longs=longs,
            shorts=shorts,
            candidates_by_ticker=candidates_by_ticker,
            broker=broker,
            portfolio=portfolio,
            db_path=state_path,
        )

        # 5) EOD reconcile
        recon = reconcile.reconcile(broker=broker, db_path=state_path)
        critical = reconcile.critical_drift(recon)
        if critical:
            alerts.alert_reconciliation_drift(
                n_drift=len(critical),
                summary="; ".join(d.detail for d in critical),
            )

        return JobResult(
            as_of=as_of,
            n_candidates=len(candidates),
            n_scored=len(signals),
            n_long=len(longs),
            n_short=len(shorts),
            n_submitted=submitted,
            n_rejected=rejected,
            risk_on=True,
            drift_alerts=len(critical),
            skipped_reason=None,
        )
    except Exception as e:  # noqa: BLE001 — alert + re-raise
        alerts.alert_unhandled_exception(
            job="earnings_daily",
            exc_type=type(e).__name__,
            detail=str(e),
        )
        logger.exception("earnings_daily: unhandled exception")
        raise


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.jobs.earnings_daily",
        description="Score today's earnings and submit the long/short basket.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="ISO timestamp for the reporting day (defaults to UTC now).",
    )
    args = parser.parse_args(argv)
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else _utc_now()
    result = run_earnings_daily(as_of=as_of)
    logger.info("earnings_daily result: {}", result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
