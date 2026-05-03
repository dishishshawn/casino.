"""Market regime filter (e.g., SPY > 200-day MA gating).

A simple binary risk-on / risk-off gate consulted by the daily earnings job
before placing new long/short baskets. The default rule mirrors PRD §5.4:
SPY closing price above its 200-day simple moving average → risk-on.

The filter is intentionally cheap to evaluate (one DuckDB query, one mean)
and fully deterministic: identical inputs in backtest and live (CLAUDE.md
§4.1, "signals must be callable from both backtest and live-trading paths
with identical inputs producing identical outputs").

PRD §10 conventions: floats are fine here — this is a research-time score,
not money. Times are UTC end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from casino.data import store


@dataclass(frozen=True)
class RegimeState:
    """Output of `evaluate_regime`.

    `risk_on` is the trading-gate boolean callers consult; the other fields
    exist so the dashboard can show *why* the gate is open or closed.
    """

    as_of: datetime
    benchmark_ticker: str
    benchmark_close: float | None
    moving_average: float | None
    window: int
    risk_on: bool
    reason: str


_DEFAULT_BENCHMARK: str = "SPY"
_DEFAULT_WINDOW: int = 200


def _fetch_closes(
    ticker: str,
    *,
    as_of: datetime,
    window: int,
    db_path: Path | None,
) -> list[float]:
    """Return the most recent `window` daily close prices on or before `as_of`.

    Uses adj_close when available (split/dividend adjusted), otherwise close.
    Strict point-in-time: rows with `ts > as_of` are excluded.
    """
    sql = """
        SELECT COALESCE(adj_close, close) AS px
        FROM ohlcv
        WHERE ticker = ?
          AND ts <= ?
          AND COALESCE(adj_close, close) IS NOT NULL
        ORDER BY ts DESC
        LIMIT ?
    """
    with store.get_duckdb_conn(db_path, read_only=False) as conn:
        rows = conn.execute(sql, [ticker.upper(), as_of, window]).fetchall()
    return [float(r[0]) for r in rows]


def evaluate_regime(
    *,
    as_of: datetime,
    benchmark_ticker: str = _DEFAULT_BENCHMARK,
    window: int = _DEFAULT_WINDOW,
    db_path: Path | None = None,
) -> RegimeState:
    """Compute the regime state at `as_of`.

    Default rule (PRD §5.4): risk-on iff `benchmark_close >= moving_average`.
    Fails closed: when we don't have enough history, `risk_on` is False so
    the daily job won't enter new positions on a missing-data day.
    """
    closes = _fetch_closes(
        benchmark_ticker,
        as_of=as_of,
        window=window,
        db_path=db_path,
    )
    if len(closes) < window:
        logger.warning(
            "regime: insufficient history for {} ({} of {} bars before {}); defaulting risk-off",
            benchmark_ticker,
            len(closes),
            window,
            as_of.isoformat(),
        )
        return RegimeState(
            as_of=as_of,
            benchmark_ticker=benchmark_ticker.upper(),
            benchmark_close=closes[0] if closes else None,
            moving_average=None,
            window=window,
            risk_on=False,
            reason=f"insufficient history ({len(closes)}/{window})",
        )

    last_close = closes[0]
    moving_average = sum(closes) / float(window)
    risk_on = last_close >= moving_average
    reason = (
        f"{benchmark_ticker.upper()} {last_close:.4f} "
        f"{'>=' if risk_on else '<'} {window}-day MA {moving_average:.4f}"
    )
    logger.debug("regime: {}", reason)
    return RegimeState(
        as_of=as_of,
        benchmark_ticker=benchmark_ticker.upper(),
        benchmark_close=last_close,
        moving_average=moving_average,
        window=window,
        risk_on=risk_on,
        reason=reason,
    )


def is_risk_on(
    *,
    as_of: datetime,
    benchmark_ticker: str = _DEFAULT_BENCHMARK,
    window: int = _DEFAULT_WINDOW,
    db_path: Path | None = None,
) -> bool:
    """Convenience wrapper: return the boolean gate only.

    Identical semantics to `evaluate_regime(...).risk_on`. Provided so jobs
    that don't need the structured result can stay terse.
    """
    return evaluate_regime(
        as_of=as_of,
        benchmark_ticker=benchmark_ticker,
        window=window,
        db_path=db_path,
    ).risk_on
