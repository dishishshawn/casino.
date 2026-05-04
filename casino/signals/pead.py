"""Baseline PEAD signal: classical Standardized Unexpected Earnings (SUE).

Pure historical-data signal — no LLM, no environment branching. Identical
inputs produce identical outputs in backtest and live (PRD §4.1).

PRD §10 conventions: floats are acceptable for research-only score
computation here. EPS values are accepted as Decimal at the API boundary,
then converted to float for the variance/std math (numpy + scipy can't
operate on Decimal natively).
"""

from __future__ import annotations

import statistics
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TypedDict

import pandas as pd
from loguru import logger

from casino.data import store

# Market-wide fallback std. Tunable; matches "configurable market-wide default
# std (e.g., 0.05)" from task 12.3.
DEFAULT_INDUSTRY_STD: float = 0.05


class EarningsRow(TypedDict, total=False):
    quarter_end: datetime
    actual_eps: float | None
    consensus_eps: float | None
    surprise: float | None


def get_earnings_surprises(
    ticker: str,
    *,
    lookback_quarters: int = 8,
    as_of_date: datetime | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch the most recent `lookback_quarters` EPS surprises for a ticker.

    Point-in-time: only rows with `report_date < as_of_date` are returned, so
    callers can reproduce historical signals without look-ahead leakage.

    Returns a DataFrame with columns: quarter_end, actual_eps, consensus_eps, surprise.
    """
    sql = """
        SELECT report_date, period_end, actual_eps, consensus_eps,
               (actual_eps - consensus_eps) AS surprise
        FROM earnings
        WHERE ticker = ?
          AND consensus_eps IS NOT NULL
          AND actual_eps IS NOT NULL
        {as_of_clause}
        ORDER BY report_date DESC
        LIMIT ?
    """
    params: list[object] = [ticker.upper()]
    if as_of_date is not None:
        sql = sql.format(as_of_clause="AND report_date < ?")
        params.append(as_of_date)
    else:
        sql = sql.format(as_of_clause="")
    params.append(lookback_quarters)

    with store.get_duckdb_conn(db_path, read_only=True) as conn:
        df = conn.execute(sql, params).df()

    if df.empty:
        return pd.DataFrame(columns=["quarter_end", "actual_eps", "consensus_eps", "surprise"])
    df = df.rename(columns={"period_end": "quarter_end"})
    return df[["quarter_end", "actual_eps", "consensus_eps", "surprise"]]


def get_industry_std(
    ticker: str,  # noqa: ARG001 — sector lookup not yet wired (Phase 2)
    as_of_date: datetime,  # noqa: ARG001
    *,
    db_path: Path | None = None,  # noqa: ARG001
) -> float:
    """Return the industry-level surprise std for use as a fallback.

    # Phase 2: GICS sector lookup is not yet ingested. Until then we return the
    # configurable market-wide default (DEFAULT_INDUSTRY_STD), matching the
    # spec's "configurable market-wide default std (e.g., 0.05)" fallback.
    """
    return DEFAULT_INDUSTRY_STD


def compute_sue(
    ticker: str,
    actual_eps: Decimal,
    consensus_eps: Decimal | None,
    as_of_date: datetime,
    *,
    lookback_quarters: int = 8,
    db_path: Path | None = None,
) -> float | None:
    """Compute Standardized Unexpected Earnings.

    Returns None when consensus is missing (we cannot define surprise) or when
    even the industry fallback is unavailable.

    Edge cases:
        * < `lookback_quarters` historical surprises: fall back to industry std.
        * Zero std (degenerate history): fall back to industry std.
        * Missing consensus on this quarter: return None.
    """
    if consensus_eps is None:
        return None

    surprise = float(actual_eps - consensus_eps)
    history = get_earnings_surprises(
        ticker,
        lookback_quarters=lookback_quarters,
        as_of_date=as_of_date,
        db_path=db_path,
    )
    surprises = [float(s) for s in history["surprise"].tolist() if s is not None and not pd.isna(s)]

    std: float
    if len(surprises) >= 2:
        std = statistics.stdev(surprises)
        if std == 0.0:
            std = get_industry_std(ticker, as_of_date, db_path=db_path)
            logger.debug("zero historical std for {}; using industry fallback", ticker)
    else:
        std = get_industry_std(ticker, as_of_date, db_path=db_path)
        logger.debug(
            "insufficient history ({} quarters) for {}; using industry std", len(surprises), ticker
        )

    if std == 0.0:
        return None
    return surprise / std
