"""Time-Series Momentum (TSMOM) signal — Hurst, Ooi, Pedersen 2017.

Per-asset signal on each rebalance date: equal-weighted sign blend of trailing
1m / 3m / 6m / 12m total returns. Scale to target volatility via a 60-day
realized-vol denominator (capped to keep position-sizing finite when an asset
goes to near-zero vol).

The function returns a wide DataFrame (date × asset) of *target weights* in
[-1, +1] suitable for the existing vbt_research sweep harness. Long-only
mode clips negatives to zero (useful when shorting an ETF is operationally
costly even if technically legal).

References:
    * Hurst, Ooi, Pedersen (2017), "A Century of Evidence on Trend-Following Investing"
    * Moskowitz, Ooi, Pedersen (2012), "Time Series Momentum" *JFE*
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from casino.data import store

_LOOKBACKS_BDAYS: tuple[int, ...] = (21, 63, 126, 252)  # 1m, 3m, 6m, 12m
_VOL_LOOKBACK_BDAYS: int = 60
_TARGET_VOL_ANN: float = 0.10  # 10% annualized vol target per asset
_MIN_VOL_ANN: float = 0.05  # floor to prevent infinite leverage on bond-like assets
_BDAYS_PER_YEAR: int = 252


def load_ohlcv_panel(
    *,
    start: datetime,
    end: datetime,
    universe: list[str] | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Pull adj_close panel from DuckDB. Returns wide DataFrame (date × ticker)."""
    where_universe = ""
    params: list[object] = [start, end]
    if universe:
        placeholders = ",".join("?" * len(universe))
        where_universe = f" AND ticker IN ({placeholders})"
        params.extend(universe)
    sql = f"""
        SELECT ticker, ts, COALESCE(adj_close, close) AS close
        FROM ohlcv
        WHERE ts BETWEEN ? AND ? {where_universe}
    """
    with store.get_duckdb_conn(db_path, read_only=True) as conn:
        df = conn.execute(sql, params).df()
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="ts", columns="ticker", values="close").sort_index()


def _trailing_return(prices: pd.DataFrame, lookback_bdays: int) -> pd.DataFrame:
    """Per-cell trailing total return over `lookback_bdays`. NaN where insufficient history."""
    return prices / prices.shift(lookback_bdays) - 1.0


def _realized_vol(prices: pd.DataFrame, lookback_bdays: int = _VOL_LOOKBACK_BDAYS) -> pd.DataFrame:
    """Annualized realized vol from log-return std."""
    log_ret = np.log(prices / prices.shift(1))
    vol = log_ret.rolling(window=lookback_bdays, min_periods=lookback_bdays // 2).std()
    return vol * np.sqrt(_BDAYS_PER_YEAR)


def compute_tsmom_panel(
    prices: pd.DataFrame,
    *,
    lookbacks: tuple[int, ...] = _LOOKBACKS_BDAYS,
    target_vol: float = _TARGET_VOL_ANN,
    mode: Literal["long_short", "long_only"] = "long_short",
) -> pd.DataFrame:
    """Compute the TSMOM target-weight panel.

    Args:
        prices: wide adj-close panel (date × ticker).
        lookbacks: trailing-return windows in business days.
        target_vol: per-asset annualized volatility target.
        mode: "long_short" (sign in {-1,0,+1}) or "long_only" (clipped).

    Returns:
        Same shape as `prices`, with float weights. NaN before history is sufficient.
    """
    if prices.empty:
        return prices.copy()

    # Sign blend across lookbacks, equal-weighted.
    sign_sum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    valid_count = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    for lb in lookbacks:
        ret = _trailing_return(prices, lb)
        s = np.sign(ret).fillna(0.0)
        # Track which cells contributed (had enough history) to avoid biasing the average
        # toward zero when long lookbacks are NaN.
        contributed = (~ret.isna()).astype(int)
        sign_sum = sign_sum + s
        valid_count = valid_count + contributed

    # Average sign in [-1, +1]. Cells with no contributions stay at 0 (mask later).
    avg_sign = sign_sum / valid_count.replace(0, np.nan)

    # Vol-target sizing per asset: weight = target_vol / max(realized_vol, floor).
    vol = _realized_vol(prices).clip(lower=_MIN_VOL_ANN)
    leverage = (target_vol / vol).clip(upper=3.0)  # cap leverage for sanity

    # Final weight panel.
    weights = avg_sign * leverage
    if mode == "long_only":
        weights = weights.clip(lower=0.0)

    # Mask the burn-in period (longest lookback hasn't closed yet for that cell).
    longest = max(lookbacks)
    burn_in_mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for col in prices.columns:
        first_valid = prices[col].first_valid_index()
        if first_valid is None:
            continue
        cutoff_loc = prices.index.get_loc(first_valid)
        cutoff_int = int(cutoff_loc) if isinstance(cutoff_loc, int) else 0
        end_int = min(cutoff_int + longest, len(prices.index))
        burn_in_mask.iloc[cutoff_int:end_int, prices.columns.get_loc(col)] = True
    weights = weights.where(~burn_in_mask, np.nan)
    return weights
