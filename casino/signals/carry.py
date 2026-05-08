"""Cross-asset carry signal (Koijen-Moskowitz-Pedersen-Vrugt 2018).

After the TSMOM sensitivity grid placed the chosen blend at only the 55th
percentile of single-lookback variants (memory: tsmom_killed_2026-05-07.md),
Branch C pivots to carry as the lead diversifying signal. Carry is the
*expected* return component of holding an asset to maturity / over the funding
horizon, net of the risk-free rate. Empirically:

    Bond carry      = long-term yield − short-term funding rate (DGS10 − DTB3,
                     DGS5 − DTB3 for IEF, etc.)
    Equity carry    = trailing-12m distribution yield − T-bill
                     (sum of declared dividends / current price − DTB3)
    Commodity carry = roll yield (front-back futures spread).
                     Deferred for v1: GLD/DBC/USO carry returns NaN, and the
                     cross-sectional rank operates on the 7 carry-bearing
                     tickers only.

Per-row weighting matches `casino.signals.ts_momentum.compute_tsmom_panel`:
vol-target sizing on a 60-day realized-vol denominator, gross capped to 1.0,
NaN before sufficient history. The `mode` parameter is the same shape so
ensemble work in tasks 39/40 can swap signals identically.

No look-ahead invariant: row T's weight uses only data with timestamp strictly
before T. The Koijen et al. carry definition is naturally point-in-time —
trailing dividends end at T-1, FRED yields end at T-1, prices use T-1's close.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

_VOL_LOOKBACK_BDAYS: int = 60
_TARGET_VOL_ANN: float = 0.10
_MIN_VOL_ANN: float = 0.05
_BDAYS_PER_YEAR: int = 252
_DIVIDEND_LOOKBACK_BDAYS: int = 252  # trailing 12 months

# Each ETF's bond carry definition: which long-rate FRED series minus which
# short-rate (Tbill). For ETFs without a clean carry definition (gold, broad
# commodities, oil), we leave them out of the rank universe by returning NaN.
_BOND_CARRY_SPEC: dict[str, tuple[str, str]] = {
    "TLT": ("DGS10", "DTB3"),
    "IEF": ("DGS5", "DTB3"),
}

# Equity ETFs whose carry is dividend-yield − Tbill.
_EQUITY_CARRY_TICKERS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA", "EEM")

# Commodity ETFs deferred for v1 (no roll-yield ingest).
_COMMODITY_DEFERRED: frozenset[str] = frozenset({"GLD", "DBC", "USO"})


def _realized_vol(prices: pd.DataFrame, lookback_bdays: int = _VOL_LOOKBACK_BDAYS) -> pd.DataFrame:
    """Annualized realized vol from log-return std (matches TSMOM's denominator)."""
    log_ret = np.log(prices / prices.shift(1))
    vol = log_ret.rolling(window=lookback_bdays, min_periods=lookback_bdays // 2).std()
    return vol * np.sqrt(_BDAYS_PER_YEAR)


def _bond_carry_panel(
    fred_yields: pd.DataFrame,
    *,
    price_index: pd.DatetimeIndex,
    universe: list[str],
) -> pd.DataFrame:
    """Bond carry = long-rate − T-bill, expressed as decimals (e.g. 0.025 = 2.5%).

    Forward-fill onto the price index and shift by one day so row T uses only
    yields observed strictly before T's close.

    `fred_yields` is the wide panel returned by `store.load_fred_panel` with
    values in *percent units* (e.g. 4.25 for 4.25% yield), matching the FRED
    CSV convention. We divide by 100 here.
    """
    out = pd.DataFrame(np.nan, index=price_index, columns=universe, dtype=float)
    if fred_yields.empty:
        return out

    # Normalize fred index to UTC midnight DatetimeIndex on a consistent dtype.
    fy = fred_yields.copy()
    if not isinstance(fy.index, pd.DatetimeIndex):
        fy.index = pd.to_datetime(fy.index)

    # Reindex onto the price grid via merge_asof to be tz-safe across naive/tz indexes.
    p_idx = pd.DatetimeIndex(price_index)
    fy_aligned = (
        fy.reindex(fy.index.union(p_idx).sort_values())
        .ffill()
        .reindex(p_idx)
        .shift(1)  # use yield observed before today's close
    )

    for ticker in universe:
        spec = _BOND_CARRY_SPEC.get(ticker)
        if spec is None:
            continue
        long_id, short_id = spec
        if long_id not in fy_aligned.columns or short_id not in fy_aligned.columns:
            continue
        carry_pct = fy_aligned[long_id] - fy_aligned[short_id]
        out[ticker] = carry_pct / 100.0

    return out


def _equity_carry_panel(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    fred_yields: pd.DataFrame,
    *,
    universe: list[str],
) -> pd.DataFrame:
    """Equity carry = trailing-12m dividend yield − T-bill, decimals.

    Implementation: compute trailing-252-bday sum of dividend amounts per ticker
    on the price grid (forward-filling onto trading days, then summing the
    rolling window), divide by yesterday's close to get a yield, then subtract
    DTB3/100 (also lagged one day).

    `dividends` is long-format with columns (ticker, ts, amount); we pivot
    inline to a per-ticker daily series.
    """
    out = pd.DataFrame(np.nan, index=prices.index, columns=universe, dtype=float)
    if prices.empty:
        return out

    # T-bill series in decimal, lagged one day.
    if not fred_yields.empty and "DTB3" in fred_yields.columns:
        fy = fred_yields.copy()
        if not isinstance(fy.index, pd.DatetimeIndex):
            fy.index = pd.to_datetime(fy.index)
        p_idx = pd.DatetimeIndex(prices.index)
        tbill_pct = (
            fy["DTB3"].reindex(fy.index.union(p_idx).sort_values()).ffill().reindex(p_idx).shift(1)
        )
        tbill = tbill_pct / 100.0
    else:
        tbill = pd.Series(np.nan, index=prices.index)

    # Pivot dividends to wide panel on the price grid. Each cell holds the
    # dividend declared on that trading day (0 elsewhere). NaN stays NaN.
    if not dividends.empty:
        d = dividends.copy()
        d["ts"] = pd.to_datetime(d["ts"])
        # Snap dividend timestamps to the nearest trading day (use date alignment).
        d["date"] = d["ts"].dt.normalize()
        # Floor price index to date so we can index by date.
        price_dates = pd.to_datetime(prices.index).normalize()
        dates_to_pos: dict[pd.Timestamp, int] = {}
        for i, d_ in enumerate(price_dates):
            dates_to_pos.setdefault(pd.Timestamp(d_), i)
        # Build wide divs panel: (date × ticker) of dividend amount, default 0.
        wide_divs = pd.DataFrame(0.0, index=prices.index, columns=universe, dtype=float)
        for ticker, group in d.groupby("ticker"):
            if ticker not in universe:
                continue
            for _, row in group.iterrows():
                snapped = pd.Timestamp(row["date"])
                # Find first price index >= dividend date.
                pos = None
                for k, dt in enumerate(price_dates):
                    if pd.Timestamp(dt) >= snapped:
                        pos = k
                        break
                if pos is None:
                    continue
                wide_divs.iloc[pos, wide_divs.columns.get_loc(ticker)] += float(row["amount"])
    else:
        wide_divs = pd.DataFrame(0.0, index=prices.index, columns=universe, dtype=float)

    trailing_div = wide_divs.rolling(window=_DIVIDEND_LOOKBACK_BDAYS, min_periods=60).sum()
    # Use yesterday's close to compute today's yield (no look-ahead).
    px_lag = prices.shift(1)

    for ticker in universe:
        if ticker not in _EQUITY_CARRY_TICKERS:
            continue
        if ticker not in trailing_div.columns or ticker not in px_lag.columns:
            continue
        div_yield = trailing_div[ticker] / px_lag[ticker]
        out[ticker] = div_yield - tbill

    return out


def _combine_carry(
    bond_carry: pd.DataFrame,
    equity_carry: pd.DataFrame,
) -> pd.DataFrame:
    """Combine per-asset carries into one wide panel, NaN where no definition."""
    cols = list(dict.fromkeys(list(bond_carry.columns) + list(equity_carry.columns)))
    idx = bond_carry.index if not bond_carry.empty else equity_carry.index
    out = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
    for c in cols:
        if c in bond_carry.columns:
            out[c] = out[c].where(~bond_carry[c].notna(), bond_carry[c])
        if c in equity_carry.columns:
            out[c] = out[c].where(~equity_carry[c].notna(), equity_carry[c])
    return out


def _cross_sectional_rank(
    carry: pd.DataFrame,
    *,
    mode: Literal["long_short", "long_only"],
) -> pd.DataFrame:
    """Cross-sectional rank → +1/-1 (long-short top/bottom half) or 0/+1 (long-only top quintile).

    For long_short: long the upper-half ranks, short the lower-half.
    For long_only: long the top quintile (top 20% of carry-bearing tickers).
    """
    out = pd.DataFrame(0.0, index=carry.index, columns=carry.columns, dtype=float)
    for ts, row in carry.iterrows():
        valid = row.dropna()
        if len(valid) < 2:
            # No cross-section to rank against.
            continue
        if mode == "long_short":
            median = float(valid.median())
            for tkr, v in valid.items():
                out.loc[ts, tkr] = 1.0 if v > median else (-1.0 if v < median else 0.0)
        else:  # long_only
            n = len(valid)
            # Top quintile = top 20%; floor of n*0.2, but at least 1.
            k = max(1, int(np.floor(n * 0.2)))
            top = valid.nlargest(k).index
            for tkr in top:
                out.loc[ts, tkr] = 1.0
    return out


def compute_carry_panel(
    prices: pd.DataFrame,
    *,
    dividends: pd.DataFrame,
    fred_yields: pd.DataFrame,
    mode: Literal["long_short", "long_only"] = "long_short",
    target_vol: float = _TARGET_VOL_ANN,
    gross_target: float = 1.0,
) -> pd.DataFrame:
    """Compute the carry target-weight panel.

    Args:
        prices: wide adj-close panel (date × ticker).
        dividends: long-format dividends (columns: ticker, ts, amount).
        fred_yields: wide panel of FRED yield series (values in PERCENT units;
            e.g. 4.25 for 4.25%). Index is timestamps; columns are series IDs
            ('DGS10', 'DGS5', 'DTB3', ...).
        mode: 'long_short' (sign in {-1, 0, +1}) or 'long_only' (top quintile).
        target_vol: per-asset annualized vol target.
        gross_target: cap on per-row sum |w|.

    Returns:
        Wide DataFrame matching `prices.index`, columns ⊆ `prices.columns`.
        NaN where insufficient history; 0.0 where carry is undefined for a
        ticker but the panel is otherwise valid (e.g. GLD/DBC/USO in v1).
    """
    if prices.empty:
        return prices.copy()

    universe = list(prices.columns)

    bond_carry = _bond_carry_panel(
        fred_yields,
        price_index=pd.DatetimeIndex(prices.index),
        universe=universe,
    )
    equity_carry = _equity_carry_panel(
        prices,
        dividends,
        fred_yields,
        universe=universe,
    )
    carry = _combine_carry(bond_carry, equity_carry)
    # Reindex columns to match prices columns (NaN for deferred commodity tickers).
    carry = carry.reindex(columns=universe)

    # Cross-sectional sign / membership.
    sign = _cross_sectional_rank(carry, mode=mode)

    # Vol-target sizing: weight = sign * (target_vol / max(realized_vol, floor)).
    vol = _realized_vol(prices).clip(lower=_MIN_VOL_ANN)
    leverage = (target_vol / vol).clip(upper=3.0)
    weights = sign * leverage

    if mode == "long_only":
        weights = weights.clip(lower=0.0)

    # Renormalize each row so sum |w| <= gross_target.
    abs_sum = weights.abs().sum(axis=1)
    scale = (gross_target / abs_sum).clip(upper=1.0).fillna(0.0)
    weights = weights.mul(scale, axis=0)

    # Burn-in mask: insufficient dividend or vol history → NaN.
    burn_in = max(_VOL_LOOKBACK_BDAYS, _DIVIDEND_LOOKBACK_BDAYS)
    burn_mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for col in prices.columns:
        first_valid = prices[col].first_valid_index()
        if first_valid is None:
            continue
        cutoff_loc = prices.index.get_loc(first_valid)
        cutoff_int = int(cutoff_loc) if isinstance(cutoff_loc, int) else 0
        end_int = min(cutoff_int + burn_in, len(prices.index))
        burn_mask.iloc[cutoff_int:end_int, prices.columns.get_loc(col)] = True
    weights = weights.where(~burn_mask, np.nan)

    return weights
