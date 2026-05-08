"""Regime-filtered Time-Series Momentum (TSMOM) — Branch C shadow variant.

Companion to ``casino.signals.ts_momentum.compute_tsmom_panel``. The vanilla
TSMOM panel goes long bonds (TLT, IEF) whenever their trailing trend turns
positive. In the 2022-2024 yield-curve inversion that signal correctly
identified bonds as down-trending and went short, then *whipsawed* repeatedly
as the curve flapped near zero. AQR's "Trend Following in Focus" 2023 update
and the Hurst-Ooi-Pedersen 2024 follow-up both report that conditioning
bond-leg sizing on the level of the yield-curve slope (10y - 3m) materially
improves drawdowns over 2018-2024 without giving back compounded returns.

This module implements that overlay. The decision is intentionally coarse:

    if (DGS10 - DTB3) < slope_threshold:
        bond_legs (TLT, IEF) → 0
    else:
        bond_legs untouched

The slope timestamp is the *most recent* FRED observation as of the close of
trading day T-1. FRED publishes daily but lags the equity close; using T-1's
slope at row T eliminates any ambiguity about look-ahead.

After zeroing the bond legs, weights are re-normalized per row so
``sum(|w|) <= gross_target``. This preserves the gross cap when bonds go to
0 and keeps the equity legs at their original vol-targeted relative weights.

Hard rule from CLAUDE.md: ``signals/*`` must be deterministic with no
environment-dependent branching. The slope-filter logic only consults its
inputs; both unit tests and the live shadow runner pass the same FRED panel
shape. No env-var checks, no timestamp-of-now branches.

The shadow runner (``casino.execution.tsmom_shadow_runner``) is the only
production caller. The module also supports the BT-style baseline
distribution that the shadow's KS-test compares against — that lives in
``casino.backtest.tsmom_regime_baseline``.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from casino.signals.ts_momentum import compute_tsmom_panel

# Default bond legs — universe_tsmom.txt has TLT (20y) and IEF (7-10y).
DEFAULT_BOND_LEGS: tuple[str, ...] = ("TLT", "IEF")

# Default slope threshold. The 10y-3m spread has been negative when curve is
# inverted (a recession-leading-indicator regime); positive in normal regimes.
# Using 0.0 as the threshold (i.e. flatten bonds whenever the curve is
# inverted, not just deeply inverted) matches the AQR 2023 update.
DEFAULT_SLOPE_THRESHOLD: float = 0.0

# Default series IDs from FRED. DGS10 = 10y constant-maturity treasury;
# DTB3 = 3-month T-bill secondary market rate. Both are free, daily, and
# already ingested by ``casino.data.ingest_fred``.
DEFAULT_LONG_RATE: str = "DGS10"
DEFAULT_SHORT_RATE: str = "DTB3"


def compute_tsmom_regime_panel(
    prices: pd.DataFrame,
    fred_yields: pd.DataFrame,
    *,
    lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    target_vol: float = 0.10,
    mode: Literal["long_short", "long_only"] = "long_only",
    gross_target: float = 1.0,
    bond_legs: tuple[str, ...] = DEFAULT_BOND_LEGS,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
    long_rate_series: str = DEFAULT_LONG_RATE,
    short_rate_series: str = DEFAULT_SHORT_RATE,
) -> pd.DataFrame:
    """Compute the regime-filtered TSMOM target-weight panel.

    Args:
        prices: wide adj-close panel (date index x ticker columns), same
            shape ``compute_tsmom_panel`` accepts.
        fred_yields: wide panel (date x series_id) of FRED yield values.
            Must contain at minimum ``long_rate_series`` and
            ``short_rate_series`` columns. Values are yield-in-percent
            (FRED convention; e.g. 4.25 means 4.25%).
        lookbacks: TSMOM lookbacks in business days. Forwarded to vanilla.
        target_vol: per-asset annualized vol target. Forwarded to vanilla.
        mode: ``"long_only"`` for paper-cash account; ``"long_short"`` for
            research. The shadow runner uses ``"long_only"`` to mirror the
            live runner's cash-account constraint.
        gross_target: gross-exposure cap, post re-normalization.
        bond_legs: tickers to flatten when the curve is inverted. Tickers
            absent from ``prices.columns`` are silently ignored (e.g. if
            the user runs the signal on an equities-only universe).
        slope_threshold: flatten bond legs when slope < threshold.
        long_rate_series: FRED series id for the long end of the curve.
        short_rate_series: FRED series id for the short end.

    Returns:
        Same shape as the vanilla ``compute_tsmom_panel`` output. NaN
        burn-in semantics preserved. Where the slope filter fires, bond-
        leg cells are 0.0 and the row is re-normalized so
        ``sum(|w|) <= gross_target``.
    """
    base = compute_tsmom_panel(
        prices,
        lookbacks=lookbacks,
        target_vol=target_vol,
        mode=mode,
        gross_target=gross_target,
    )
    if base.empty:
        return base

    # Build the slope series indexed against ``prices.index``. We take the
    # most-recent slope strictly *before* each row's date — i.e. row T uses
    # slope observable at close T-1. This eliminates any ambiguity about
    # FRED's same-day publication lag.
    slope = _build_lagged_slope(
        target_index=prices.index,
        fred_yields=fred_yields,
        long_rate_series=long_rate_series,
        short_rate_series=short_rate_series,
    )

    # Mask of rows where the regime filter fires.
    inverted_mask = slope < slope_threshold  # NaN compares False (no fire)

    # Tickers to zero. Intersect with actual columns so a caller passing
    # an equities-only universe doesn't crash on a missing TLT/IEF column.
    legs_present = [b for b in bond_legs if b in base.columns]
    if not legs_present or not inverted_mask.any():
        return base.copy()

    out = base.copy()
    # Zero bond legs on inverted rows. CRITICAL: do NOT overwrite vanilla's
    # NaN burn-in cells with 0.0 — the burn-in semantics ("not enough
    # history to score") must be preserved. We only zero cells that were
    # already finite in the vanilla output.
    fire_rows = out.index[inverted_mask.reindex(out.index).fillna(False)]
    if len(fire_rows) > 0:
        for leg in legs_present:
            mask = out.index.isin(fire_rows) & out[leg].notna()
            out.loc[mask, leg] = 0.0

    # Re-normalize so sum(|w|) <= gross_target on those rows. We only touch
    # the rows where we zeroed bonds — equity rows that were already at the
    # gross cap pre-zeroing now have headroom, but we INTENTIONALLY do NOT
    # re-up the equity weights to absorb the freed gross. The hypothesis is
    # "flatten bonds in inverted regimes", not "rotate bond budget into
    # equities". Re-up would introduce a second free parameter.
    #
    # However, if the per-row sum *exceeds* gross_target due to upstream
    # numerical drift, we scale down. This keeps the gross-cap invariant
    # that ``compute_tsmom_panel`` enforces.
    abs_sum = out.loc[fire_rows].abs().sum(axis=1)
    over_cap = abs_sum > gross_target
    if over_cap.any():
        rows_to_scale = fire_rows[over_cap.values]
        scale = gross_target / abs_sum.loc[rows_to_scale]
        out.loc[rows_to_scale] = out.loc[rows_to_scale].mul(scale, axis=0)

    return out


def _build_lagged_slope(
    *,
    target_index: pd.Index,
    fred_yields: pd.DataFrame,
    long_rate_series: str,
    short_rate_series: str,
) -> pd.Series:
    """Construct slope = long - short, lagged one observation, reindexed.

    Returns a Series indexed by ``target_index``. Cells where no prior
    FRED observation exists are NaN; downstream ``< threshold`` comparison
    treats NaN as False (no fire), which is the safe default — without a
    confirmed inverted curve we don't flatten bonds.
    """
    if fred_yields.empty:
        return pd.Series(float("nan"), index=target_index, dtype="float64")
    if long_rate_series not in fred_yields.columns or short_rate_series not in fred_yields.columns:
        return pd.Series(float("nan"), index=target_index, dtype="float64")

    raw = fred_yields[long_rate_series] - fred_yields[short_rate_series]
    raw = raw.dropna().sort_index()
    if raw.empty:
        return pd.Series(float("nan"), index=target_index, dtype="float64")

    # Lag by one OBSERVATION (FRED-publishes-daily). Using shift(1) means at
    # row T we read the slope as of the previous FRED publication, which is
    # always strictly before close T regardless of timezone alignment.
    lagged = raw.shift(1)

    # Make timezone-naive so reindex against an arbitrary equity index works.
    if isinstance(lagged.index, pd.DatetimeIndex) and lagged.index.tz is not None:
        lagged.index = lagged.index.tz_localize(None)

    target = target_index
    if isinstance(target, pd.DatetimeIndex) and target.tz is not None:
        target = target.tz_localize(None)

    # asof-merge: at each target date use the most-recent lagged slope.
    # ``reindex(method="ffill")`` does this when the source is sorted.
    out = lagged.reindex(target, method="ffill")
    out.index = target_index  # restore original tz-aware index for caller
    return out
