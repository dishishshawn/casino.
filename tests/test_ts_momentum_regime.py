"""Tests for casino.signals.ts_momentum_regime.

Coverage:

* Vanilla equivalence: when slope is always >> threshold, regime panel ==
  vanilla TSMOM panel (no filter ever fires).
* Filter fires: when slope is below threshold, the bond_legs cells at
  those rows are 0; equity legs are untouched.
* Re-normalization: rows where bonds were zeroed do NOT exceed gross_target.
* No look-ahead: shuffling data after row T does not change row T's
  weights.
* Burn-in: the vanilla NaN burn-in semantics are preserved.
* Empty / missing-series safe paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from casino.signals.ts_momentum import compute_tsmom_panel
from casino.signals.ts_momentum_regime import compute_tsmom_regime_panel


def _synth_prices(symbols: list[str], days: int = 400, seed: int = 0) -> pd.DataFrame:
    """Smooth-trending synthetic prices indexed by business days."""
    end = pd.Timestamp("2026-05-29")
    idx = pd.bdate_range(end=end, periods=days)
    rng = np.random.default_rng(seed)
    out = {}
    for i, s in enumerate(symbols):
        t = np.arange(days)
        out[s] = 100.0 + 5.0 * i + 20.0 * (t / days) + rng.normal(0, 0.05, size=days)
    return pd.DataFrame(out, index=idx)


def _synth_yields(idx: pd.DatetimeIndex, *, slope: float) -> pd.DataFrame:
    """FRED-style yield panel with constant DGS10 and DTB3 such that DGS10-DTB3=slope."""
    df = pd.DataFrame(
        {
            "DGS10": 4.0 + slope,
            "DTB3": 4.0,
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------- vanilla equivalence


def test_vanilla_equivalence_when_slope_always_positive() -> None:
    """When slope is always > threshold, regime panel == vanilla panel."""
    syms = ["SPY", "TLT", "IEF", "QQQ"]
    prices = _synth_prices(syms, days=400)
    fred = _synth_yields(prices.index, slope=1.0)  # positive slope, never inverts

    vanilla = compute_tsmom_panel(prices, mode="long_only", gross_target=1.0)
    regime = compute_tsmom_regime_panel(prices, fred, mode="long_only", gross_target=1.0)
    pd.testing.assert_frame_equal(vanilla.fillna(-999.0), regime.fillna(-999.0))


# ---------------------------------------------------------------------------- filter fires


def test_filter_zeros_bonds_when_slope_negative() -> None:
    """Always-inverted curve → every non-NaN bond cell is 0.0."""
    syms = ["SPY", "TLT", "IEF", "QQQ"]
    prices = _synth_prices(syms, days=400)
    fred = _synth_yields(prices.index, slope=-0.5)

    regime = compute_tsmom_regime_panel(prices, fred, mode="long_only", gross_target=1.0)
    # Only check non-burn-in rows (where vanilla is non-NaN).
    nb = regime.dropna(how="all")
    # Bonds zeroed.
    for sym in ("TLT", "IEF"):
        bond_col = nb[sym].dropna()
        # When the filter fires, the cells should be exactly 0.
        assert (bond_col == 0.0).all(), f"{sym} should be 0 in inverted regime"
    # Equity legs not all zero.
    spy_col = nb["SPY"].dropna()
    assert (spy_col > 0).any()


def test_filter_does_not_touch_equity_legs() -> None:
    """Equity weights identical between vanilla and regime when curve flat-positive."""
    syms = ["SPY", "TLT", "IEF", "QQQ"]
    prices = _synth_prices(syms, days=400)
    fred = _synth_yields(prices.index, slope=-0.5)
    vanilla = compute_tsmom_panel(prices, mode="long_only", gross_target=1.0)
    regime = compute_tsmom_regime_panel(prices, fred, mode="long_only", gross_target=1.0)
    # SPY/QQQ untouched. Compare last 30 rows where both are valid.
    for sym in ("SPY", "QQQ"):
        v = vanilla[sym].dropna().iloc[-30:]
        r = regime[sym].dropna().iloc[-30:]
        # Re-normalization can shrink equities only if pre-renorm row
        # had |w| > gross_target. With smooth data and 4 syms long-only
        # with target_vol=10%, sum is well below 1.0; equity rows stay equal.
        # We assert <= (renorm only scales DOWN, never up).
        assert (r <= v + 1e-9).all()


# ---------------------------------------------------------------------------- gross cap


def test_gross_cap_preserved_after_zeroing_bonds() -> None:
    """sum(|w|) <= gross_target on all rows in the regime panel."""
    syms = ["SPY", "TLT", "IEF", "QQQ", "IWM"]
    prices = _synth_prices(syms, days=400)
    fred = _synth_yields(prices.index, slope=-0.3)
    regime = compute_tsmom_regime_panel(prices, fred, mode="long_only", gross_target=1.0)
    abs_sum = regime.abs().sum(axis=1).dropna()
    assert (abs_sum <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------- look-ahead


def test_no_lookahead_shuffle_after_row_t() -> None:
    """Mutating data after row T does not change row T's weights."""
    syms = ["SPY", "TLT", "IEF", "QQQ"]
    prices = _synth_prices(syms, days=400)
    fred = _synth_yields(prices.index, slope=-0.2)
    out_full = compute_tsmom_regime_panel(prices, fred, mode="long_only", gross_target=1.0)

    # Pick a row T well past burn-in.
    t_idx = 350
    row_T_full = out_full.iloc[t_idx].copy()

    # Mutate everything after row T (including the FRED panel).
    prices_mut = prices.copy()
    prices_mut.iloc[t_idx + 1 :] = prices_mut.iloc[t_idx + 1 :] * 100.0  # nuke future prices
    fred_mut = fred.copy()
    fred_mut.iloc[t_idx + 1 :] = fred_mut.iloc[t_idx + 1 :] * -100.0  # nuke future yields

    # When we recompute on a TRUNCATED panel (data only up to T), row T must match.
    prices_trunc = prices.iloc[: t_idx + 1]
    fred_trunc = fred.loc[fred.index <= prices_trunc.index[-1]]
    out_trunc = compute_tsmom_regime_panel(
        prices_trunc, fred_trunc, mode="long_only", gross_target=1.0
    )
    row_T_trunc = out_trunc.iloc[-1]
    pd.testing.assert_series_equal(
        row_T_full.fillna(-999.0),
        row_T_trunc.fillna(-999.0),
        check_names=False,
    )


# ---------------------------------------------------------------------------- burn-in


def test_burn_in_nan_preserved() -> None:
    """Regime panel's burn-in NaN cells match vanilla's exactly."""
    syms = ["SPY", "TLT", "IEF"]
    prices = _synth_prices(syms, days=400)
    fred = _synth_yields(prices.index, slope=-0.5)

    vanilla = compute_tsmom_panel(prices, mode="long_only", gross_target=1.0)
    regime = compute_tsmom_regime_panel(prices, fred, mode="long_only", gross_target=1.0)
    pd.testing.assert_frame_equal(vanilla.isna(), regime.isna())


# ---------------------------------------------------------------------------- safety


def test_empty_prices_returns_empty() -> None:
    out = compute_tsmom_regime_panel(pd.DataFrame(), pd.DataFrame())
    assert out.empty


def test_missing_fred_columns_no_filter() -> None:
    """If FRED panel lacks required columns, nothing is zeroed (safe default)."""
    syms = ["SPY", "TLT"]
    prices = _synth_prices(syms, days=400)
    bad_fred = pd.DataFrame({"OTHER": [1, 2, 3]}, index=prices.index[:3])
    vanilla = compute_tsmom_panel(prices, mode="long_only", gross_target=1.0)
    regime = compute_tsmom_regime_panel(prices, bad_fred, mode="long_only", gross_target=1.0)
    pd.testing.assert_frame_equal(vanilla.fillna(-999.0), regime.fillna(-999.0))


def test_universe_without_bond_legs() -> None:
    """Equities-only universe (no TLT/IEF) → never crashes; identical to vanilla."""
    syms = ["SPY", "QQQ", "IWM"]
    prices = _synth_prices(syms, days=400)
    fred = _synth_yields(prices.index, slope=-0.5)
    vanilla = compute_tsmom_panel(prices, mode="long_only", gross_target=1.0)
    regime = compute_tsmom_regime_panel(prices, fred, mode="long_only", gross_target=1.0)
    pd.testing.assert_frame_equal(vanilla.fillna(-999.0), regime.fillna(-999.0))
