"""Tests for casino.signals.carry — KMPV cross-asset carry."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from casino.signals.carry import compute_carry_panel


def _make_panel(n_days: int = 400) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a deterministic synthetic prices/dividends/fred panel.

    8 ETFs in the universe (matches the spec): SPY, QQQ, IWM, EFA, EEM, TLT,
    IEF, GLD. (We omit DBC/USO to keep the panel small; the carry signal is
    NaN for them anyway.)
    """
    start = datetime(2020, 1, 1, tzinfo=UTC)
    # Use weekday calendar to avoid weekend gaps.
    idx = pd.date_range(start=start, periods=n_days, freq="B", tz="UTC")
    universe = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD"]
    rng = np.random.default_rng(42)

    # Trending prices with stable per-ticker drift so vol/dividend math is stable.
    prices = pd.DataFrame(index=idx, columns=universe, dtype=float)
    for j, t in enumerate(universe):
        rets = rng.normal(0.0003 + j * 1e-5, 0.01, size=n_days)
        prices[t] = 100.0 * np.exp(np.cumsum(rets))

    # Quarterly dividends ~ small for equity ETFs only. Bond ETFs we leave
    # empty; bond carry comes from FRED, not dividends.
    div_rows: list[dict[str, object]] = []
    equity_tickers = ("SPY", "QQQ", "IWM", "EFA", "EEM")
    for t in equity_tickers:
        for q in range(0, n_days, 63):
            ts = idx[q]
            div_rows.append({"ticker": t, "ts": ts.to_pydatetime(), "amount": 0.5})
    dividends = pd.DataFrame(div_rows)

    # FRED yields in PERCENT. DGS10 above DTB3 → positive bond carry.
    fy = pd.DataFrame(index=idx, dtype=float)
    fy["DGS10"] = 4.0
    fy["DGS5"] = 3.5
    fy["DGS2"] = 3.0
    fy["DGS3MO"] = 2.0
    fy["DTB3"] = 2.0

    return prices, dividends, fy


# ----------------------------------------------------------- shape contract
def test_shape_matches_prices() -> None:
    prices, divs, fy = _make_panel()
    out = compute_carry_panel(prices, dividends=divs, fred_yields=fy)
    assert out.shape == prices.shape
    assert list(out.index) == list(prices.index)
    assert set(out.columns) == set(prices.columns)


def test_long_only_has_no_negative_weights() -> None:
    prices, divs, fy = _make_panel()
    out = compute_carry_panel(prices, dividends=divs, fred_yields=fy, mode="long_only")
    # Drop NaN burn-in cells.
    flat = out.to_numpy().ravel()
    flat = flat[~np.isnan(flat)]
    assert (flat >= -1e-12).all()


def test_long_short_produces_both_signs_after_burnin() -> None:
    prices, divs, fy = _make_panel()
    out = compute_carry_panel(prices, dividends=divs, fred_yields=fy, mode="long_short")
    # Past the burn-in window we should have at least one positive and one
    # negative weight cell across the whole panel.
    post = out.iloc[300:].to_numpy().ravel()
    post = post[~np.isnan(post)]
    assert (post > 0).any()
    assert (post < 0).any()


# ----------------------------------------------------------- look-ahead invariant
def test_no_look_ahead_when_future_data_changes() -> None:
    """Row T's weights must be byte-identical when only data at index ≥ T+1 changes.

    The carry signal at row T uses:
      - prices.shift(1)  (yesterday's close)
      - fred_yields.shift(1)  (yesterday's yields)
      - trailing 12m dividends ending at T (which only sees data from <= T)
    Thus changing prices/yields/dividends strictly *after* row T must not
    affect row T's output.
    """
    prices, divs, fy = _make_panel(n_days=500)
    out_orig = compute_carry_panel(prices, dividends=divs, fred_yields=fy)

    # Pick an inspection row in the middle so we have both burn-in headroom
    # and tail data to scramble.
    T = 400
    cutoff_ts = prices.index[T]

    # Permute prices and fred values strictly after the cutoff.
    rng = np.random.default_rng(0)
    p2 = prices.copy()
    p2.iloc[T + 1 :] = rng.normal(150.0, 10.0, size=p2.iloc[T + 1 :].shape)
    fy2 = fy.copy()
    # Make future yields wild.
    future_mask = fy2.index > cutoff_ts
    for col in fy2.columns:
        fy2.loc[future_mask, col] = rng.uniform(0.5, 7.0, size=future_mask.sum())

    # And drop / add dividends after T.
    divs2 = divs[divs["ts"].apply(lambda x: pd.Timestamp(x).tz_convert(UTC) <= cutoff_ts)].copy()
    # Append a giant phony dividend after the cutoff for SPY.
    after_ts = prices.index[T + 5].to_pydatetime()
    divs2 = pd.concat(
        [divs2, pd.DataFrame([{"ticker": "SPY", "ts": after_ts, "amount": 999.0}])],
        ignore_index=True,
    )

    out_perturbed = compute_carry_panel(p2, dividends=divs2, fred_yields=fy2)

    # NOTE: realized vol at row T uses prices up to T, which is unchanged in
    # p2 (only T+1 onward was scrambled). So row T's *exact* weight should be
    # identical.
    row_orig = out_orig.iloc[T]
    row_pert = out_perturbed.iloc[T]
    # NaN-safe equality.
    for col in row_orig.index:
        a, b = row_orig[col], row_pert[col]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == pytest.approx(b, abs=1e-12), f"{col}: {a} vs {b}"


def test_burnin_cells_are_nan() -> None:
    prices, divs, fy = _make_panel(n_days=200)
    out = compute_carry_panel(prices, dividends=divs, fred_yields=fy)
    # First few rows must be NaN since we need 252 days of dividend history.
    assert out.iloc[:50].isna().all().all()


def test_deferred_commodity_tickers_get_no_weight() -> None:
    """DBC/USO must receive 0 weight (or NaN), not be ranked."""
    prices, divs, fy = _make_panel()
    # Inject DBC and USO columns with the same trending series.
    rng = np.random.default_rng(1)
    for extra in ("DBC", "USO"):
        rets = rng.normal(0.0003, 0.01, size=len(prices))
        prices[extra] = 100.0 * np.exp(np.cumsum(rets))
    out = compute_carry_panel(prices, dividends=divs, fred_yields=fy)
    # Past burn-in, DBC/USO must be 0 (no carry definition → not selected).
    post = out.iloc[300:][["DBC", "USO"]]
    flat = post.to_numpy().ravel()
    flat = flat[~np.isnan(flat)]
    assert np.allclose(flat, 0.0, atol=1e-12)


def test_gross_target_respected() -> None:
    prices, divs, fy = _make_panel()
    out = compute_carry_panel(
        prices, dividends=divs, fred_yields=fy, gross_target=1.0, mode="long_short"
    )
    abs_sum = out.abs().sum(axis=1).dropna()
    # Each row's gross must be ≤ 1.0 + tiny float slack.
    assert (abs_sum <= 1.0 + 1e-9).all()


def test_empty_prices_returns_empty() -> None:
    out = compute_carry_panel(
        pd.DataFrame(),
        dividends=pd.DataFrame(columns=["ticker", "ts", "amount"]),
        fred_yields=pd.DataFrame(),
    )
    assert out.empty
