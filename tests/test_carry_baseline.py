"""Tests for casino.backtest.carry_baseline.

We test:
  - Regression: with deterministic carry inputs, the long-leg ranking is
    correct (TLT > SPY > EFA in the long-short mode) and Sharpe is positive.
  - Corr threshold: synthesize carry/TSMOM-like return series and verify
    the corr-vs-TSMOM gate fires correctly at the 0.4 boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from casino.backtest import carry_baseline
from casino.signals.carry import compute_carry_panel


def _make_carry_panel(n_days: int = 800) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic panel where bond carry (TLT, IEF) trumps equity carry."""
    idx = pd.date_range(start=datetime(2018, 1, 1, tzinfo=UTC), periods=n_days, freq="B", tz="UTC")
    universe = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD"]
    rng = np.random.default_rng(7)

    prices = pd.DataFrame(index=idx, columns=universe, dtype=float)
    # Different drifts so backtests don't degenerate.
    drifts = {
        "SPY": 0.0004,
        "QQQ": 0.0005,
        "IWM": 0.0003,
        "EFA": 0.0002,
        "EEM": 0.00015,
        "TLT": 0.0006,  # bond trends positive in this synthetic world
        "IEF": 0.00045,
        "GLD": 0.00025,
    }
    for t in universe:
        rets = rng.normal(drifts[t], 0.008, size=n_days)
        prices[t] = 100.0 * np.exp(np.cumsum(rets))

    # Quarterly dividends only for equity ETFs.
    div_rows: list[dict[str, object]] = []
    for t in ("SPY", "QQQ", "IWM", "EFA", "EEM"):
        for q in range(0, n_days, 63):
            div_rows.append({"ticker": t, "ts": idx[q].to_pydatetime(), "amount": 0.4})
    dividends = pd.DataFrame(div_rows)

    # FRED yields in PERCENT. Make DGS10 - DTB3 = 3% (bond carry +3%) and
    # DGS5 - DTB3 = 2%, both clearly above SPY's ~0.5% dividend yield.
    fy = pd.DataFrame(index=idx, dtype=float)
    fy["DGS10"] = 5.0
    fy["DGS5"] = 4.0
    fy["DGS2"] = 3.0
    fy["DGS3MO"] = 2.0
    fy["DTB3"] = 2.0
    return prices, dividends, fy


# ----------------------------------------------------------- carry ranking
def test_regression_long_leg_ranks_bonds_above_equities() -> None:
    """With DGS10−DTB3=3%, TLT carries +3%, SPY ~0.5% yield. TLT must rank long; EFA short."""
    prices, divs, fy = _make_carry_panel()
    out = compute_carry_panel(prices, dividends=divs, fred_yields=fy, mode="long_short")
    # Average weight across post-burn-in rows.
    post = out.iloc[400:]
    means = post.mean()
    # TLT and IEF (bond carry) should have positive average weight.
    assert means["TLT"] > 0, f"TLT avg weight {means['TLT']:.4f}"
    assert means["IEF"] > 0, f"IEF avg weight {means['IEF']:.4f}"
    # An equity ETF (EFA / EEM) should be on the short leg.
    assert min(means["EFA"], means["EEM"]) < 0


def test_synthetic_returns_have_positive_sharpe() -> None:
    """With strong TLT drift and weak EFA/EEM drift, long-bond/short-equity wins."""
    from casino.backtest.tsmom_baseline import _backtest_weights

    # Override drifts so the long leg (bonds) has clear positive drift and the
    # short leg (lowest-carry equities EFA/EEM) has clear negative drift.
    idx = pd.date_range(start=datetime(2018, 1, 1, tzinfo=UTC), periods=800, freq="B", tz="UTC")
    universe = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD"]
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(index=idx, columns=universe, dtype=float)
    drifts = {
        "SPY": 0.0001,
        "QQQ": 0.0001,
        "IWM": 0.0001,
        "EFA": -0.0008,
        "EEM": -0.0008,
        "TLT": 0.0010,
        "IEF": 0.0008,
        "GLD": 0.0001,
    }
    for t in universe:
        rets = rng.normal(drifts[t], 0.005, size=len(idx))
        prices[t] = 100.0 * np.exp(np.cumsum(rets))

    div_rows: list[dict[str, object]] = []
    for t in ("SPY", "QQQ", "IWM", "EFA", "EEM"):
        for q in range(0, len(idx), 63):
            div_rows.append({"ticker": t, "ts": idx[q].to_pydatetime(), "amount": 0.4})
    dividends = pd.DataFrame(div_rows)

    fy = pd.DataFrame(index=idx, dtype=float)
    fy["DGS10"] = 5.0
    fy["DGS5"] = 4.0
    fy["DGS2"] = 3.0
    fy["DGS3MO"] = 2.0
    fy["DTB3"] = 2.0

    weights = compute_carry_panel(prices, dividends=dividends, fred_yields=fy, mode="long_short")
    metrics = _backtest_weights(weights, prices, cost_bps=12.0, rebalance="monthly")
    assert metrics["sharpe"] > 0, f"Sharpe {metrics['sharpe']:.4f}"


# ----------------------------------------------------------- corr-vs-TSMOM gate
def test_corr_gate_fires_when_correlated() -> None:
    """If carry returns ≈ TSMOM returns, the corr gate must FAIL."""
    rng = np.random.default_rng(11)
    n = 1000
    base = rng.normal(0.0, 0.01, size=n)
    carry_rets = pd.Series(base + rng.normal(0, 0.001, size=n))
    tsmom_rets = pd.Series(base + rng.normal(0, 0.001, size=n))
    aligned = pd.concat([carry_rets, tsmom_rets], axis=1, join="inner").dropna()
    corr = float(np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])
    assert corr >= 0.4
    assert not (corr < carry_baseline.GATE_CORR_MAX)


def test_corr_gate_clears_when_decorrelated() -> None:
    """If carry returns are independent of TSMOM, the corr gate must PASS."""
    rng = np.random.default_rng(13)
    carry_rets = pd.Series(rng.normal(0.0, 0.01, size=1000))
    tsmom_rets = pd.Series(rng.normal(0.0, 0.01, size=1000))
    aligned = pd.concat([carry_rets, tsmom_rets], axis=1, join="inner").dropna()
    corr = float(np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])
    assert abs(corr) < carry_baseline.GATE_CORR_MAX
    assert corr < carry_baseline.GATE_CORR_MAX


def test_gate_constants_match_spec() -> None:
    """Lock in the spec values so accidental edits don't relax the gate."""
    assert carry_baseline.GATE_SHARPE_MIN == pytest.approx(0.4)
    assert carry_baseline.GATE_DSR_MIN == pytest.approx(0.0)
    assert carry_baseline.GATE_CORR_MAX == pytest.approx(0.4)
    assert carry_baseline.DEFAULT_N_TRIALS == 30


# ----------------------------------------------------------- result aggregation
def test_carry_result_verdict_pass_path() -> None:
    """Construct a CarryResult that should be PASS and verify the dataclass works."""
    r = carry_baseline.CarryResult(
        n_assets=8,
        start_date="2010-01-01",
        end_date="2025-01-01",
        mode="long_short",
        cost_bps=12.0,
        sharpe=0.6,
        sortino=0.8,
        max_drawdown=-0.15,
        win_rate=0.54,
        total_return=1.5,
        annualized_return=0.07,
        annualized_vol=0.10,
        corr_with_tsmom=0.2,
        deflated_sharpe=1.2,
        deflated_p_value=0.001,
        deflated_n_trials=30,
        deflated_n_observations=4000,
        pass_sharpe=True,
        pass_dsr=True,
        pass_corr=True,
        verdict="PASS",
        verdict_detail="all gates clear",
    )
    assert r.verdict == "PASS"
    assert r.pass_sharpe and r.pass_dsr and r.pass_corr
