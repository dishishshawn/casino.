"""Tests for casino.backtest.tsmom_sensitivity (TSMOM grid)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casino.backtest import tsmom_sensitivity as ts


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """3-ticker x 800-bday synthetic price panel with mild momentum + noise.

    Daily log returns: AR(1) component (positive autocorrelation -> momentum)
    plus Gaussian noise. Different drift per ticker.
    """
    rng = np.random.default_rng(seed=2026)
    n_days = 800
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B", tz="UTC")
    tickers = ["AAA", "BBB", "CCC"]
    drifts = [0.0003, 0.0001, 0.0002]

    out = {}
    for t, mu in zip(tickers, drifts, strict=True):
        eps = rng.normal(0.0, 0.012, size=n_days)
        # AR(1) on returns to inject persistent trend.
        rets = np.zeros(n_days)
        prev = 0.0
        for i in range(n_days):
            rets[i] = mu + 0.15 * prev + eps[i]
            prev = rets[i]
        prices = 100.0 * np.exp(np.cumsum(rets))
        out[t] = prices
    return pd.DataFrame(out, index=idx)


def test_run_cell_produces_finite_metrics(synthetic_prices: pd.DataFrame) -> None:
    cell = ts.run_cell(
        synthetic_prices,
        lookbacks=(63,),
        lookback_label="3",
        vol_target=0.10,
        rebalance="monthly",
        cost_bps=12.0,
        n_trials=8,
    )
    assert np.isfinite(cell.sharpe)
    assert np.isfinite(cell.max_drawdown)
    assert np.isfinite(cell.ann_return)
    assert np.isfinite(cell.ann_vol)


def test_reduced_grid_runs_without_nan(synthetic_prices: pd.DataFrame) -> None:
    """Run a reduced 2x2x2 grid and assert all cells are finite."""
    cells = ts.run_grid(
        synthetic_prices,
        cost_bps=12.0,
        lookbacks=((63,), (126,)),
        vol_targets=(0.08, 0.10),
        rebalances=("weekly", "monthly"),
        lookback_labels={(63,): "3", (126,): "6"},
        n_trials=8,
    )
    assert len(cells) == 8
    for cell in cells:
        assert np.isfinite(cell.sharpe), f"NaN Sharpe for cell {cell}"
        assert np.isfinite(cell.max_drawdown), f"NaN MaxDD for cell {cell}"


def test_chosen_percentile_rank_in_unit_interval(
    synthetic_prices: pd.DataFrame,
) -> None:
    cells = ts.run_grid(
        synthetic_prices,
        cost_bps=12.0,
        lookbacks=((63,), (126,)),
        vol_targets=(0.08, 0.10),
        rebalances=("weekly", "monthly"),
        lookback_labels={(63,): "3", (126,): "6"},
        n_trials=8,
    )
    # Chose Sharpe = grid median; percentile rank should be in [0, 1].
    sharpes = [c.sharpe for c in cells]
    median_sharpe = float(np.median(sharpes))
    pct = ts.chosen_percentile_rank(cells, median_sharpe)
    assert 0.0 <= pct <= 1.0


def test_biweekly_rebalance_differs_from_weekly_and_monthly(
    synthetic_prices: pd.DataFrame,
) -> None:
    """Biweekly rebal must produce different metrics than weekly or monthly.

    Same lookback + vol-target across the three; if biweekly were silently
    aliased to weekly or monthly, Sharpes would coincide.
    """
    common = {
        "lookbacks": (63,),
        "lookback_label": "3",
        "vol_target": 0.10,
        "cost_bps": 12.0,
        "n_trials": 8,
    }
    weekly = ts.run_cell(synthetic_prices, rebalance="weekly", **common)
    biweekly = ts.run_cell(synthetic_prices, rebalance="biweekly", **common)
    monthly = ts.run_cell(synthetic_prices, rebalance="monthly", **common)

    assert np.isfinite(biweekly.sharpe)
    # Strict inequality: biweekly must not collapse onto either neighbour.
    assert not np.isclose(biweekly.sharpe, weekly.sharpe, atol=1e-6)
    assert not np.isclose(biweekly.sharpe, monthly.sharpe, atol=1e-6)
