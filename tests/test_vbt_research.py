"""Tests for casino.backtest.vbt_research."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casino.backtest import vbt_research as vbt_r


@pytest.fixture
def fake_prices() -> pd.DataFrame:
    """100 trading days, 6 tickers. Drift varies per ticker so a top-quintile
    long would outperform a bottom-quintile short."""
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B", tz="UTC")
    drifts = {"A": 0.001, "B": 0.0008, "C": 0.0, "D": 0.0, "E": -0.0008, "F": -0.001}
    cols = {}
    for t, mu in drifts.items():
        rets = rng.normal(mu, 0.01, len(dates))
        cols[t] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(cols, index=dates)


def _const_signal(scores: pd.DataFrame, value: float) -> pd.DataFrame:
    return pd.DataFrame(value, index=scores.index, columns=scores.columns)


def test_quintile_positions_picks_extremes() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    scores = pd.DataFrame(
        {
            "A": [3.0, 3.0],
            "B": [2.0, 2.0],
            "C": [1.0, 1.0],
            "D": [-1.0, -1.0],
            "E": [-2.0, -2.0],
            "F": [-3.0, -3.0],
        },
        index=idx,
    )
    pos = vbt_r._quintile_positions(scores, quintile=0.20, max_per_side=10)
    # one in top quintile (20% of 6 = 1) and one in bottom
    assert pos.iloc[0]["A"] == 1
    assert pos.iloc[0]["F"] == -1
    # middle is zero
    assert pos.iloc[0]["C"] == 0
    assert pos.iloc[0]["D"] == 0


def test_quintile_positions_max_per_side_caps() -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="B", tz="UTC")
    cols = [f"T{i}" for i in range(20)]
    scores = pd.DataFrame([list(range(20))], index=idx, columns=cols, dtype=float)
    pos = vbt_r._quintile_positions(scores, quintile=0.40, max_per_side=3)
    longs = (pos.iloc[0] == 1).sum()
    shorts = (pos.iloc[0] == -1).sum()
    assert longs == 3 and shorts == 3


def test_metrics_from_returns_basic() -> None:
    s = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    m = vbt_r._metrics_from_returns(s)
    assert m["win_rate"] == pytest.approx(3 / 5)
    assert m["sharpe"] == m["sharpe"]  # not nan
    assert m["max_drawdown"] <= 0


def test_run_single_long_top_short_bottom_outperforms(fake_prices: pd.DataFrame) -> None:
    # Signal = forward 5-day returns. A perfect oracle should yield positive Sharpe.
    forward = fake_prices.pct_change(5).shift(-5)
    bt = vbt_r.VectorBacktest(cost_bps=0.0, max_positions_per_side=10, quintile=0.34)
    metrics = bt.run_single(forward.dropna(), fake_prices)
    assert metrics["sharpe"] > 0


def test_transaction_cost_reduces_returns(fake_prices: pd.DataFrame) -> None:
    forward = fake_prices.pct_change(5).shift(-5).dropna()
    no_cost = vbt_r.VectorBacktest(cost_bps=0.0).run_single(forward, fake_prices)
    high_cost = vbt_r.VectorBacktest(cost_bps=200.0).run_single(forward, fake_prices)
    assert high_cost["total_return"] < no_cost["total_return"]


def test_run_parameter_sweep_outputs_csv(tmp_path: Path, fake_prices: pd.DataFrame) -> None:
    def signal_fn(_uni, _start, _end, *, hold: int = 5) -> pd.DataFrame:
        return fake_prices.pct_change(hold).shift(-hold)

    df, csv_path = vbt_r.run_parameter_sweep(
        signal_fn,
        param_grid={"hold": [3, 5, 10]},
        universe=list(fake_prices.columns),
        prices=fake_prices,
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 6, 1, tzinfo=UTC),
        save_results=True,
        output_dir=tmp_path / "results",
    )
    assert len(df) == 3
    assert "sharpe" in df.columns
    assert csv_path is not None
    assert csv_path.exists()
    # sorted descending by sharpe
    assert df["sharpe"].is_monotonic_decreasing or df["sharpe"].isna().any()


def test_run_parameter_sweep_no_save(tmp_path: Path, fake_prices: pd.DataFrame) -> None:
    def signal_fn(_uni, _start, _end, *, hold: int = 5) -> pd.DataFrame:
        return fake_prices.pct_change(hold).shift(-hold)

    df, csv_path = vbt_r.run_parameter_sweep(
        signal_fn,
        param_grid={"hold": [5]},
        universe=list(fake_prices.columns),
        prices=fake_prices,
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 6, 1, tzinfo=UTC),
        save_results=False,
        output_dir=tmp_path / "results",
    )
    assert csv_path is None
    assert len(df) == 1
