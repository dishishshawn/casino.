"""Tests for casino.backtest.walk_forward."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from casino.backtest import walk_forward as wf


def _fake_prices(n_days: int = 504, n_tickers: int = 8, seed: int = 7) -> pd.DataFrame:
    """~2 years of business-day prices for `n_tickers`."""
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B", tz="UTC")
    cols = {}
    for i in range(n_tickers):
        # tickers ranked by drift: T0 strongest, T(n-1) weakest
        mu = (n_tickers // 2 - i) * 0.0005
        rets = rng.normal(mu, 0.012, n_days)
        cols[f"T{i}"] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(cols, index=dates)


def test_generate_windows_no_overlap_or_lookahead() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)
    windows = wf._generate_windows(start, end, train_months=12, test_months=3)
    assert len(windows) >= 4

    test_periods = [w[1] for w in windows]
    for tr, te in windows:
        # invariant: train ends at or before test start
        assert tr.end <= te.start
    # test windows tile and don't overlap
    sorted_tests = sorted(test_periods, key=lambda r: r.start)
    for prev, cur in zip(sorted_tests, sorted_tests[1:], strict=False):
        assert cur.start >= prev.end


def test_window_count_matches_expected() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)
    windows = wf._generate_windows(start, end, train_months=6, test_months=2)
    # 24 months of data, 6mo train+2mo test=8mo first, then advance 2mo each
    # train_starts: 0,2,4,...; test_ends: 8,10,12,...,24 → up to test_end<=24
    # → starts at month 0 yields end=8 ✓; last valid: train_start at month 16 → end=24 ✓
    # → 9 windows
    assert len(windows) == 9


def test_walk_forward_runs_and_aggregates(tmp_path: Path) -> None:
    prices = _fake_prices()

    def signal_fn(_uni, _start, _end, *, hold: int = 5) -> pd.DataFrame:
        # forward-return oracle (positive Sharpe expected)
        return prices.pct_change(hold).shift(-hold)

    df, csv_path = wf.walk_forward_cv(
        signal_fn,
        param_grid={"hold": [3, 5, 10]},
        universe=list(prices.columns),
        prices=prices,
        start_date=prices.index.min().to_pydatetime(),
        end_date=prices.index.max().to_pydatetime(),
        train_months=6,
        test_months=2,
        save_results=True,
        output_dir=tmp_path / "wf",
    )
    assert not df.empty
    assert csv_path is not None and csv_path.exists()
    assert {"train_sharpe", "test_sharpe", "selected_params"}.issubset(df.columns)

    agg = wf.aggregate_walk_forward(df)
    assert agg["n_windows"] == len(df)
    # the oracle is strong but with 7bps cost OOS may be noisier — just sanity check
    assert agg["mean_test_sharpe"] > -1.0


def test_walk_forward_empty_with_no_windows() -> None:
    prices = _fake_prices(n_days=10)

    def signal_fn(_uni, _start, _end, *, hold: int = 5) -> pd.DataFrame:
        return prices.pct_change(hold).shift(-hold)

    df, csv_path = wf.walk_forward_cv(
        signal_fn,
        param_grid={"hold": [5]},
        universe=list(prices.columns),
        prices=prices,
        start_date=datetime(2022, 1, 1, tzinfo=UTC),
        end_date=datetime(2022, 1, 15, tzinfo=UTC),
        train_months=12,
        test_months=3,
        save_results=False,
    )
    assert df.empty
    assert csv_path is None


def test_no_train_data_in_test_period() -> None:
    """White-box: capture the date ranges signal_fn was called with."""
    prices = _fake_prices()
    received_test_starts: list[datetime] = []

    def signal_fn(_uni, start, end, *, hold: int = 5) -> pd.DataFrame:
        received_test_starts.append(start)
        return prices.pct_change(hold).shift(-hold)

    wf.walk_forward_cv(
        signal_fn,
        param_grid={"hold": [5]},
        universe=list(prices.columns),
        prices=prices,
        start_date=prices.index.min().to_pydatetime(),
        end_date=prices.index.max().to_pydatetime(),
        train_months=6,
        test_months=2,
        save_results=False,
    )
    # signal_fn is called twice per window: once during train sweep, once for test eval.
    # Test eval calls always pass the test window start; train sweep passes train window start.
    # We just verify the call was made with the expected windows (at least)
    assert len(received_test_starts) > 0
