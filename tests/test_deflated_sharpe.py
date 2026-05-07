"""Tests for casino.backtest.deflated_sharpe."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from casino.backtest import deflated_sharpe as ds


def test_expected_max_sr_grows_with_trials() -> None:
    e1 = ds._expected_max_sr(10)
    e2 = ds._expected_max_sr(1000)
    e3 = ds._expected_max_sr(100_000)
    assert 0 < e1 < e2 < e3


def test_deflated_sr_penalizes_high_trial_count() -> None:
    sr = 1.5
    low = ds.deflated_sharpe(sr, n_trials=5, n_observations=500)
    high = ds.deflated_sharpe(sr, n_trials=1000, n_observations=500)
    assert low > high


def test_deflated_sr_negative_when_overfit() -> None:
    # Per-period SR=0.05 (≈ annualized 0.79 / sqrt(252)·something) with 1000 trials
    # and only 100 observations: the standardized SR (~0.5σ) is well below the
    # E[max SR null] of ~3.7σ → deflates strongly negative.
    val = ds.deflated_sharpe(0.05, n_trials=1000, n_observations=100)
    assert val < 0


def test_deflated_sr_positive_when_clean() -> None:
    # SR=2.0 with 5 trials and 1000 observations should deflate positive
    val = ds.deflated_sharpe(2.0, n_trials=5, n_observations=1000)
    assert val > 0


def test_haircut_returns_full_record() -> None:
    out = ds.haircut_sharpe(1.5, {"n_trials": 50, "n_observations": 500})
    for k in ("observed", "deflated", "p_value", "is_significant", "n_trials"):
        assert k in out
    assert out["observed"] == 1.5
    assert out["n_trials"] == 50


def test_haircut_with_returns_computes_moments() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 500).tolist()
    out = ds.haircut_sharpe(1.2, {"n_trials": 20, "n_observations": 500, "returns": rets})
    # skew and kurtosis must have been derived from the returns
    assert out["skew"] != 0.0 or out["kurtosis"] != 3.0


def test_phi_matches_known_values() -> None:
    assert math.isclose(ds._phi(0.0), 0.5, abs_tol=1e-6)
    assert math.isclose(ds._phi(1.96), 0.975, abs_tol=1e-3)
    assert math.isclose(ds._phi(-1.96), 0.025, abs_tol=1e-3)


def test_save_deflated_result_writes_json(tmp_path: Path) -> None:
    out = ds.haircut_sharpe(0.8, {"n_trials": 10, "n_observations": 200})
    path = ds.save_deflated_result(out, output_dir=tmp_path / "results")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "deflated" in data
    assert "computation_timestamp" in data
    assert "casino_version" in data


def test_insufficient_observations_returns_nan() -> None:
    val = ds.deflated_sharpe(1.5, n_trials=10, n_observations=1)
    assert math.isnan(val)
