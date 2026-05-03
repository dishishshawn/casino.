"""Deflated Sharpe Ratio (Bailey & Lopez de Prado).

References:
    Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."

Implementation:
    1. variance_inflation V = (1 - skew*SR + ((kurt-1)/4)*SR^2) / (n_obs - 1)
       (we use the Mertens 1991 small-sample variance of the Sharpe estimator)
    2. expected_max_sr E[max SR] under the null is approximated by
       sqrt(2*log(n_trials)) * (1 - gamma/(2*log(n_trials)))
            + gamma / sqrt(2*log(n_trials))
       with gamma = Euler-Mascheroni ≈ 0.5772156649.
    3. deflated_sr = (SR_observed - E[max SR]) / sqrt(V)
    4. p_value = 1 - Φ(deflated_sr)

PRD §6 / §9 / §11: Deflated Sharpe is a release gate. A backtest that doesn't
deflate above zero is overfit; do not "iterate on it" without re-examining
the hypothesis.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_GAMMA: float = 0.5772156649015329  # Euler-Mascheroni


def _phi(x: float) -> float:
    """Standard normal CDF without scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected_max_sr(n_trials: int) -> float:
    """E[max SR] under the null hypothesis of zero-skill across `n_trials`.

    The first-order approximation from Bailey & Lopez de Prado (2014).
    """
    if n_trials <= 1:
        return 0.0
    log_n = math.log(n_trials)
    sqrt_2log_n = math.sqrt(2.0 * log_n)
    return sqrt_2log_n * (1.0 - _GAMMA / (2.0 * log_n)) + _GAMMA / sqrt_2log_n


def deflated_sharpe(
    sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Return the deflated Sharpe ratio.

    Negative values mean the observed SR is consistent with overfitting given
    the trial count. `kurtosis` is *non-excess* (3.0 = normal).
    """
    if n_observations <= 1:
        return float("nan")
    # Mertens variance of SR estimator
    var_sr = (1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2) / (n_observations - 1)
    if var_sr <= 0.0:
        return float("nan")
    e_max = _expected_max_sr(n_trials)
    return (sharpe - e_max) / math.sqrt(var_sr)


def haircut_sharpe(
    observed_sr: float,
    backtest_metadata: dict[str, Any],
) -> dict[str, Any]:
    """High-level wrapper returning {observed, deflated, p_value, is_significant}.

    `backtest_metadata` keys consumed:
        n_trials       (int)         — number of param configurations tested
        n_observations (int)         — trade or daily-return count
        returns        (Sequence)    — optional: returns to compute moments
        skew           (float)       — optional: pre-computed skew
        kurtosis       (float)       — optional: pre-computed *non-excess* kurtosis
    """
    n_trials = int(backtest_metadata.get("n_trials", 1))
    n_obs = int(backtest_metadata.get("n_observations", 0))
    skew_v = backtest_metadata.get("skew")
    kurt_v = backtest_metadata.get("kurtosis")

    if (skew_v is None or kurt_v is None) and "returns" in backtest_metadata:
        returns = pd.Series(backtest_metadata["returns"]).dropna()
        if not returns.empty and len(returns) > 3:
            arr = returns.to_numpy(dtype=float)
            mu = float(arr.mean())
            sigma = float(arr.std(ddof=1))
            if sigma > 0:
                centered = (arr - mu) / sigma
                # population moments (Fisher's g1, g2 not used; use raw moments)
                if skew_v is None:
                    skew_v = float(np.mean(centered**3))
                if kurt_v is None:
                    kurt_v = float(np.mean(centered**4))  # non-excess

    skew_v = float(skew_v) if skew_v is not None else 0.0
    kurt_v = float(kurt_v) if kurt_v is not None else 3.0

    deflated = deflated_sharpe(
        observed_sr,
        n_trials=n_trials,
        n_observations=n_obs,
        skew=skew_v,
        kurtosis=kurt_v,
    )
    if math.isnan(deflated):
        p_value = float("nan")
        is_sig = False
    else:
        p_value = 1.0 - _phi(deflated)
        is_sig = bool(deflated > 0.0 and p_value < 0.05)

    return {
        "observed": observed_sr,
        "deflated": deflated,
        "p_value": p_value,
        "is_significant": is_sig,
        "n_trials": n_trials,
        "n_observations": n_obs,
        "skew": skew_v,
        "kurtosis": kurt_v,
    }


def save_deflated_result(
    result: dict[str, Any],
    *,
    output_dir: Path = Path("backtests/results"),
) -> Path:
    """Persist a deflation result to JSON and return the path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"deflated_sharpe_{ts}.json"
    payload = {
        "computation_timestamp": ts,
        "casino_version": "0.1.0",
        **result,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("deflated sharpe result written to {}", out_path)
    return out_path


def analyze_walkforward_results(wf_results: pd.DataFrame) -> dict[str, Any]:
    """Aggregate a walk-forward DataFrame into a deflated-Sharpe verdict.

    Treats:
        n_trials       = unique selected_params combos (proxy)
        observed_sr    = mean test_sharpe
        n_observations = number of test windows × ~63 (quarterly) trading days
                         is too crude — instead we expect the caller to pass
                         a more accurate trade count via metadata override.
    """
    if wf_results.empty:
        return haircut_sharpe(float("nan"), {"n_trials": 1, "n_observations": 0})
    observed_sr = float(wf_results["test_sharpe"].mean())
    n_unique_params = int(wf_results["selected_params"].astype(str).nunique())
    metadata: dict[str, Any] = {
        "n_trials": max(1, n_unique_params),
        "n_observations": int(len(wf_results) * 63),  # ~quarterly windows
        "returns": wf_results["test_total_return"].dropna().tolist(),
    }
    return haircut_sharpe(observed_sr, metadata)
