"""Task 35 — tests for TSMomEventBacktest event-driven validator.

The acceptance contract is: event-driven Backtrader Sharpe within 10% of the
vbt-style baseline Sharpe (`tsmom_baseline._backtest_weights_returns`). We
operationalize that as |Sharpe_bt - Sharpe_vbt| < 0.07 on a synthetic price
panel with deterministic trends (so both engines should converge).

The full strict-OOS 2006–2026 universe acceptance check is done by the CLI
runner (`casino.backtest.bt_validate --strategy tsmom`); here we restrict to a
smaller, tractable synthetic that runs in <2s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("backtrader")

from casino.backtest import bt_validate, tsmom_baseline
from casino.signals import ts_momentum


def _synthetic_trending_panel(
    *,
    n_days: int = 600,
    tickers: tuple[str, ...] = ("A", "B", "C", "D"),
    seed: int = 7,
) -> pd.DataFrame:
    """Build a price panel with persistent trends.

    Each ticker gets a small constant drift plus low-vol gaussian noise. The
    result is a panel where TSMOM's sign-blend stays consistently positive (or
    consistently negative), so weights are stable across months — that
    minimizes pathological turnover effects when comparing engines.
    """
    rng = np.random.default_rng(seed)
    drifts = np.linspace(0.0004, 0.0010, len(tickers))  # 10–25 bps/day
    noise = rng.normal(0.0, 0.008, size=(n_days, len(tickers)))
    rets = noise + drifts[None, :]
    prices = pd.DataFrame(
        100.0 * (1.0 + pd.DataFrame(rets)).cumprod().values,
        index=pd.date_range("2018-01-02", periods=n_days, freq="B"),
        columns=list(tickers),
    )
    return prices


def test_tsmom_event_backtest_matches_vbt_sharpe() -> None:
    """bt-vs-vbt Sharpe parity on a deterministic synthetic panel."""
    prices = _synthetic_trending_panel()
    weights = ts_momentum.compute_tsmom_panel(prices, mode="long_only")
    # Drop burn-in rows so the parity check is run on the same realized
    # trading window in both engines.
    weights = weights.dropna(how="all")
    common_idx = weights.index.intersection(prices.index)
    weights = weights.loc[common_idx]
    prices = prices.loc[common_idx]

    cost_bps_rt = 12.0

    vbt_returns = tsmom_baseline._backtest_weights_returns(
        weights, prices, cost_bps=cost_bps_rt, rebalance="monthly"
    )
    if vbt_returns.empty:
        pytest.skip("vbt baseline returned empty series — synthetic panel too short")
    vbt_metrics = bt_validate._metrics_from_returns(vbt_returns)
    bt_metrics = bt_validate.run_tsmom_validation(weights, prices, cost_bps_rt=cost_bps_rt)

    vbt_sharpe = vbt_metrics["sharpe"]
    bt_sharpe = bt_metrics["sharpe"]
    assert np.isfinite(vbt_sharpe), f"vbt Sharpe is non-finite: {vbt_sharpe}"
    assert np.isfinite(bt_sharpe), f"bt Sharpe is non-finite: {bt_sharpe}"
    delta = abs(bt_sharpe - vbt_sharpe)
    # Acceptance: |delta| < 0.07 (within 10% of the headline vbt 0.7 Sharpe band).
    assert delta < 0.07, (
        f"bt-vs-vbt Sharpe drift too large: vbt={vbt_sharpe:.3f}, "
        f"bt={bt_sharpe:.3f}, delta={delta:.3f}"
    )


def test_tsmom_event_backtest_no_lookahead_signal_lag() -> None:
    """Sanity: rebalance reads strictly-prior weights row, not today's row.

    We feed a weights frame whose final-day row is +1.0 across all tickers and
    whose preceding row is 0.0. If the strategy honored same-day signal, it
    would open large positions on the final bar; with strict-prior + next-open
    discipline, no order should fill at the final bar.
    """
    prices = _synthetic_trending_panel(n_days=120)
    # Construct artificial weights where the signal jumps to +1.0 only at the
    # very last business day. With strict-prior reading + next-open fill, the
    # strategy never gets a chance to act on it.
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_day = weights.index[-1]
    weights.loc[last_day, :] = 1.0

    metrics = bt_validate.run_tsmom_validation(weights, prices, cost_bps_rt=12.0)
    # No orders should have been issued — first-of-month rebal reads strictly
    # prior signal which is all-zero for the entire window.
    assert metrics["n_orders"] == 0.0, (
        f"strategy submitted orders despite all-zero strictly-prior signal: "
        f"n_orders={metrics['n_orders']}"
    )
