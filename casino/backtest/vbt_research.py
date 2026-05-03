"""Vectorbt research backtest harness.

Fast parameter sweeps over signal configurations. Cross-section equal-weight
long-top-quintile / short-bottom-quintile per PRD §5.4, with daily rebalancing
and round-trip transaction costs.

Output: per-config metrics DataFrame; persisted to backtests/results/sweep_<ts>.csv.

Pure backtest module: never imports from `casino.execution.*`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# vectorbt is heavy and pulls many native deps; we type as Any to keep this
# module strict-typecheckable even if vbt's stubs are missing.
try:  # pragma: no cover — import guard
    import vectorbt as vbt
except Exception as e:  # pragma: no cover
    vbt = None
    _vbt_import_error: Exception | None = e
else:
    _vbt_import_error = None


SignalFunc = Callable[..., pd.DataFrame]
"""A signal function: takes (universe, start, end, **params) → wide score DataFrame
(index=dates, columns=tickers)."""


# Default cost: 7 bps round-trip on liquid US equities (CLAUDE.md §9).
_DEFAULT_COST_BPS: float = 7.0
_DEFAULT_MAX_POS_PER_SIDE: int = 10
_DEFAULT_QUINTILE: float = 0.20  # top/bottom 20%


def _quintile_positions(
    scores: pd.DataFrame,
    *,
    quintile: float,
    max_per_side: int,
) -> pd.DataFrame:
    """Convert a wide score DataFrame to a +1 / -1 / 0 position DataFrame.

    For each row (date), pick the top quintile to long and bottom to short,
    capped at `max_per_side` names per side. Equal weighted: positions are
    raw +1 / -1 markers; weighting happens at the portfolio layer.
    """
    out = pd.DataFrame(0, index=scores.index, columns=scores.columns, dtype=np.int8)
    for date, row in scores.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        n = len(valid)
        n_pick = max(1, int(np.floor(n * quintile)))
        n_pick = min(n_pick, max_per_side)
        ranked = valid.sort_values(ascending=False)
        longs = ranked.head(n_pick).index
        shorts = ranked.tail(n_pick).index
        out.loc[date, longs] = 1
        out.loc[date, shorts] = -1
    return out


def _metrics_from_returns(returns: pd.Series) -> dict[str, float]:
    """Compute summary metrics from a daily returns series. Annualization = 252."""
    returns = returns.dropna()
    if returns.empty:
        return {
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "win_rate": float("nan"),
            "total_return": float("nan"),
        }
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    sharpe = float(mu / sigma * np.sqrt(252)) if sigma > 0 else float("nan")
    downside = returns.clip(upper=0).std(ddof=1)
    sortino = float(mu / downside * np.sqrt(252)) if downside and downside > 0 else float("nan")
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1).min()
    win_rate = float((returns > 0).sum() / len(returns))
    total_return = float(cum.iloc[-1] - 1)
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd),
        "win_rate": win_rate,
        "total_return": total_return,
    }


class VectorBacktest:
    """Lightweight vectorized cross-section backtester.

    Wraps vectorbt.Portfolio when available, but the metrics path only needs
    daily portfolio returns, so this class also works in environments where
    vectorbt fails to import (the analytical metrics path is independent).
    """

    def __init__(
        self,
        *,
        cost_bps: float = _DEFAULT_COST_BPS,
        max_positions_per_side: int = _DEFAULT_MAX_POS_PER_SIDE,
        quintile: float = _DEFAULT_QUINTILE,
    ) -> None:
        self.cost_bps = cost_bps
        self.max_positions_per_side = max_positions_per_side
        self.quintile = quintile

    def run_single(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> dict[str, float]:
        """Run one backtest. Returns metrics dict.

        - signals: wide DataFrame of signal scores (index=dates, columns=tickers)
        - prices: wide DataFrame of close prices, same shape & index
        """
        # Align frames
        common_cols = signals.columns.intersection(prices.columns)
        common_idx = signals.index.intersection(prices.index)
        if common_cols.empty or common_idx.empty:
            return _metrics_from_returns(pd.Series(dtype=float))
        s = signals.loc[common_idx, common_cols].sort_index()
        p = prices.loc[common_idx, common_cols].sort_index()

        positions = _quintile_positions(
            s, quintile=self.quintile, max_per_side=self.max_positions_per_side
        )

        # Equal-weighted portfolio: weight = sign / (count of nonzero per row)
        nonzero = positions.abs().sum(axis=1).replace(0, np.nan)
        weights = positions.div(nonzero, axis=0).fillna(0.0)

        # Daily asset returns from close-to-close
        rets = p.pct_change(fill_method=None).fillna(0.0)

        # Apply yesterday's weights to today's returns (no look-ahead)
        port_ret = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)

        # Subtract round-trip transaction costs based on weight turnover
        turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
        # cost_bps is round-trip; turnover is one-way absolute change in weights,
        # so we charge half the bps per unit turnover. 1 bp = 1e-4.
        per_day_cost = turnover * (self.cost_bps / 2.0) * 1e-4
        port_ret = port_ret - per_day_cost

        return _metrics_from_returns(port_ret)


def _expand_grid(param_grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*[param_grid[k] for k in keys]):
        out.append(dict(zip(keys, combo, strict=True)))
    return out


def run_parameter_sweep(
    signal_func: SignalFunc,
    param_grid: Mapping[str, Sequence[Any]],
    *,
    universe: Iterable[str],
    prices: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    cost_bps: float = _DEFAULT_COST_BPS,
    max_positions_per_side: int = _DEFAULT_MAX_POS_PER_SIDE,
    quintile: float = _DEFAULT_QUINTILE,
    save_results: bool = True,
    output_dir: Path = Path("backtests/results"),
) -> tuple[pd.DataFrame, Path | None]:
    """Run a grid search and return (results_df, output_csv_path | None).

    `signal_func(universe, start, end, **params)` must return a wide score
    DataFrame keyed by date × ticker.

    Prices are passed in (rather than re-loaded per config) so that a single
    sweep doesn't re-hit DuckDB per parameter combo.
    """
    universe_list = list(universe)
    configs = _expand_grid(param_grid)
    rows: list[dict[str, Any]] = []
    bt = VectorBacktest(
        cost_bps=cost_bps,
        max_positions_per_side=max_positions_per_side,
        quintile=quintile,
    )
    for params in configs:
        scores = signal_func(universe_list, start_date, end_date, **params)
        metrics = bt.run_single(scores, prices)
        row: dict[str, Any] = {
            **params,
            **metrics,
            "universe_count": len(universe_list),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "cost_bps": cost_bps,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "sharpe" in df.columns:
        df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    out_path: Path | None = None
    if save_results and not df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = output_dir / f"sweep_{ts}.csv"
        df.to_csv(out_path, index=False)
        logger.info("vbt sweep results written to {}", out_path)

    return df, out_path
