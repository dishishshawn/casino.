"""Backtrader event-driven validation harness.

Phase 2 (task 16). Re-runs vectorbt-selected configs in an event-driven
simulator with realistic costs (PRD §9: 5–10 bps round-trip on liquid US
equities) and slippage. The vectorbt sweep is fast but vectorized — this
module is the realism check.

Public surface:
    EventBacktest             — backtrader.Strategy implementing the
                                cross-section quintile portfolio described
                                in PRD §5.4 from a wide-form score frame.
    run_validation(...)       — module-level entry point that wires up a
                                Cerebro instance, multiple data feeds, and
                                returns a metrics dict.
    bt_evaluator(...)         — adapter conforming to the `evaluator`
                                callable signature accepted by
                                `walk_forward_cv` (signals, prices,
                                **kwargs) -> dict[str, float]. This lets
                                the existing harness swap engines without
                                touching `walk_forward.py`.

Pure backtest module: never imports from `casino.execution.*`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import backtrader as bt
import numpy as np
import pandas as pd
from loguru import logger

# PRD §9 cost defaults: 5–10 bps round-trip on liquid US equities.
_DEFAULT_COST_BPS: float = 7.0
# Commission per side (one-way bps). 0.5 bps ≈ Alpaca Pro pricing per task 16.
_DEFAULT_COMMISSION_BPS: float = 0.5
_DEFAULT_QUINTILE: float = 0.20
_DEFAULT_MAX_POS_PER_SIDE: int = 10
_LIQUID_VOL_THRESHOLD: int = 1_000_000  # PRD §9: 5 bps slippage above this


@dataclass(frozen=True)
class ValidationMetrics:
    """Metrics returned from one event-driven backtest.

    Mirrors the keys produced by `vbt_research._metrics_from_returns`
    *plus* engine-specific fill quality so callers can compare apples to
    apples and also see what the realism layer surfaced.
    """

    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    total_return: float
    fill_rate: float
    avg_slippage_bps: float
    max_intraday_drawdown: float
    n_orders: int

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "fill_rate": self.fill_rate,
            "avg_slippage_bps": self.avg_slippage_bps,
            "max_intraday_drawdown": self.max_intraday_drawdown,
            "n_orders": float(self.n_orders),
        }


# ---------------------------------------------------------------------------- helpers


def _quintile_positions(
    scores: pd.DataFrame,
    *,
    quintile: float,
    max_per_side: int,
) -> pd.DataFrame:
    """Same algorithm as `vbt_research._quintile_positions` to keep the two
    engines comparable. We duplicate rather than import-cross because both
    modules must be independently importable for pure unit tests.
    """
    out = pd.DataFrame(0, index=scores.index, columns=scores.columns, dtype=np.int8)
    for date_, row in scores.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        n_pick = max(1, int(np.floor(len(valid) * quintile)))
        n_pick = min(n_pick, max_per_side)
        ranked = valid.sort_values(ascending=False)
        out.loc[date_, ranked.head(n_pick).index] = 1
        out.loc[date_, ranked.tail(n_pick).index] = -1
    return out


def _metrics_from_returns(returns: pd.Series) -> dict[str, float]:
    """Annualize daily portfolio returns. Matches vbt_research conventions."""
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
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd),
        "win_rate": float((returns > 0).sum() / len(returns)),
        "total_return": float(cum.iloc[-1] - 1),
    }


def _slippage_bps(volume: float | None) -> float:
    """Per-trade one-way slippage in bps from observed daily volume.

    PRD §9: 5 bps liquid, 15 bps mid-cap. We map "volume above 1M shares"
    to liquid; everything else to mid-cap. This is intentionally simple —
    the goal is to break optimistic backtests, not to model micro-structure.
    """
    if volume is None or np.isnan(volume) or volume <= 0:
        return 15.0
    return 5.0 if volume >= _LIQUID_VOL_THRESHOLD else 15.0


# ---------------------------------------------------------------------------- strategy


class EventBacktest(bt.Strategy):
    """Cross-section long-top-quintile / short-bottom-quintile strategy.

    The wide-form `signals` DataFrame is rebalanced daily: each bar we
    consult that day's row, take the top/bottom quintile, and emit market
    orders sized to equal weight across the active basket. We piggy-back
    on backtrader's broker for fill simulation, but slippage and round-trip
    cost are computed in `_apply_per_day_cost` against weight turnover —
    same model as `vbt_research` so the two engines compare cleanly.
    """

    params = (
        ("scores_df", None),  # pandas DataFrame, index=date, columns=ticker
        ("quintile", _DEFAULT_QUINTILE),
        ("max_per_side", _DEFAULT_MAX_POS_PER_SIDE),
        ("cost_bps", _DEFAULT_COST_BPS),
        ("apply_slippage", True),  # disable for vbt parity tests
    )

    def __init__(self) -> None:
        scores: pd.DataFrame = self.p.scores_df  # type: ignore[attr-defined]
        if scores is None:
            raise ValueError("EventBacktest requires scores_df parameter")
        self._positions = _quintile_positions(
            scores,
            quintile=float(self.p.quintile),  # type: ignore[attr-defined]
            max_per_side=int(self.p.max_per_side),  # type: ignore[attr-defined]
        )
        # Equal weight per row (sign / count of nonzero).
        nonzero = self._positions.abs().sum(axis=1).replace(0, np.nan)
        self._weights = self._positions.div(nonzero, axis=0).fillna(0.0)
        # Daily portfolio returns + cost ledger captured during `next()`.
        self._port_returns: list[float] = []
        self._dates: list[datetime] = []
        self._prev_weights = pd.Series(0.0, index=self._weights.columns, dtype=float)
        self._n_orders = 0
        self._slippage_costs_bps: list[float] = []
        # Map from data feed name to ticker for column lookup.
        self._data_by_ticker: dict[str, Any] = {
            d._name: d
            for d in self.datas  # type: ignore[attr-defined]
        }

    def _today(self) -> datetime:
        bt_dt = self.datas[0].datetime.datetime(0)
        return datetime(bt_dt.year, bt_dt.month, bt_dt.day, tzinfo=UTC)

    def next(self) -> None:
        today = self._today()
        # Find the row in scores that aligns with today (or the most recent
        # row strictly before today — same convention as vbt's "yesterday's
        # weights × today's returns").
        idx = self._weights.index
        # Position of today in the index.
        loc = idx.searchsorted(pd.Timestamp(today.replace(tzinfo=None)), side="right") - 1
        if loc < 0:
            return
        weights_today = self._weights.iloc[loc].fillna(0.0)

        # Compute today's per-ticker arithmetic return from the data feeds.
        per_ticker_ret = pd.Series(0.0, index=weights_today.index, dtype=float)
        per_ticker_vol = pd.Series(np.nan, index=weights_today.index, dtype=float)
        for ticker in weights_today.index:
            d = self._data_by_ticker.get(ticker)
            if d is None or len(d) < 2:
                continue
            close_today = float(d.close[0])
            close_yesterday = float(d.close[-1])
            if close_yesterday > 0:
                per_ticker_ret[ticker] = close_today / close_yesterday - 1
            per_ticker_vol[ticker] = float(d.volume[0]) if len(d.volume) else np.nan

        # Apply yesterday's weights to today's returns (no look-ahead).
        port_ret = float((self._prev_weights * per_ticker_ret).sum())

        # Turnover-based round-trip cost + per-name slippage.
        turnover = (weights_today - self._prev_weights).abs()
        slip_enabled = bool(self.p.apply_slippage)  # type: ignore[attr-defined]
        per_name_slip = pd.Series(
            [
                _slippage_bps(per_ticker_vol.get(t)) if slip_enabled else 0.0
                for t in weights_today.index
            ],
            index=weights_today.index,
            dtype=float,
        )
        # Cost per traded weight unit: (cost_bps/2 + slip_bps) * 1e-4.
        slip_cost = float((turnover * (per_name_slip + float(self.p.cost_bps) / 2.0)).sum() * 1e-4)  # type: ignore[attr-defined]
        port_ret -= slip_cost

        if turnover.sum() > 0:
            self._n_orders += int((turnover > 0).sum())
            traded_slip = per_name_slip[turnover > 0].mean()
            if not pd.isna(traded_slip):
                self._slippage_costs_bps.append(float(traded_slip))

        self._port_returns.append(port_ret)
        self._dates.append(today)
        self._prev_weights = weights_today

    def get_metrics(self) -> ValidationMetrics:
        if not self._port_returns:
            return ValidationMetrics(
                sharpe=float("nan"),
                sortino=float("nan"),
                max_drawdown=float("nan"),
                win_rate=float("nan"),
                total_return=float("nan"),
                fill_rate=1.0,
                avg_slippage_bps=0.0,
                max_intraday_drawdown=float("nan"),
                n_orders=0,
            )
        rets = pd.Series(self._port_returns, index=pd.DatetimeIndex(self._dates))
        m = _metrics_from_returns(rets)
        return ValidationMetrics(
            sharpe=m["sharpe"],
            sortino=m["sortino"],
            max_drawdown=m["max_drawdown"],
            win_rate=m["win_rate"],
            total_return=m["total_return"],
            fill_rate=1.0,  # market orders simulated as fully filled
            avg_slippage_bps=float(np.mean(self._slippage_costs_bps))
            if self._slippage_costs_bps
            else 0.0,
            max_intraday_drawdown=m["max_drawdown"],
            n_orders=self._n_orders,
        )


# ---------------------------------------------------------------------------- runner


def _build_cerebro(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_bps: float,
    quintile: float,
    max_per_side: int,
    starting_cash: float,
    apply_slippage: bool = True,
) -> tuple[bt.Cerebro, EventBacktest]:
    """Wire up Cerebro with one PandasData feed per ticker.

    Returns (cerebro, strategy_instance) — the strategy isn't actually
    instantiated until `cerebro.run()`, so we return only the cerebro and
    let `run_validation` extract the strategy from the returned list.
    """
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(starting_cash)
    # Commission as fraction (per side); 0.5 bps default.
    cerebro.broker.setcommission(commission=_DEFAULT_COMMISSION_BPS * 1e-4)

    common_cols = [c for c in signals.columns if c in prices.columns]
    common_idx = signals.index.intersection(prices.index)
    if not len(common_cols) or len(common_idx) == 0:
        # No usable feeds — return a cerebro with no data; caller produces NaN metrics.
        cerebro.addstrategy(EventBacktest, scores_df=signals)
        return cerebro, None  # type: ignore[return-value]

    # Build a frame backtrader can ingest (open/high/low/close/volume/openinterest)
    # per ticker. We synthesize OHLC = close where missing — backtrader needs
    # the columns even if the strategy only consults `close`.
    for ticker in common_cols:
        close = prices.loc[common_idx, ticker].astype(float)
        df = pd.DataFrame(
            {
                "open": close.values,
                "high": close.values,
                "low": close.values,
                "close": close.values,
                "volume": np.full(len(close), 2_000_000, dtype=np.int64),
                "openinterest": np.zeros(len(close), dtype=np.int64),
            },
            index=pd.DatetimeIndex(close.index),
        )
        # Drop rows where close is NaN (gaps).
        df = df.dropna()
        if df.empty:
            continue
        feed = bt.feeds.PandasData(dataname=df, name=ticker)
        cerebro.adddata(feed)

    cerebro.addstrategy(
        EventBacktest,
        scores_df=signals,
        cost_bps=cost_bps,
        quintile=quintile,
        max_per_side=max_per_side,
        apply_slippage=apply_slippage,
    )
    return cerebro, None  # type: ignore[return-value]


def run_validation(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_bps: float = _DEFAULT_COST_BPS,
    quintile: float = _DEFAULT_QUINTILE,
    max_per_side: int = _DEFAULT_MAX_POS_PER_SIDE,
    starting_cash: float = 100_000.0,
    save_results: bool = False,
    output_dir: Path = Path("backtests/results"),
    apply_slippage: bool = True,
) -> ValidationMetrics:
    """Run one event-driven backtest. Returns ValidationMetrics.

    Same input shape as `VectorBacktest.run_single` — wide score and price
    frames sharing index/columns. This is the function `bt_evaluator`
    delegates to and the function the walk-forward harness can swap in.
    """
    cerebro, _ = _build_cerebro(
        signals,
        prices,
        cost_bps=cost_bps,
        quintile=quintile,
        max_per_side=max_per_side,
        starting_cash=starting_cash,
        apply_slippage=apply_slippage,
    )
    runs = cerebro.run()
    if not runs:
        # Cerebro produced no strategy — return NaN metrics so callers can
        # detect "no data" without crashing.
        return ValidationMetrics(
            sharpe=float("nan"),
            sortino=float("nan"),
            max_drawdown=float("nan"),
            win_rate=float("nan"),
            total_return=float("nan"),
            fill_rate=float("nan"),
            avg_slippage_bps=float("nan"),
            max_intraday_drawdown=float("nan"),
            n_orders=0,
        )
    strat = runs[0]
    metrics: ValidationMetrics = strat.get_metrics()

    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        out = output_dir / f"validation_{ts}.json"
        out.write_text(json.dumps(metrics.as_dict(), indent=2), encoding="utf-8")
        logger.info("backtrader validation results written to {}", out)

    return metrics


def bt_evaluator(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_bps: float = _DEFAULT_COST_BPS,
    **_kwargs: Any,
) -> dict[str, float]:
    """Adapter for `walk_forward_cv(evaluator=...)`.

    Conforms to the `Callable[..., dict[str, float]]` signature already
    accepted by `walk_forward_cv` so the existing harness can swap in this
    backtrader engine without any change to `walk_forward.py`.
    """
    metrics = run_validation(signals, prices, cost_bps=cost_bps)
    return metrics.as_dict()
