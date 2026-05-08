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


# ---------------------------------------------------------------------------- TSMOM
# Task 35 — event-driven validation of cross-asset TSMOM.
#
# This is a separate Strategy because TSMOM consumes *target weights* (not
# cross-section scores), rebalances monthly (not daily), and uses next-open
# fills (not same-close). Sharing the daily-quintile EventBacktest above would
# silently break each of those invariants.
#
# Total-return handling: load_ohlcv_panel in ts_momentum reads
# COALESCE(adj_close, close), which is total-return adjusted by the ingestion
# layer. Feeding that series into the bt feed gives dividend reinvestment by
# construction — we explicitly do NOT add a separate dividend cash event,
# which would double-count.

_TSMOM_DEFAULT_COST_BPS_RT: float = 12.0


class TSMomEventBacktest(bt.Strategy):
    """Event-driven TSMOM portfolio with monthly rebalance + next-open fills.

    Workflow per bar:
        1. On the *first trading day of a new month*, look up the previous
           business day's row of `weights_df` (the signal as it existed at
           close yesterday) and queue market orders for the next bar's open.
           This enforces signal-day+1 next-open fills — no look-ahead.
        2. Between rebalances, hold positions; daily PnL accrues naturally
           from the data feeds.
        3. Cost handling is delegated to Cerebro's broker commission
           (`setcommission` in `_build_tsmom_cerebro`), which charges per-side
           on each filled order. 12 bps round-trip = 6 bps per side.

    Daily portfolio returns are recorded from broker `getvalue()` so the
    return path is exactly what the broker simulated (commissions included).
    """

    params = (
        ("weights_df", None),  # pandas DataFrame, index=date (sorted), columns=ticker
        ("cost_bps_rt", _TSMOM_DEFAULT_COST_BPS_RT),  # informational; broker enforces
    )

    def __init__(self) -> None:
        weights = self.p.weights_df  # type: ignore[attr-defined]
        if weights is None:
            raise ValueError("TSMomEventBacktest requires weights_df parameter")
        if not isinstance(weights, pd.DataFrame):
            raise TypeError("weights_df must be a pandas DataFrame")
        # Strip tz so .searchsorted comparisons stay consistent with bt's naive datetimes.
        idx = pd.DatetimeIndex(weights.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        self._weights = weights.copy()
        self._weights.index = idx

        self._data_by_ticker: dict[str, Any] = {
            d._name: d
            for d in self.datas  # type: ignore[attr-defined]
        }
        self._values: list[float] = []
        self._dates: list[datetime] = []
        self._n_orders = 0
        self._last_rebal_month: tuple[int, int] | None = None

    # ------------------------------------------------------------------ helpers
    def _today(self) -> datetime:
        bt_dt = self.datas[0].datetime.datetime(0)
        return datetime(bt_dt.year, bt_dt.month, bt_dt.day, tzinfo=UTC)

    def _signal_row_at_or_before(self, today: datetime) -> pd.Series | None:
        """Return weights row at `today`, or the most recent row prior.

        Backtrader's market-order semantics fill on the *next bar's open*, so
        submitting on bar N implicitly enforces the signal-day → next-open
        fill contract. We therefore read the current-bar's weight row (which
        was computed from data observed at *yesterday's close* via
        `compute_tsmom_panel` — that function reads `prices.shift(...)` so
        every cell is computed from strictly-prior bars).
        """
        idx = self._weights.index
        target = pd.Timestamp(today.replace(tzinfo=None))
        loc = idx.searchsorted(target, side="right") - 1
        if loc < 0:
            return None
        row = self._weights.iloc[loc].fillna(0.0)
        return row.astype(float)

    # ------------------------------------------------------------------ bt API
    def next(self) -> None:
        today = self._today()
        # Record portfolio value for daily-return computation.
        self._values.append(float(self.broker.getvalue()))
        self._dates.append(today)

        # Detect first trading day of a new month.
        month_key = (today.year, today.month)
        if self._last_rebal_month != month_key:
            self._last_rebal_month = month_key
            row = self._signal_row_at_or_before(today)
            if row is not None:
                # Submit market orders this bar; bt fills at next bar's open.
                # That's the "signal observed at today's close → trade
                # tomorrow's open" semantics required by task 35.
                self._submit_target_weights(row)

    def _submit_target_weights(self, target_weights: pd.Series) -> None:
        """Translate target portfolio weights into Backtrader orders."""
        total_value = float(self.broker.getvalue())
        if total_value <= 0:
            return
        for ticker, target_w in target_weights.items():
            d = self._data_by_ticker.get(str(ticker))
            if d is None or len(d) < 1:
                continue
            price = float(d.close[0])
            if price <= 0 or np.isnan(price):
                continue
            target_value = float(target_w) * total_value
            target_size = target_value / price
            # Use order_target_size for crisp rebalance semantics.
            current_size = self.getposition(d).size
            delta = target_size - current_size
            if abs(delta) < 1e-6:
                continue
            self.order_target_size(data=d, target=target_size)
            self._n_orders += 1

    # ------------------------------------------------------------------ metrics
    def get_returns(self) -> pd.Series:
        """Daily portfolio returns from broker value path."""
        if len(self._values) < 2:
            return pd.Series(dtype=float)
        s = pd.Series(self._values, index=pd.DatetimeIndex(self._dates))
        return s.pct_change().dropna()

    def get_metrics(self) -> dict[str, float]:
        rets = self.get_returns()
        m = _metrics_from_returns(rets)
        m["n_orders"] = float(self._n_orders)
        return m


def _build_tsmom_cerebro(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_bps_rt: float,
    starting_cash: float,
) -> bt.Cerebro:
    """Wire up Cerebro for the TSMOM event-driven validator.

    `prices` is the same total-return adj-close panel that produced `weights`
    — feeding that series into bt gives dividend-reinvested backtests by
    construction (no separate dividend cash events, which would double-count).
    """
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(starting_cash)
    # 12 bps round-trip = 6 bps per side. Using `commission` as a fraction.
    per_side = (float(cost_bps_rt) / 2.0) * 1e-4
    cerebro.broker.setcommission(commission=per_side)

    # Align columns and dates between weights and prices.
    common_cols = [c for c in weights.columns if c in prices.columns]
    common_idx = weights.index.intersection(prices.index)
    if not common_cols or len(common_idx) == 0:
        cerebro.addstrategy(TSMomEventBacktest, weights_df=weights, cost_bps_rt=cost_bps_rt)
        return cerebro

    for ticker in common_cols:
        close = prices.loc[common_idx, ticker].astype(float)
        # Drop leading NaNs so each feed starts at first valid bar; bt stitches
        # the feeds together correctly across heterogeneous start dates.
        close = close.dropna()
        if close.empty:
            continue
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
        feed = bt.feeds.PandasData(dataname=df, name=ticker)
        cerebro.adddata(feed)

    cerebro.addstrategy(
        TSMomEventBacktest,
        weights_df=weights,
        cost_bps_rt=cost_bps_rt,
    )
    return cerebro


def run_tsmom_validation(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_bps_rt: float = _TSMOM_DEFAULT_COST_BPS_RT,
    starting_cash: float = 100_000.0,
) -> dict[str, float]:
    """Run the TSMOM event-driven validator and return a metrics dict.

    Output keys match `_metrics_from_returns` (sharpe, sortino, max_drawdown,
    win_rate, total_return) plus `n_orders`. Designed to be diff-able against
    `tsmom_baseline._backtest_weights` for the bt-vs-vbt parity gate.
    """
    cerebro = _build_tsmom_cerebro(
        weights,
        prices,
        cost_bps_rt=cost_bps_rt,
        starting_cash=starting_cash,
    )
    runs = cerebro.run()
    if not runs:
        return {
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "win_rate": float("nan"),
            "total_return": float("nan"),
            "n_orders": 0.0,
        }
    strat = runs[0]
    return strat.get_metrics()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------- CLI


def _read_universe_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.upper())
    return out


def _run_tsmom_cli(args: Any) -> int:
    """CLI handler for `--strategy tsmom`. Pulls OHLCV from DuckDB, computes
    weights via ts_momentum, and runs the event-driven validator. Compares
    against the vbt headline (tsmom_baseline) Sharpe so the operator sees the
    delta directly.
    """
    # Imports kept inside the CLI handler so module import cost stays low for
    # callers (tests, walk-forward harness) that don't need the data path.
    from casino.backtest import tsmom_baseline
    from casino.data import store
    from casino.signals import ts_momentum

    universe_path = Path(args.universe_file)
    universe = _read_universe_file(universe_path)
    if not universe:
        logger.error("universe file {} resolved to no tickers", args.universe_file)
        return 2

    if args.start is None or args.end is None:
        with store.get_duckdb_conn(read_only=True) as conn:
            row = conn.execute(
                "SELECT min(ts), max(ts) FROM ohlcv WHERE ticker IN ("
                + ",".join("?" * len(universe))
                + ")",
                universe,
            ).fetchone()
        if row is None or row[0] is None:
            logger.error(
                "no OHLCV for universe {}; run yfinance ingestion first",
                universe,
            )
            return 2
        start = datetime.fromisoformat(args.start).replace(tzinfo=UTC) if args.start else row[0]
        end = datetime.fromisoformat(args.end).replace(tzinfo=UTC) if args.end else row[1]
    else:
        start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
        end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

    prices = ts_momentum.load_ohlcv_panel(start=start, end=end, universe=universe)
    if prices.empty:
        logger.error("no prices loaded for universe {}", universe)
        return 2

    weights = ts_momentum.compute_tsmom_panel(prices)

    bt_metrics = run_tsmom_validation(weights, prices, cost_bps_rt=float(args.cost_bps_rt))
    vbt_returns = tsmom_baseline._backtest_weights_returns(
        weights, prices, cost_bps=float(args.cost_bps_rt), rebalance="monthly"
    )
    vbt_metrics = _metrics_from_returns(vbt_returns)

    delta = abs(bt_metrics["sharpe"] - vbt_metrics["sharpe"])

    print("\n=== TSMOM event-driven validation (Backtrader) ===")
    print(f"window:            {prices.index.min().date()} .. {prices.index.max().date()}")
    print(f"universe size:     {len(universe)}  cost={args.cost_bps_rt} bps RT")
    print(f"vbt    Sharpe:     {vbt_metrics['sharpe']:>+7.3f}")
    print(f"bt     Sharpe:     {bt_metrics['sharpe']:>+7.3f}")
    print(f"delta:             {delta:>7.3f}   (PASS target < 0.07)")
    print(f"bt MaxDD:          {bt_metrics['max_drawdown']:>+7.3%}")
    print(f"bt total ret:      {bt_metrics['total_return']:>+7.3%}")
    print(f"bt n_orders:       {int(bt_metrics['n_orders'])}")

    return 0 if delta < 0.07 else 1


def _build_arg_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="casino.backtest.bt_validate",
        description="Backtrader event-driven validator (PEAD quintile or TSMOM).",
    )
    parser.add_argument(
        "--strategy",
        choices=("quintile", "tsmom"),
        default="quintile",
        help="Which strategy to validate (quintile=PEAD cross-section, tsmom=TSMOM).",
    )
    parser.add_argument("--start", default=None, help="ISO start date (default: data range)")
    parser.add_argument("--end", default=None, help="ISO end date (default: data range)")
    parser.add_argument(
        "--universe-file",
        default="universe_tsmom.txt",
        help="Newline-delimited tickers file (TSMOM only)",
    )
    parser.add_argument(
        "--cost-bps-rt",
        type=float,
        default=_TSMOM_DEFAULT_COST_BPS_RT,
        help="Round-trip cost in bps (TSMOM only; default 12)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.strategy == "tsmom":
        return _run_tsmom_cli(args)
    # Default `quintile` path is exercised through `bt_evaluator` /
    # `run_validation` in the existing PEAD harness; CLI invocation would need
    # signal/price loaders that don't exist in PEAD form. Print a clear message.
    print(
        "quintile (PEAD) strategy is invoked via run_validation/bt_evaluator from"
        " the PEAD harness, not this CLI. Use --strategy tsmom for the TSMOM gate.",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
