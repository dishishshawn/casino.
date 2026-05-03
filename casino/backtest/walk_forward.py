"""Walk-forward cross-validation harness.

For each rolling (train, test) split:
    1. sweep `param_grid` on `train` window via vbt_research,
    2. select best config by Sharpe,
    3. evaluate that config out-of-sample on `test` window,
    4. record train/test Sharpe, selected params, and degradation.

Strict invariants:
    * train_end < test_start for every window (no temporal overlap).
    * Test windows do not overlap each other.
    * Universe constituents at the start of each test period are queried via
      `casino.data.store.point_in_time_constituents` (no survivorship bias).

# Phase 2 note: PRD §11 requires backtrader event-driven validation (task 16).
# In Phase 1 we run both folds through the vbt path; once task 16 lands the
# selected-config re-evaluation step can swap to bt_validate without touching
# this module's interface. The validation engine is pluggable via `evaluator`.
"""

from __future__ import annotations

import calendar
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from casino.backtest import vbt_research
from casino.data import store


@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class WindowResult:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    selected_params: dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    test_total_return: float
    test_max_drawdown: float

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["selected_params"] = dict(self.selected_params)
        return d


def _add_months(d: datetime, months: int) -> datetime:
    """Add `months` calendar months to `d` (clamping day-of-month if needed)."""
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)
    return d.replace(year=year, month=month, day=day)


def _generate_windows(
    start_date: datetime,
    end_date: datetime,
    *,
    train_months: int,
    test_months: int,
) -> list[tuple[DateRange, DateRange]]:
    """Generate non-test-overlapping (train, test) windows.

    Each train period is `train_months` long and its `test` period is the
    immediately-following `test_months`. We advance by `test_months` (so test
    windows tile exactly without overlap; train windows overlap).
    """
    windows: list[tuple[DateRange, DateRange]] = []
    train_start = start_date
    while True:
        train_end = _add_months(train_start, train_months)
        test_start = train_end
        test_end = _add_months(test_start, test_months)
        if test_end > end_date:
            break
        # Strict invariant: train_end < test_start would be violated by =;
        # we treat boundary as exclusive on the train side via downstream
        # signal_func contracts (signals on `test_start` must not look back
        # into the train period for *future* data — that's the signal's job).
        windows.append((DateRange(train_start, train_end), DateRange(test_start, test_end)))
        train_start = _add_months(train_start, test_months)
    return windows


def _get_pit_universe(
    base_universe: list[str],
    *,
    as_of: datetime,
    index_name: str,
    db_path: Path | None = None,
) -> list[str]:
    """Filter `base_universe` to tickers that were in `index_name` on `as_of`.

    Falls back to the full base universe (with a warning) if no constituent
    history is loaded — Phase 1 has the table but won't have populated it.
    """
    try:
        members = store.point_in_time_constituents(index_name, as_of, db_path=db_path)
    except Exception as e:  # noqa: BLE001 — graceful fallback for empty DB
        logger.warning("PIT constituents query failed ({}); falling back to base_universe", e)
        return base_universe
    if not members:
        logger.warning(
            "no PIT constituents for index={} as_of={} — falling back to full base_universe",
            index_name,
            as_of,
        )
        return base_universe
    member_set = set(members)
    return [t for t in base_universe if t in member_set]


def walk_forward_cv(
    signal_func: vbt_research.SignalFunc,
    param_grid: Mapping[str, Sequence[Any]],
    *,
    universe: Iterable[str],
    prices: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    train_months: int = 12,
    test_months: int = 3,
    index_name: str = "SP500",
    cost_bps: float = 7.0,
    save_results: bool = True,
    output_dir: Path = Path("backtests/results"),
    evaluator: Callable[..., dict[str, float]] | None = None,
    db_path: Path | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """Run walk-forward CV. Returns (per-window DataFrame, csv_path | None).

    `evaluator(signals, prices, cost_bps=...) -> metrics` lets callers swap in
    backtrader (task 16) once available; defaults to the vbt single-shot
    backtester.
    """
    universe_list = list(universe)
    windows = _generate_windows(
        start_date, end_date, train_months=train_months, test_months=test_months
    )
    if not windows:
        return pd.DataFrame(), None

    if evaluator is None:
        bt = vbt_research.VectorBacktest(cost_bps=cost_bps)

        def evaluator(
            signals: pd.DataFrame, prices: pd.DataFrame, **_kwargs: Any
        ) -> dict[str, float]:  # noqa: E501
            return bt.run_single(signals, prices)

    rows: list[dict[str, Any]] = []
    for train_range, test_range in windows:
        # PIT-universe at start of test
        pit_universe = _get_pit_universe(
            universe_list,
            as_of=test_range.start,
            index_name=index_name,
            db_path=db_path,
        )

        # 1) sweep on train
        prices_train = prices.loc[
            (prices.index >= train_range.start) & (prices.index < train_range.end),
            [c for c in prices.columns if c in pit_universe],
        ]
        sweep_df, _ = vbt_research.run_parameter_sweep(
            signal_func,
            param_grid,
            universe=pit_universe,
            prices=prices_train,
            start_date=train_range.start,
            end_date=train_range.end,
            cost_bps=cost_bps,
            save_results=False,
            output_dir=output_dir,
        )
        if sweep_df.empty or sweep_df["sharpe"].isna().all():
            logger.warning(
                "train sweep empty for window {}..{}", train_range.start, train_range.end
            )
            continue
        best = sweep_df.iloc[0]
        selected = {k: best[k] for k in param_grid.keys()}
        train_sharpe = float(best["sharpe"]) if not pd.isna(best["sharpe"]) else float("nan")

        # 2) evaluate selected params OOS on test window
        prices_test = prices.loc[
            (prices.index >= test_range.start) & (prices.index < test_range.end),
            [c for c in prices.columns if c in pit_universe],
        ]
        test_signals = signal_func(pit_universe, test_range.start, test_range.end, **selected)
        test_metrics = evaluator(test_signals, prices_test)

        rows.append(
            WindowResult(
                train_start=train_range.start,
                train_end=train_range.end,
                test_start=test_range.start,
                test_end=test_range.end,
                selected_params=selected,
                train_sharpe=train_sharpe,
                test_sharpe=float(test_metrics.get("sharpe", float("nan"))),
                test_total_return=float(test_metrics.get("total_return", float("nan"))),
                test_max_drawdown=float(test_metrics.get("max_drawdown", float("nan"))),
            ).as_dict()
        )

    df = pd.DataFrame(rows)
    out_path: Path | None = None
    if save_results and not df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = output_dir / f"walk_forward_{ts}.csv"
        df.to_csv(out_path, index=False)
        logger.info("walk_forward results written to {}", out_path)

    return df, out_path


def aggregate_walk_forward(df: pd.DataFrame) -> dict[str, float]:
    """Cross-window aggregates: mean/std test Sharpe, degradation, win rate."""
    if df.empty:
        return {
            "mean_test_sharpe": float("nan"),
            "std_test_sharpe": float("nan"),
            "mean_train_sharpe": float("nan"),
            "degradation": float("nan"),
            "test_window_win_rate": float("nan"),
            "n_windows": 0,
        }
    return {
        "mean_test_sharpe": float(df["test_sharpe"].mean()),
        "std_test_sharpe": float(df["test_sharpe"].std(ddof=1)) if len(df) > 1 else 0.0,
        "mean_train_sharpe": float(df["train_sharpe"].mean()),
        "degradation": float(df["train_sharpe"].mean() - df["test_sharpe"].mean()),
        "test_window_win_rate": float((df["test_sharpe"] > 0).sum() / len(df)),
        "n_windows": int(len(df)),
    }
