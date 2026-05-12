"""Branch C step 1 — Time-Series Momentum gate.

After EDLLLS empirically died (memory: edlls_killed_2026-05-07.md), the strategy
pivot is to free, retail-tradeable signals on cross-asset ETFs. TSMOM is the
most-documented retail-feasible factor (Hurst-Ooi-Pedersen 2017; Moskowitz-
Ooi-Pedersen 2012 *JFE*).

PASS gate:
    Sharpe >= 0.5 AND Max Drawdown >  -25%.
    (Net of 7.5 bps round-trip, monthly rebalance assumption.)

FAIL otherwise. Run after the ETF universe (universe_tsmom.txt) is OHLCV-ingested.

Run:
    uv run python -m casino.backtest.tsmom_baseline
    uv run python -m casino.backtest.tsmom_baseline --mode long_only
    uv run python -m casino.backtest.tsmom_baseline --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from casino.backtest import deflated_sharpe as ds
from casino.data import store
from casino.signals import ts_momentum

_DEFAULT_COST_BPS = 7.5
_DEFAULT_MODE: Literal["long_short", "long_only"] = "long_short"
_DEFAULT_UNIVERSE_FILE = "universe_tsmom.txt"
_BDAYS_PER_YEAR = 252


def _per_year_sharpe(returns: pd.Series) -> dict[str, float]:
    """Annual Sharpe + total-return breakdown so we can spot regime concentration."""
    if returns.empty:
        return {}
    out: dict[str, float] = {}
    for year, group in returns.groupby(returns.index.year):
        if len(group) < 20:
            continue
        mu = float(group.mean())
        sd = float(group.std(ddof=1))
        sharpe = mu / sd * np.sqrt(_BDAYS_PER_YEAR) if sd > 0 else float("nan")
        total = float((1.0 + group).prod() - 1.0)
        out[str(int(year))] = round(sharpe, 3)
        out[f"{int(year)}_ret"] = round(total, 4)
    return out


def _backtest_weights_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_bps: float,
    rebalance: Literal["monthly", "biweekly", "weekly", "daily"] = "monthly",
) -> pd.Series:
    """Run weight-based backtest and return the daily portfolio return series."""
    common_cols = weights.columns.intersection(prices.columns)
    if common_cols.empty:
        return pd.Series(dtype=float)
    w = weights[common_cols].sort_index()
    p = prices[common_cols].sort_index()
    common_idx = w.index.intersection(p.index)
    w = w.loc[common_idx]
    p = p.loc[common_idx]

    if rebalance == "daily":
        rebal_mask = pd.Series(True, index=w.index)
    elif rebalance == "weekly":
        rebal_mask = (
            pd.Series(w.index.isocalendar().week.values, index=w.index).diff().fillna(1) != 0
        )
    elif rebalance == "biweekly":
        # Biweekly = every-other ISO-week boundary: rebalance at the first day of a
        # new ISO week whose week number has even parity. Approximates "every 10
        # trading days" without drifting through holidays.
        weeks = pd.Series(w.index.isocalendar().week.values, index=w.index)
        new_week = weeks.diff().fillna(1) != 0
        even_week = (weeks % 2) == 0
        rebal_mask = new_week & even_week
        # Always rebalance on first bar so positions are seeded.
        if len(rebal_mask) > 0:
            rebal_mask.iloc[0] = True
    else:  # monthly
        idx_dt = pd.DatetimeIndex(w.index)
        idx_naive = idx_dt.tz_localize(None) if idx_dt.tz is not None else idx_dt
        months = pd.Series(idx_naive.to_period("M"), index=w.index)
        rebal_mask = months.ne(months.shift(1)).fillna(True)

    held = w.where(rebal_mask, np.nan).ffill().fillna(0.0)
    rets = p.pct_change(fill_method=None).fillna(0.0)
    port_ret = (held.shift(1).fillna(0.0) * rets).sum(axis=1)
    weight_change = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_drag = weight_change * (cost_bps / 2.0) * 1e-4
    return (port_ret - cost_drag).dropna()


def _backtest_weights(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_bps: float,
    rebalance: Literal["monthly", "biweekly", "weekly", "daily"] = "monthly",
) -> dict[str, float]:
    """Run a position-weight backtest with periodic rebalancing.

    Unlike vbt_research's quintile-rank harness, this respects the actual
    target weights (vol-targeting matters here). Rebalances on the *first*
    trading day of each new month/(2-)week (or every day in 'daily' mode).
    """
    port_ret = _backtest_weights_returns(weights, prices, cost_bps=cost_bps, rebalance=rebalance)
    if port_ret.empty:
        return _empty_metrics()

    # Recompute turnover for the avg-turnover metric (cheap to redo).
    common_cols = weights.columns.intersection(prices.columns)
    w = weights[common_cols].sort_index()
    common_idx = w.index.intersection(prices.index)
    w = w.loc[common_idx]
    if rebalance == "monthly":
        idx_dt = pd.DatetimeIndex(w.index)
        idx_naive = idx_dt.tz_localize(None) if idx_dt.tz is not None else idx_dt
        months = pd.Series(idx_naive.to_period("M"), index=w.index)
        rebal_mask = months.ne(months.shift(1)).fillna(True)
    elif rebalance == "weekly":
        rebal_mask = (
            pd.Series(w.index.isocalendar().week.values, index=w.index).diff().fillna(1) != 0
        )
    elif rebalance == "biweekly":
        weeks = pd.Series(w.index.isocalendar().week.values, index=w.index)
        new_week = weeks.diff().fillna(1) != 0
        even_week = (weeks % 2) == 0
        rebal_mask = new_week & even_week
        if len(rebal_mask) > 0:
            rebal_mask.iloc[0] = True
    else:
        rebal_mask = pd.Series(True, index=w.index)
    held = w.where(rebal_mask, np.nan).ffill().fillna(0.0)
    weight_change = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    avg_turnover = float(weight_change[rebal_mask].mean())

    mu = float(port_ret.mean())
    sigma = float(port_ret.std(ddof=1))
    sharpe = mu / sigma * np.sqrt(_BDAYS_PER_YEAR) if sigma > 0 else float("nan")
    downside_sigma = float(port_ret.clip(upper=0).std(ddof=1))
    sortino = mu / downside_sigma * np.sqrt(_BDAYS_PER_YEAR) if downside_sigma > 0 else float("nan")
    cum = (1.0 + port_ret).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())
    total_return = float(cum.iloc[-1] - 1.0)
    win_rate = float((port_ret > 0).sum() / len(port_ret))

    yearly = _per_year_sharpe(port_ret)
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_return": total_return,
        "ann_vol": sigma * np.sqrt(_BDAYS_PER_YEAR),
        "avg_turnover": avg_turnover,
        **{f"yearly_{k}": v for k, v in yearly.items()},
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "sharpe": float("nan"),
        "sortino": float("nan"),
        "max_drawdown": float("nan"),
        "win_rate": float("nan"),
        "total_return": float("nan"),
        "ann_vol": float("nan"),
        "avg_turnover": float("nan"),
    }


@dataclass(frozen=True)
class TSMomResult:
    n_assets: int
    start_date: str
    end_date: str
    mode: str
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    total_return: float
    annualized_return: float
    annualized_vol: float
    cost_bps: float
    per_asset_avg_weight: dict[str, float]
    yearly_sharpe: dict[str, float]
    deflated_sharpe: float
    deflated_p_value: float
    deflated_n_trials: int
    deflated_n_observations: int
    verdict: str
    verdict_detail: str


def _read_universe_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.upper())
    return out


def run_tsmom(
    *,
    start: datetime,
    end: datetime,
    universe: list[str],
    cost_bps: float = _DEFAULT_COST_BPS,
    mode: Literal["long_short", "long_only"] = _DEFAULT_MODE,
    n_trials: int = 10,
    db_path: Path | None = None,
    save_csv: bool = True,
) -> TSMomResult:
    prices = ts_momentum.load_ohlcv_panel(start=start, end=end, universe=universe, db_path=db_path)
    if prices.empty:
        raise RuntimeError(
            f"no OHLCV in [{start.date()}..{end.date()}] for universe {universe}. "
            "Run: uv run python -m casino.data.ingest_yfinance --tickers-file "
            f"{_DEFAULT_UNIVERSE_FILE} --mode ohlcv --ohlcv-start 2018-01-01 --rate-limit-sec 0"
        )

    weights = ts_momentum.compute_tsmom_panel(prices, mode=mode)

    metrics = _backtest_weights(weights, prices, cost_bps=cost_bps, rebalance="monthly")
    sharpe = metrics["sharpe"]
    sortino = metrics["sortino"]
    max_dd = metrics["max_drawdown"]
    win_rate = metrics["win_rate"]
    total_return = metrics["total_return"]
    ann_vol = metrics["ann_vol"]

    # Deflated Sharpe (Bailey-Lopez de Prado): correct for selection bias.
    daily_returns = _backtest_weights_returns(
        weights, prices, cost_bps=cost_bps, rebalance="monthly"
    )
    deflate_result = ds.haircut_sharpe(
        sharpe,
        {
            "n_trials": n_trials,
            "n_observations": len(daily_returns),
            "returns": daily_returns.tolist(),
        },
    )

    days = (prices.index.max() - prices.index.min()).days
    years = max(days / 365.25, 1e-6)
    ann_ret = (1 + total_return) ** (1 / years) - 1.0

    if save_csv:
        out_dir = Path("backtests/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"tsmom_{ts_str}.csv"
        pd.DataFrame([{**metrics, "mode": mode, "ann_ret": ann_ret}]).to_csv(out_path, index=False)
        logger.info("tsmom result written to {}", out_path)

    avg_w = {col: float(np.nanmean(weights[col].abs())) for col in weights.columns}
    yearly_sharpe = {
        k.removeprefix("yearly_"): v for k, v in metrics.items() if k.startswith("yearly_")
    }

    pass_sharpe = sharpe >= 0.5
    pass_dd = max_dd > -0.25
    pass_deflated = bool(deflate_result["is_significant"])
    if pass_sharpe and pass_dd and pass_deflated:
        verdict = "PASS"
        detail = (
            f"Sharpe {sharpe:.2f} >= 0.5, Max DD {max_dd:.1%} > -25%, deflated Sharpe "
            f"{deflate_result['deflated']:.2f} > 0 (p={deflate_result['p_value']:.4f}). "
            f"Annualized return {ann_ret:.1%}; signal is retail-tradeable on this universe."
        )
    else:
        verdict = "FAIL"
        reasons = []
        if not pass_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} < 0.5")
        if not pass_dd:
            reasons.append(f"Max DD {max_dd:.1%} <= -25%")
        if not pass_deflated:
            reasons.append(
                f"deflated Sharpe {deflate_result['deflated']:.2f} not significant "
                f"(p={deflate_result['p_value']:.4f}, n_trials={n_trials})"
            )
        detail = (
            "; ".join(reasons)
            + ". Try widening the lookback blend, raising vol-target, or expanding the universe "
            "(more cross-asset breadth = more diversification benefit per Hurst-Ooi-Pedersen)."
        )

    return TSMomResult(
        n_assets=int(prices.shape[1]),
        start_date=prices.index.min().date().isoformat(),
        end_date=prices.index.max().date().isoformat(),
        mode=mode,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_return=total_return,
        annualized_return=float(ann_ret),
        annualized_vol=float(abs(ann_vol)),
        cost_bps=cost_bps,
        per_asset_avg_weight=avg_w,
        yearly_sharpe=yearly_sharpe,
        deflated_sharpe=float(deflate_result["deflated"]),
        deflated_p_value=float(deflate_result["p_value"]),
        deflated_n_trials=int(deflate_result["n_trials"]),
        deflated_n_observations=int(deflate_result["n_observations"]),
        verdict=verdict,
        verdict_detail=detail,
    )


# ============================================================================ CLI
def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _print_human(r: TSMomResult) -> None:
    print("\n=== TS-Momentum -- Branch C Step 1 gate ===")
    print(f"window:       {r.start_date} .. {r.end_date}")
    print(f"assets:       {r.n_assets}  (mode={r.mode})")
    print(f"cost:         {r.cost_bps} bps round-trip")
    print()
    print(f"Sharpe:       {r.sharpe:>8.3f}   (PASS target >= 0.5)")
    print(
        f"Deflated SR:  {r.deflated_sharpe:>8.3f}   (PASS target > 0, p<0.05; "
        f"n_trials={r.deflated_n_trials}, n_obs={r.deflated_n_observations})"
    )
    print(f"  p-value:    {r.deflated_p_value:>8.4f}")
    print(f"Sortino:      {r.sortino:>8.3f}")
    print(f"Max DD:       {r.max_drawdown:>8.3%}   (PASS target > -25%)")
    print(f"Win rate:     {r.win_rate:>8.3%}")
    print(f"Total ret:    {r.total_return:>8.3%}")
    print(f"Ann. ret:     {r.annualized_return:>8.3%}")
    print(f"Ann. vol:     {r.annualized_vol:>8.3%}")
    print()
    if r.per_asset_avg_weight:
        print("Avg |weight| per asset:")
        for k, v in sorted(r.per_asset_avg_weight.items()):
            print(f"  {k:<5}  {v:>+7.3f}")
    if r.yearly_sharpe:
        print()
        print("Per-year Sharpe / total return (regime-concentration check):")
        years = sorted(k for k in r.yearly_sharpe if not k.endswith("_ret"))
        for y in years:
            s = r.yearly_sharpe.get(y, float("nan"))
            t = r.yearly_sharpe.get(f"{y}_ret", float("nan"))
            print(f"  {y}:  Sharpe {s:>+6.2f}   return {t:>+7.2%}")
    print()
    print(f"VERDICT: {r.verdict}")
    print(f"  {r.verdict_detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.backtest.tsmom_baseline",
        description="Run the TS-Momentum cross-asset gate (Branch C step 1).",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--universe-file",
        default=_DEFAULT_UNIVERSE_FILE,
        help=f"Path to newline-delimited tickers file (default {_DEFAULT_UNIVERSE_FILE})",
    )
    parser.add_argument("--cost-bps", type=float, default=_DEFAULT_COST_BPS)
    parser.add_argument(
        "--mode",
        choices=("long_short", "long_only"),
        default=_DEFAULT_MODE,
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Trial count for deflated Sharpe (default 10; bumps the haircut, set higher if you've iterated more).",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    universe = _read_universe_file(Path(args.universe_file))
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
                "no OHLCV for universe {}; run: uv run python -m casino.data.ingest_yfinance "
                "--tickers-file {} --mode ohlcv --ohlcv-start 2018-01-01 --rate-limit-sec 0",
                universe,
                args.universe_file,
            )
            return 2
        start = _parse_date(args.start) if args.start else row[0]
        end = _parse_date(args.end) if args.end else row[1]
    else:
        start = _parse_date(args.start)
        end = _parse_date(args.end)

    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)

    result = run_tsmom(
        start=start,
        end=end,
        universe=universe,
        cost_bps=args.cost_bps,
        mode=args.mode,
        n_trials=args.n_trials,
        save_csv=not args.no_save,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        _print_human(result)
    return 0 if result.verdict == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
