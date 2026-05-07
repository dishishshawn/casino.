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
from loguru import logger

from casino.backtest import vbt_research
from casino.data import store
from casino.signals import ts_momentum

_DEFAULT_COST_BPS = 7.5
_DEFAULT_MODE: Literal["long_short", "long_only"] = "long_short"
_DEFAULT_UNIVERSE_FILE = "universe_tsmom.txt"


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
    db_path: Path | None = None,
    save_csv: bool = True,
) -> TSMomResult:
    prices = ts_momentum.load_ohlcv_panel(
        start=start, end=end, universe=universe, db_path=db_path
    )
    if prices.empty:
        raise RuntimeError(
            f"no OHLCV in [{start.date()}..{end.date()}] for universe {universe}. "
            "Run: uv run python -m casino.data.ingest_yfinance --tickers-file "
            f"{_DEFAULT_UNIVERSE_FILE} --mode ohlcv --ohlcv-start 2018-01-01 --rate-limit-sec 0"
        )

    weights = ts_momentum.compute_tsmom_panel(prices, mode=mode)

    def _signal_func(_universe, _start, _end, **_kwargs):  # noqa: ANN001
        return weights

    results, _csv = vbt_research.run_parameter_sweep(
        _signal_func,
        param_grid={"mode": [mode]},
        universe=list(prices.columns),
        prices=prices,
        start_date=prices.index.min().to_pydatetime(),
        end_date=prices.index.max().to_pydatetime(),
        cost_bps=cost_bps,
        save_results=save_csv,
    )
    if results.empty:
        raise RuntimeError("vectorbt sweep returned no rows")
    best = results.iloc[0]

    sharpe = float(best["sharpe"])
    sortino = float(best["sortino"])
    max_dd = float(best["max_drawdown"])
    win_rate = float(best["win_rate"])
    total_return = float(best["total_return"])

    # Time span in years (approx).
    days = (prices.index.max() - prices.index.min()).days
    years = max(days / 365.25, 1e-6)
    ann_ret = (1 + total_return) ** (1 / years) - 1.0
    # Annualized vol estimate from sharpe (best.sharpe = ann_ret / ann_vol assuming rf=0).
    ann_vol = ann_ret / sharpe if sharpe and abs(sharpe) > 1e-9 else 0.0

    avg_w = {col: float(np.nanmean(weights[col].abs())) for col in weights.columns}

    pass_sharpe = sharpe >= 0.5
    pass_dd = max_dd > -0.25
    if pass_sharpe and pass_dd:
        verdict = "PASS"
        detail = (
            f"Sharpe {sharpe:.2f} >= 0.5 AND Max DD {max_dd:.1%} > -25%. "
            f"Annualized return {ann_ret:.1%}; signal is retail-tradeable on this universe."
        )
    else:
        verdict = "FAIL"
        reasons = []
        if not pass_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} < 0.5")
        if not pass_dd:
            reasons.append(f"Max DD {max_dd:.1%} <= -25%")
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
        save_csv=not args.no_save,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        _print_human(result)
    return 0 if result.verdict == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
