"""Phase-1 decision gate: pure SUE PEAD baseline on real data.

Implements PRD §11 Phase 1 / taskmaster Task 32 — does the classical
post-earnings-announcement-drift edge reproduce on this universe before
we layer LLM signals on top?

Pipeline:
    DuckDB earnings  ──┐
                       ├──→ pead.compute_sue (point-in-time)
    DuckDB ohlcv     ──┘
                       │
                       ├──→ Q5 vs Q1 forward-return spread
                       └──→ vbt_research.run_parameter_sweep
                            (long top quintile / short bottom)

Run:
    uv run python -m casino.backtest.sue_baseline
    uv run python -m casino.backtest.sue_baseline --start 2024-06-01 --end 2026-04-30
    uv run python -m casino.backtest.sue_baseline --json    # machine-readable

Decision gate (PRD §11.5):
    PASS — Sharpe ≥ 0.5 and Q5-Q1 forward-return spread > 0.
    FAIL — anything else; do NOT proceed to Phase 2 LLM augmentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from casino.backtest import vbt_research
from casino.data import store
from casino.signals import pead

_HOLDING_WINDOW_BDAYS = 20  # Bernard-Thomas (1989) drift is a 60-day phenomenon;
# 20-day hold captures ~1/3 of the integral while keeping turnover manageable.
# Originally 5-day; bumped after 2026-05 research showed 5-day was capturing only
# ~8% of the classical drift integral and arbitraged hardest by HFT.
_DEFAULT_COST_BPS = 7.5


@dataclass(frozen=True)
class BaselineResult:
    """Numeric verdict from the baseline run. JSON-serializable."""

    n_events: int
    n_tickers_with_events: int
    start_date: str
    end_date: str
    quintile_means: dict[str, dict[str, float]]
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    total_return: float
    q5_minus_q1_5d: float
    q5_minus_q1_10d: float
    q5_minus_q1_20d: float
    cost_bps: float
    verdict: str  # "PASS" | "FAIL"
    verdict_detail: str


def _load_data(
    start: datetime,
    end: datetime,
    *,
    db_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull earnings + close-price panel from DuckDB. Returns (earnings_df, prices_wide)."""
    with store.get_duckdb_conn(db_path, read_only=True) as conn:
        earnings = conn.execute(
            """
            SELECT ticker, report_date, actual_eps, consensus_eps
            FROM earnings
            WHERE report_date BETWEEN ? AND ?
              AND actual_eps IS NOT NULL
              AND consensus_eps IS NOT NULL
            """,
            [start, end],
        ).df()
        prices = conn.execute(
            """
            SELECT ticker, ts, COALESCE(adj_close, close) AS close
            FROM ohlcv
            WHERE ts BETWEEN ? AND ?
            """,
            [start, end],
        ).df()
    if prices.empty:
        return earnings, pd.DataFrame()
    prices_wide = prices.pivot(index="ts", columns="ticker", values="close").sort_index()
    return earnings, prices_wide


def _compute_all_sue(earnings: pd.DataFrame, *, db_path: Path | None) -> pd.DataFrame:
    """Apply pead.compute_sue per row. Returns earnings + sue column, rows with NaN dropped."""
    sues: list[float | None] = []
    for r in earnings.itertuples():
        rd = r.report_date
        if hasattr(rd, "to_pydatetime"):
            rd_py = rd.to_pydatetime()
        elif isinstance(rd, datetime):
            rd_py = rd
        else:
            sues.append(None)
            continue
        sue = pead.compute_sue(
            r.ticker,
            actual_eps=Decimal(str(r.actual_eps)),
            consensus_eps=Decimal(str(r.consensus_eps)),
            as_of_date=rd_py,
            db_path=db_path,
        )
        sues.append(sue)
    out = earnings.copy()
    out["sue"] = sues
    return out.dropna(subset=["sue"]).reset_index(drop=True)


def _forward_return(
    prices_wide: pd.DataFrame,
    ticker: str,
    after: pd.Timestamp,
    horizon_bdays: int,
) -> float | None:
    if ticker not in prices_wide.columns:
        return None
    series = prices_wide[ticker].dropna()
    series = series[series.index > after]
    if len(series) < horizon_bdays + 1:
        return None
    return float(series.iloc[horizon_bdays] / series.iloc[0] - 1)


def _attach_forward_returns(sue_df: pd.DataFrame, prices_wide: pd.DataFrame) -> pd.DataFrame:
    out = sue_df.copy()
    for h in (5, 10, 20):
        out[f"fwd_{h}d"] = out.apply(
            lambda r, h=h: _forward_return(prices_wide, r.ticker, r.report_date, h),
            axis=1,
        )
    return out


def _quintile_summary(sue_df: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Return (per-quintile mean fwd returns, q5_minus_q1 by horizon)."""
    if sue_df.empty or sue_df["sue"].nunique() < 5:
        return {}, {"5d": 0.0, "10d": 0.0, "20d": 0.0}
    df = sue_df.copy()
    df["quintile"] = pd.qcut(df["sue"], q=5, labels=False, duplicates="drop")
    means = df.groupby("quintile")[["fwd_5d", "fwd_10d", "fwd_20d"]].mean()
    qs = {str(int(q)): {k: float(v) for k, v in row.items()} for q, row in means.iterrows()}
    spread = {}
    for h in ("5d", "10d", "20d"):
        col = f"fwd_{h}"
        try:
            spread[h] = float(means.loc[4, col] - means.loc[0, col])
        except KeyError:
            spread[h] = 0.0
    return qs, spread


def _sue_panel(
    sue_df: pd.DataFrame,
    prices_wide: pd.DataFrame,
    *,
    holding_window: int = _HOLDING_WINDOW_BDAYS,
) -> pd.DataFrame:
    """Project SUE events onto the price index. Each event's SUE persists for `holding_window` bdays."""
    panel = pd.DataFrame(np.nan, index=prices_wide.index, columns=prices_wide.columns)
    for r in sue_df.itertuples():
        rd = r.report_date
        if rd not in panel.index:
            after = panel.index[panel.index > rd]
            if after.empty:
                continue
            rd = after[0]
        if r.ticker not in panel.columns:
            continue
        idx = panel.index.get_loc(rd)
        end_idx = min(int(idx) + holding_window, len(panel) - 1)
        col_loc = panel.columns.get_loc(r.ticker)
        panel.iloc[int(idx) : int(end_idx), col_loc] = r.sue
    return panel.ffill(limit=holding_window)


def run_baseline(
    *,
    start: datetime,
    end: datetime,
    cost_bps: float = _DEFAULT_COST_BPS,
    holding_window: int = _HOLDING_WINDOW_BDAYS,
    db_path: Path | None = None,
    save_csv: bool = True,
) -> BaselineResult:
    earnings, prices_wide = _load_data(start, end, db_path=db_path)
    if earnings.empty or prices_wide.empty:
        raise RuntimeError(
            f"insufficient data in window [{start.date()} .. {end.date()}]: "
            f"earnings={len(earnings)} rows, prices={prices_wide.shape}"
        )

    sue_df = _compute_all_sue(earnings, db_path=db_path)
    sue_df = _attach_forward_returns(sue_df, prices_wide)
    quintile_means, q5q1 = _quintile_summary(sue_df)

    def _signal_func(_universe, _start, _end, **_kwargs):  # noqa: ANN001
        return _sue_panel(sue_df, prices_wide, holding_window=holding_window)

    results, _csv = vbt_research.run_parameter_sweep(
        _signal_func,
        param_grid={"holding_window": [holding_window]},
        universe=list(prices_wide.columns),
        prices=prices_wide,
        start_date=prices_wide.index.min().to_pydatetime(),
        end_date=prices_wide.index.max().to_pydatetime(),
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

    pass_sharpe = sharpe >= 0.5
    pass_spread = q5q1.get("10d", 0.0) > 0
    if pass_sharpe and pass_spread:
        verdict = "PASS"
        detail = f"Sharpe {sharpe:.2f} >= 0.5 and Q5-Q1 10d spread {q5q1['10d']:+.4f} > 0."
    else:
        verdict = "FAIL"
        reasons = []
        if not pass_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} < 0.5")
        if not pass_spread:
            reasons.append(f"Q5-Q1 10d spread {q5q1.get('10d', 0.0):+.4f} <= 0")
        detail = "; ".join(reasons) + ". Do not proceed to Phase 2 LLM."

    return BaselineResult(
        n_events=int(len(sue_df)),
        n_tickers_with_events=int(sue_df["ticker"].nunique()),
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
        quintile_means=quintile_means,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_return=total_return,
        q5_minus_q1_5d=q5q1["5d"],
        q5_minus_q1_10d=q5q1["10d"],
        q5_minus_q1_20d=q5q1["20d"],
        cost_bps=cost_bps,
        verdict=verdict,
        verdict_detail=detail,
    )


# ============================================================================ CLI
def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _print_human(r: BaselineResult) -> None:
    print("\n=== SUE PEAD baseline -- Phase 1 gate ===")
    print(f"window:       {r.start_date} .. {r.end_date}")
    print(f"events:       {r.n_events} across {r.n_tickers_with_events} tickers")
    print(f"cost:         {r.cost_bps} bps round-trip")
    print()
    print(f"Sharpe:       {r.sharpe:>8.3f}   (target >= 0.5)")
    print(f"Sortino:      {r.sortino:>8.3f}")
    print(f"Max DD:       {r.max_drawdown:>8.3%}")
    print(f"Win rate:     {r.win_rate:>8.3%}")
    print(f"Total ret:    {r.total_return:>8.3%}")
    print()
    print("Forward-return spread (top quintile - bottom quintile):")
    print(f"  5-day:      {r.q5_minus_q1_5d:>+8.4f}")
    print(f"  10-day:     {r.q5_minus_q1_10d:>+8.4f}   (target > 0)")
    print(f"  20-day:     {r.q5_minus_q1_20d:>+8.4f}")
    print()
    if r.quintile_means:
        print("Mean forward returns by SUE quintile (0=lowest, 4=highest):")
        for q, vals in sorted(r.quintile_means.items()):
            print(
                f"  Q{q}:  5d={vals.get('fwd_5d', float('nan')):>+7.4f}  "
                f"10d={vals.get('fwd_10d', float('nan')):>+7.4f}  "
                f"20d={vals.get('fwd_20d', float('nan')):>+7.4f}"
            )
    print()
    print(f"VERDICT: {r.verdict}")
    print(f"  {r.verdict_detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.backtest.sue_baseline",
        description="Run the pure SUE PEAD baseline (Phase 1 decision gate).",
    )
    parser.add_argument("--start", default=None, help="ISO start date (default: earliest in DB)")
    parser.add_argument("--end", default=None, help="ISO end date (default: latest in DB)")
    parser.add_argument("--cost-bps", type=float, default=_DEFAULT_COST_BPS)
    parser.add_argument(
        "--holding-window",
        type=int,
        default=_HOLDING_WINDOW_BDAYS,
        help=f"Bdays each event's SUE persists in the score panel (default {_HOLDING_WINDOW_BDAYS}).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    parser.add_argument("--no-save", action="store_true", help="Do not write sweep CSV")
    args = parser.parse_args(argv)

    # Default window = intersection of (earnings, ohlcv) ranges in DuckDB.
    if args.start is None or args.end is None:
        with store.get_duckdb_conn(read_only=True) as conn:
            row = conn.execute(
                "SELECT GREATEST(min_e, min_o), LEAST(max_e, max_o) FROM ("
                "  SELECT min(report_date) AS min_e, max(report_date) AS max_e FROM earnings"
                "    WHERE actual_eps IS NOT NULL AND consensus_eps IS NOT NULL"
                ") e CROSS JOIN ("
                "  SELECT min(ts) AS min_o, max(ts) AS max_o FROM ohlcv"
                ") o"
            ).fetchone()
        if row is None or row[0] is None:
            logger.error("DuckDB has no overlapping earnings + ohlcv data; run ingesters first")
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

    result = run_baseline(
        start=start,
        end=end,
        cost_bps=args.cost_bps,
        holding_window=args.holding_window,
        save_csv=not args.no_save,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        _print_human(result)
    return 0 if result.verdict == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
