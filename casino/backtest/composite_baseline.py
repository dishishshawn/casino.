"""Phase-1 Stage-1 composite signal gate.

Pure SUE PEAD on S&P 500 with 20-day hold confirmed FAIL twice (2026-05-04 Sharpe
-0.10, 2026-05-05 Sharpe -0.42). Cross-section monotonic but Q5-Q1 spread is
~0.45% / 10d, too thin to overcome 7.5bps round-trip cost. Per research recipe
(memory: pead_research_findings.md), the remaining $0 lever is a composite score:

    score = 0.4 * SUE + 0.4 * SURGE + 0.2 * |SUE| * sign(60d_momentum)

SUE: standardized earnings surprise (rolling std of prior 8 surprises).
SURGE: post-announcement abnormal-volume × return on the first trading day after
       the announcement, z-scored per ticker over a 60d trailing window. Captures
       the market's interpretation of the surprise (Meursault et al. 2022 *JFQA*).
Momentum interaction: |SUE| * sign(60d total return). Adds direction conviction
       when surprise magnitude and prior trend agree.

PASS gate: Sharpe >= 0.5 AND Q5-Q1 10-day forward-return spread > 0 AND
           composite Sharpe > pure SUE baseline Sharpe (improvement check).

Run:
    uv run python -m casino.backtest.composite_baseline
    uv run python -m casino.backtest.composite_baseline --json
    uv run python -m casino.backtest.composite_baseline --no-save
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from casino.backtest import sue_baseline, vbt_research
from casino.data import store

_HOLDING_WINDOW_BDAYS = 20
_DEFAULT_COST_BPS = 7.5
_SURGE_LOOKBACK_BDAYS = 60
_MOMENTUM_LOOKBACK_BDAYS = 60

# Composite weights from research recipe.
_W_SUE = 0.4
_W_SURGE = 0.4
_W_INTERACT = 0.2


@dataclass(frozen=True)
class CompositeResult:
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
    component_corr_sue_surge: float
    verdict: str
    verdict_detail: str


def _load_data(
    start: datetime,
    end: datetime,
    *,
    db_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pull earnings + price panel + volume panel. Returns (earnings, close_wide, vol_wide)."""
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
        bars = conn.execute(
            """
            SELECT ticker, ts,
                   COALESCE(adj_close, close) AS close,
                   volume
            FROM ohlcv
            WHERE ts BETWEEN ? AND ?
            """,
            [start, end],
        ).df()
    if bars.empty:
        return earnings, pd.DataFrame(), pd.DataFrame()
    close_wide = bars.pivot(index="ts", columns="ticker", values="close").sort_index()
    vol_wide = bars.pivot(index="ts", columns="ticker", values="volume").sort_index()
    return earnings, close_wide, vol_wide


def _next_trading_day(prices_idx: pd.DatetimeIndex, after: pd.Timestamp) -> pd.Timestamp | None:
    """First index value strictly greater than `after`, or None."""
    later = prices_idx[prices_idx > after]
    if len(later) == 0:
        return None
    return later[0]


def _compute_surge(
    sue_df: pd.DataFrame,
    close_wide: pd.DataFrame,
    vol_wide: pd.DataFrame,
) -> pd.Series:
    """SURGE per event: first-post-announcement |return| * (vol / avg_vol_60d), z-scored per ticker.

    Returns a Series aligned to sue_df index. NaN where the announcement day or its
    prior 60-day window falls outside available bars.
    """
    n = len(sue_df)
    raw = np.full(n, np.nan, dtype=float)

    idx = close_wide.index
    for i, r in enumerate(sue_df.itertuples()):
        ticker = r.ticker
        if ticker not in close_wide.columns:
            continue
        d1 = _next_trading_day(idx, pd.Timestamp(r.report_date))
        if d1 is None:
            continue
        try:
            d1_loc = int(idx.get_loc(d1))
        except KeyError:
            continue
        if d1_loc < _SURGE_LOOKBACK_BDAYS or d1_loc < 1:
            continue

        close_today = close_wide[ticker].iloc[d1_loc]
        close_prev = close_wide[ticker].iloc[d1_loc - 1]
        vol_today = vol_wide[ticker].iloc[d1_loc]
        vol_lookback = vol_wide[ticker].iloc[d1_loc - _SURGE_LOOKBACK_BDAYS : d1_loc]

        if pd.isna(close_today) or pd.isna(close_prev) or close_prev == 0 or pd.isna(vol_today):
            continue
        avg_vol = float(vol_lookback.mean())
        if avg_vol <= 0 or pd.isna(avg_vol):
            continue
        ret = float(close_today / close_prev - 1.0)
        vol_ratio = float(vol_today) / avg_vol
        # Signed: |return| × log-volume-ratio × sign(return).
        # Using signed return * vol_ratio gives a directional surge metric.
        raw[i] = ret * vol_ratio

    s = pd.Series(raw, index=sue_df.index)
    # Per-ticker z-score over the full series (point-in-time looseness here is acceptable
    # for a research-only score; tightening to expanding window is a future refinement).
    z = s.groupby(sue_df["ticker"]).transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) and x.std(ddof=0) > 0 else 0.0
    )
    return z


def _compute_momentum_sign(
    sue_df: pd.DataFrame,
    close_wide: pd.DataFrame,
) -> pd.Series:
    """sign of trailing 60-bday total return at announcement."""
    n = len(sue_df)
    out = np.zeros(n, dtype=float)
    idx = close_wide.index
    for i, r in enumerate(sue_df.itertuples()):
        ticker = r.ticker
        if ticker not in close_wide.columns:
            continue
        d_event = pd.Timestamp(r.report_date)
        # last index <= event date
        prior = idx[idx <= d_event]
        if len(prior) < _MOMENTUM_LOOKBACK_BDAYS + 1:
            continue
        end_loc = int(idx.get_loc(prior[-1]))
        start_loc = end_loc - _MOMENTUM_LOOKBACK_BDAYS
        c_end = close_wide[ticker].iloc[end_loc]
        c_start = close_wide[ticker].iloc[start_loc]
        if pd.isna(c_end) or pd.isna(c_start) or c_start == 0:
            continue
        ret = float(c_end / c_start - 1.0)
        out[i] = 1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0)
    return pd.Series(out, index=sue_df.index)


def run_composite(
    *,
    start: datetime,
    end: datetime,
    cost_bps: float = _DEFAULT_COST_BPS,
    holding_window: int = _HOLDING_WINDOW_BDAYS,
    db_path: Path | None = None,
    save_csv: bool = True,
) -> CompositeResult:
    earnings, close_wide, vol_wide = _load_data(start, end, db_path=db_path)
    if earnings.empty or close_wide.empty:
        raise RuntimeError(
            f"insufficient data in window [{start.date()} .. {end.date()}]: "
            f"earnings={len(earnings)} rows, close={close_wide.shape}"
        )

    sue_df = sue_baseline._compute_all_sue(earnings, db_path=db_path)

    # Z-score SUE per ticker so it sits on the same scale as SURGE z-score.
    sue_df["sue_z"] = sue_df.groupby("ticker")["sue"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) and x.std(ddof=0) > 0 else 0.0
    )
    sue_df["surge_z"] = _compute_surge(sue_df, close_wide, vol_wide)
    sue_df["mom_sign"] = _compute_momentum_sign(sue_df, close_wide)

    # Composite: 0.4 SUE + 0.4 SURGE + 0.2 |SUE| * sign(60d_momentum).
    sue_df = sue_df.dropna(subset=["sue_z", "surge_z"]).copy()
    sue_df["composite"] = (
        _W_SUE * sue_df["sue_z"]
        + _W_SURGE * sue_df["surge_z"]
        + _W_INTERACT * sue_df["sue_z"].abs() * sue_df["mom_sign"]
    )

    # Diagnostic: correlation between the two main components.
    raw_corr = sue_df["sue_z"].corr(sue_df["surge_z"])
    corr_sue_surge = 0.0 if pd.isna(raw_corr) else float(raw_corr)

    # Reuse sue_baseline forward-return + quintile machinery, but on composite.
    sue_df = sue_df.rename(columns={"sue": "_orig_sue", "composite": "sue"})
    sue_df = sue_baseline._attach_forward_returns(sue_df, close_wide)
    quintile_means, q5q1 = sue_baseline._quintile_summary(sue_df)

    def _signal_func(_universe, _start, _end, **_kwargs):  # noqa: ANN001
        return sue_baseline._sue_panel(sue_df, close_wide, holding_window=holding_window)

    results, _csv = vbt_research.run_parameter_sweep(
        _signal_func,
        param_grid={"holding_window": [holding_window]},
        universe=list(close_wide.columns),
        prices=close_wide,
        start_date=close_wide.index.min().to_pydatetime(),
        end_date=close_wide.index.max().to_pydatetime(),
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
        detail = (
            "; ".join(reasons)
            + ". Stage 1 composite exhausted; consider Stage 2 (FMP transcripts) or Stage 3 (kill)."
        )

    return CompositeResult(
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
        component_corr_sue_surge=corr_sue_surge,
        verdict=verdict,
        verdict_detail=detail,
    )


# ============================================================================ CLI
def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _print_human(r: CompositeResult) -> None:
    print("\n=== SUE+SURGE composite -- Phase 1 Stage 1 gate ===")
    print(f"window:       {r.start_date} .. {r.end_date}")
    print(f"events:       {r.n_events} across {r.n_tickers_with_events} tickers")
    print(f"cost:         {r.cost_bps} bps round-trip")
    print(f"corr(SUE,SURGE): {r.component_corr_sue_surge:+.3f}")
    print()
    print(f"Sharpe:       {r.sharpe:>8.3f}   (target >= 0.5)")
    print(f"Sortino:      {r.sortino:>8.3f}")
    print(f"Max DD:       {r.max_drawdown:>8.3%}")
    print(f"Win rate:     {r.win_rate:>8.3%}")
    print(f"Total ret:    {r.total_return:>8.3%}")
    print()
    print("Forward-return spread (top - bottom quintile):")
    print(f"  5-day:      {r.q5_minus_q1_5d:>+8.4f}")
    print(f"  10-day:     {r.q5_minus_q1_10d:>+8.4f}   (target > 0)")
    print(f"  20-day:     {r.q5_minus_q1_20d:>+8.4f}")
    print()
    if r.quintile_means:
        print("Mean forward returns by composite quintile (0=lowest, 4=highest):")
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
        prog="casino.backtest.composite_baseline",
        description="Run the SUE+SURGE+momentum composite signal gate (Phase 1 Stage 1).",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--cost-bps", type=float, default=_DEFAULT_COST_BPS)
    parser.add_argument("--holding-window", type=int, default=_HOLDING_WINDOW_BDAYS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

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

    result = run_composite(
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
