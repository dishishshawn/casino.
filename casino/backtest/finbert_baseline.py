"""Phase-1 Stage-1 text-PEAD kill-test gate.

The decisive empirical test for the EDLLLS (earnings-drift LLM long-short) project.
Pure SUE PEAD (-0.42 Sharpe) and SUE+SURGE composite (-0.36 Sharpe) both FAILed on
S&P 500. Per memory `fmp_decision_2026-05-05.md`, the only $0 lever left is to test
the text-based PEAD (PEAD.txt, Meursault et al. 2023 *JFQA*) hypothesis using:

    * Free Hugging Face transcript snapshot (kurry/sp500_earnings_transcripts).
    * 2018-vintage FinBERT (ProsusAI/finbert) — knowledge cutoff predates our 2018-2026
      backtest window, so no LLM look-ahead bias.

**Kill threshold:** Q4-Q0 20-day forward-return spread < +1.0% gross →
permanently kill EDLLLS on US large caps. Otherwise, the signal is at least
*directionally* present and a paid Stage-2 (FMP Ultimate or institutional
transcripts) is *defensible* — though Sharpe ≥ 0.5 still gates production.

Run:
    uv run python -m casino.backtest.finbert_baseline
    uv run python -m casino.backtest.finbert_baseline --json
    uv run python -m casino.backtest.finbert_baseline --no-save
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
_KILL_SPREAD_THRESHOLD = 0.01  # +1.0% Q4-Q0 20d gross


@dataclass(frozen=True)
class FinbertResult:
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
    verdict: str  # "PASS" | "FAIL" | "KILL"
    verdict_detail: str


def _load_data(
    start: datetime,
    end: datetime,
    *,
    db_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull joined finbert + earnings + close prices.

    Joins finbert_scores (event_date) to the nearest *prior* earnings report_date
    per ticker (within ±5 days) so each text-score is tagged with the actual
    earnings event we will trade off. Falls back to raw event_date if no earnings
    match is found (the score still has timing info).
    """
    with store.get_duckdb_conn(db_path, read_only=True) as conn:
        scores = conn.execute(
            """
            SELECT ticker, event_date, score_net, score_pos, score_neg
            FROM finbert_scores
            WHERE event_date BETWEEN ? AND ?
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
    if prices.empty or scores.empty:
        return scores, pd.DataFrame()
    prices_wide = prices.pivot(index="ts", columns="ticker", values="close").sort_index()
    return scores, prices_wide


def _attach_forward_returns(
    df: pd.DataFrame,
    prices_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Add fwd_5d / fwd_10d / fwd_20d columns based on event_date."""
    out = df.copy()
    out = out.rename(columns={"event_date": "report_date"})
    return sue_baseline._attach_forward_returns(out, prices_wide)


def _quintile_summary(
    score_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    if score_df.empty or score_df["score_net"].nunique() < 5:
        return {}, {"5d": 0.0, "10d": 0.0, "20d": 0.0}
    df = score_df.copy()
    df["quintile"] = pd.qcut(df["score_net"], q=5, labels=False, duplicates="drop")
    means = df.groupby("quintile")[["fwd_5d", "fwd_10d", "fwd_20d"]].mean()
    qs = {str(int(q)): {k: float(v) for k, v in row.items()} for q, row in means.iterrows()}
    spread: dict[str, float] = {}
    for h in ("5d", "10d", "20d"):
        col = f"fwd_{h}"
        try:
            spread[h] = float(means.loc[4, col] - means.loc[0, col])
        except KeyError:
            spread[h] = 0.0
    return qs, spread


def _score_panel(
    score_df: pd.DataFrame,
    prices_wide: pd.DataFrame,
    *,
    holding_window: int = _HOLDING_WINDOW_BDAYS,
) -> pd.DataFrame:
    """Project FinBERT scores onto the price index for vectorbt sweep."""
    panel = pd.DataFrame(np.nan, index=prices_wide.index, columns=prices_wide.columns)
    for r in score_df.itertuples():
        rd = r.report_date if hasattr(r, "report_date") else r.event_date
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
        panel.iloc[int(idx) : int(end_idx), col_loc] = r.score_net
    return panel.ffill(limit=holding_window)


def run_finbert_gate(
    *,
    start: datetime,
    end: datetime,
    cost_bps: float = _DEFAULT_COST_BPS,
    holding_window: int = _HOLDING_WINDOW_BDAYS,
    db_path: Path | None = None,
    save_csv: bool = True,
) -> FinbertResult:
    scores, prices_wide = _load_data(start, end, db_path=db_path)
    if scores.empty or prices_wide.empty:
        raise RuntimeError(
            f"insufficient data in [{start.date()}..{end.date()}]: "
            f"finbert_scores={len(scores)} rows, prices={prices_wide.shape}. "
            "Run ingest_transcripts_hf + finbert_score first."
        )

    score_df = _attach_forward_returns(scores, prices_wide)
    quintile_means, q5q1 = _quintile_summary(score_df)

    def _signal_func(_universe, _start, _end, **_kwargs):  # noqa: ANN001
        # Renamed back to event_date so _score_panel finds report_date col.
        panel_df = score_df.rename(columns={"report_date": "event_date"})
        return _score_panel(panel_df, prices_wide, holding_window=holding_window)

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

    spread_20d = q5q1.get("20d", 0.0)
    pass_sharpe = sharpe >= 0.5
    pass_spread = q5q1.get("10d", 0.0) > 0
    is_kill = spread_20d < _KILL_SPREAD_THRESHOLD

    if pass_sharpe and pass_spread:
        verdict = "PASS"
        detail = (
            f"Sharpe {sharpe:.2f} >= 0.5 AND Q5-Q1 10d spread {q5q1['10d']:+.4f} > 0. "
            "Text-PEAD signal exists net of cost; Stage 2 (paid transcripts) is defensible."
        )
    elif is_kill:
        verdict = "KILL"
        detail = (
            f"Q5-Q1 20d gross spread {spread_20d:+.4f} < +{_KILL_SPREAD_THRESHOLD:.4f}. "
            "Even gross of cost, text-PEAD does not exist on S&P 500 large-caps with 2018-vintage FinBERT. "
            "Permanently kill EDLLLS. Pivot to free signals (Branch C)."
        )
    else:
        verdict = "FAIL"
        reasons = []
        if not pass_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} < 0.5")
        if not pass_spread:
            reasons.append(f"Q5-Q1 10d spread {q5q1.get('10d', 0.0):+.4f} <= 0")
        detail = (
            "; ".join(reasons)
            + f". 20d gross spread {spread_20d:+.4f} >= +{_KILL_SPREAD_THRESHOLD:.4f}: signal is "
            "directionally present but not cost-survivable. FMP Ultimate ($1788/yr) NOT justified at "
            "<$150k account; consider Stage 2 only at higher NAV with Sharpe-improvement path."
        )

    return FinbertResult(
        n_events=int(len(score_df)),
        n_tickers_with_events=int(score_df["ticker"].nunique()),
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


def _print_human(r: FinbertResult) -> None:
    print("\n=== FinBERT text-PEAD -- Phase 1 Stage 1 KILL-TEST gate ===")
    print(f"window:       {r.start_date} .. {r.end_date}")
    print(f"events:       {r.n_events} across {r.n_tickers_with_events} tickers")
    print(f"cost:         {r.cost_bps} bps round-trip")
    print()
    print(f"Sharpe:       {r.sharpe:>8.3f}   (PASS target >= 0.5)")
    print(f"Sortino:      {r.sortino:>8.3f}")
    print(f"Max DD:       {r.max_drawdown:>8.3%}")
    print(f"Win rate:     {r.win_rate:>8.3%}")
    print(f"Total ret:    {r.total_return:>8.3%}")
    print()
    print("Forward-return spread (top - bottom quintile):")
    print(f"  5-day:      {r.q5_minus_q1_5d:>+8.4f}")
    print(f"  10-day:     {r.q5_minus_q1_10d:>+8.4f}")
    print(f"  20-day:     {r.q5_minus_q1_20d:>+8.4f}   (KILL if < +0.0100)")
    print()
    if r.quintile_means:
        print("Mean forward returns by FinBERT-net quintile (0=lowest, 4=highest):")
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
        prog="casino.backtest.finbert_baseline",
        description="Run the FinBERT text-PEAD kill-test gate (Phase 1 Stage 1).",
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
                "SELECT GREATEST(min_f, min_o), LEAST(max_f, max_o) FROM ("
                "  SELECT min(event_date) AS min_f, max(event_date) AS max_f FROM finbert_scores"
                ") f CROSS JOIN ("
                "  SELECT min(ts) AS min_o, max(ts) AS max_o FROM ohlcv"
                ") o"
            ).fetchone()
        if row is None or row[0] is None:
            logger.error(
                "DuckDB has no overlapping finbert_scores + ohlcv data; "
                "run ingest_transcripts_hf + finbert_score first"
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

    result = run_finbert_gate(
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
