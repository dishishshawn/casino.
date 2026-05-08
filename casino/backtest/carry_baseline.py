"""Branch C step 2 - Cross-asset carry signal baseline gate.

After the TSMOM sensitivity grid placed the chosen blend at only the 55th
percentile (memory: tsmom_killed_2026-05-07.md), Branch C pivots: carry is
the new lead signal. This module is its standalone gate.

PASS gate (per PRD §3 / task 38 spec):

    Sharpe       >=  0.4    (net of 12 bps round-trip, monthly rebal)
    Deflated SR  >   0      (Bailey-Lopez de Prado, n_trials = 30)
    corr(carry, tsmom) <  0.4  on the *same* OOS sample

The corr-vs-TSMOM check is the diversification requirement: Branch C only
benefits from carry if carry returns are not just a relabeled TSMOM. We
therefore compute TSMOM returns on the same prices/dates and assert
np.corrcoef(carry_rets, tsmom_rets)[0, 1] < 0.4.

Selection-bias note: n_trials = 30 is the spec value. Because carry was
chosen *after* TSMOM died, this is post-selection — the "true" trial count
is closer to 30 + the TSMOM grid (90) ≈ 120. We flag this in the .md output
and recommend re-running with n_trials = 60 if any gate is borderline.

Run:
    uv run python -m casino.backtest.carry_baseline
    uv run python -m casino.backtest.carry_baseline --mode long_only
    uv run python -m casino.backtest.carry_baseline --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

from casino.backtest import deflated_sharpe as ds
from casino.backtest.tsmom_baseline import (
    _backtest_weights,
    _backtest_weights_returns,
    _read_universe_file,
)
from casino.data import store
from casino.signals import carry, ts_momentum

_DEFAULT_COST_BPS = 12.0
_DEFAULT_MODE: Literal["long_short", "long_only"] = "long_short"
_DEFAULT_UNIVERSE_FILE = "universe_tsmom.txt"
_DEFAULT_FRED_SERIES: tuple[str, ...] = ("DGS10", "DGS5", "DGS2", "DGS3MO", "DTB3")
_BDAYS_PER_YEAR = 252

GATE_SHARPE_MIN = 0.4
GATE_DSR_MIN = 0.0  # strictly > 0
GATE_CORR_MAX = 0.4
DEFAULT_N_TRIALS = 30


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CarryResult:
    n_assets: int
    start_date: str
    end_date: str
    mode: str
    cost_bps: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    total_return: float
    annualized_return: float
    annualized_vol: float
    corr_with_tsmom: float
    deflated_sharpe: float
    deflated_p_value: float
    deflated_n_trials: int
    deflated_n_observations: int
    per_asset_avg_weight: dict[str, float] = field(default_factory=dict)
    yearly_sharpe: dict[str, float] = field(default_factory=dict)
    pass_sharpe: bool = False
    pass_dsr: bool = False
    pass_corr: bool = False
    verdict: str = "FAIL"
    verdict_detail: str = ""


# ---------------------------------------------------------------------------
def _annualized_return(total_return: float, n_days: int) -> float:
    if n_days <= 0:
        return float("nan")
    years = max(n_days / 365.25, 1e-6)
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _load_dividends(
    *,
    tickers: list[str],
    start: datetime,
    end: datetime,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load dividends from DuckDB; if empty, fetch lazily via yfinance.

    yfinance fetch is deferred to avoid hard dependency in tests. Stored to
    `dividends` table for idempotency.
    """
    df = store.load_dividends_panel(tickers=tickers, start=start, end=end, db_path=db_path)
    if not df.empty:
        return df

    logger.info("dividends table empty for {}; fetching via yfinance", tickers)
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available; equity carry will be NaN")
        return pd.DataFrame(columns=["ticker", "ts", "amount"])

    rows: list[dict[str, object]] = []
    for t in tickers:
        try:
            divs = yf.Ticker(t).dividends
        except Exception as exc:  # noqa: BLE001 — Yahoo intermittently 404s
            logger.warning("yfinance dividends failed for {}: {}", t, exc)
            continue
        if divs is None or len(divs) == 0:
            continue
        for ts, amount in divs.items():
            ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if ts_py.tzinfo is None:
                ts_py = ts_py.replace(tzinfo=UTC)
            else:
                ts_py = ts_py.astimezone(UTC)
            rows.append({"ticker": t.upper(), "ts": ts_py, "amount": float(amount)})

    if rows:
        n = store.upsert_dividends(rows, db_path=db_path)
        logger.info("upserted {} dividend rows", n)

    return store.load_dividends_panel(tickers=tickers, start=start, end=end, db_path=db_path)


def _load_fred(
    *,
    series_ids: list[str],
    start: datetime,
    end: datetime,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load FRED panel from DuckDB; if empty, fetch via the CSV endpoint."""
    df = store.load_fred_panel(series_ids=series_ids, start=start, end=end, db_path=db_path)
    if not df.empty:
        return df

    logger.info("fred_yields table empty; fetching {}", series_ids)
    from casino.data import ingest_fred

    ingest_fred.ingest_series(
        series_ids,
        start=start.date().isoformat(),
        end=end.date().isoformat() if end else None,
        db_path=db_path,
    )
    return store.load_fred_panel(series_ids=series_ids, start=start, end=end, db_path=db_path)


def run_carry(
    *,
    start: datetime,
    end: datetime,
    universe: list[str],
    cost_bps: float = _DEFAULT_COST_BPS,
    mode: Literal["long_short", "long_only"] = _DEFAULT_MODE,
    n_trials: int = DEFAULT_N_TRIALS,
    db_path: Path | None = None,
    fred_series: list[str] | None = None,
    rebalance: Literal["monthly", "biweekly", "weekly", "daily"] = "monthly",
) -> CarryResult:
    """Compute the carry baseline metrics + verdict on the given window."""
    series = fred_series or list(_DEFAULT_FRED_SERIES)
    prices = ts_momentum.load_ohlcv_panel(start=start, end=end, universe=universe, db_path=db_path)
    if prices.empty:
        raise RuntimeError(f"no OHLCV in [{start.date()}..{end.date()}] for universe {universe}")

    fred_yields = _load_fred(series_ids=series, start=start, end=end, db_path=db_path)
    dividends = _load_dividends(tickers=universe, start=start, end=end, db_path=db_path)

    weights = carry.compute_carry_panel(
        prices, dividends=dividends, fred_yields=fred_yields, mode=mode
    )

    metrics = _backtest_weights(weights, prices, cost_bps=cost_bps, rebalance=rebalance)
    sharpe = float(metrics["sharpe"])
    sortino = float(metrics["sortino"])
    max_dd = float(metrics["max_drawdown"])
    win_rate = float(metrics["win_rate"])
    total_return = float(metrics["total_return"])
    ann_vol = float(metrics["ann_vol"])

    carry_rets = _backtest_weights_returns(weights, prices, cost_bps=cost_bps, rebalance=rebalance)

    # Compute TSMOM returns on the *same* prices/window/cost/rebal for the
    # diversification correlation check.
    tsmom_w = ts_momentum.compute_tsmom_panel(prices, mode=mode)
    tsmom_rets = _backtest_weights_returns(tsmom_w, prices, cost_bps=cost_bps, rebalance=rebalance)

    aligned = pd.concat([carry_rets, tsmom_rets], axis=1, join="inner").dropna()
    if aligned.shape[0] < 60:
        corr_with_tsmom = float("nan")
    else:
        c = float(np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])
        corr_with_tsmom = c

    # Deflated Sharpe.
    deflate_result = ds.haircut_sharpe(
        sharpe,
        {
            "n_trials": n_trials,
            "n_observations": len(carry_rets),
            "returns": carry_rets.tolist(),
        },
    )

    n_days = (prices.index.max() - prices.index.min()).days
    ann_ret = _annualized_return(total_return, n_days)

    avg_w = {col: float(np.nanmean(weights[col].abs())) for col in weights.columns}
    yearly = {
        k.removeprefix("yearly_"): float(v) for k, v in metrics.items() if k.startswith("yearly_")
    }

    pass_sharpe = sharpe >= GATE_SHARPE_MIN
    pass_dsr = bool(deflate_result["deflated"] > GATE_DSR_MIN)
    pass_corr = bool(np.isfinite(corr_with_tsmom) and corr_with_tsmom < GATE_CORR_MAX)

    if pass_sharpe and pass_dsr and pass_corr:
        verdict = "PASS"
        detail = (
            f"Sharpe {sharpe:.2f} >= {GATE_SHARPE_MIN}, "
            f"deflated SR {deflate_result['deflated']:.2f} > 0 (n_trials={n_trials}), "
            f"corr(carry, TSMOM) = {corr_with_tsmom:.2f} < {GATE_CORR_MAX}. "
            f"Carry clears its standalone gate; ensemble work (task 39+) is unblocked."
        )
    else:
        verdict = "FAIL"
        reasons: list[str] = []
        if not pass_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} < {GATE_SHARPE_MIN}")
        if not pass_dsr:
            reasons.append(
                f"deflated SR {deflate_result['deflated']:.2f} <= 0 "
                f"(p={deflate_result['p_value']:.4f}, n_trials={n_trials})"
            )
        if not pass_corr:
            if not np.isfinite(corr_with_tsmom):
                reasons.append("corr(carry, TSMOM) is undefined (insufficient overlap)")
            else:
                reasons.append(f"corr(carry, TSMOM) = {corr_with_tsmom:.2f} >= {GATE_CORR_MAX}")
        detail = "; ".join(reasons)

    return CarryResult(
        n_assets=int(prices.shape[1]),
        start_date=prices.index.min().date().isoformat(),
        end_date=prices.index.max().date().isoformat(),
        mode=mode,
        cost_bps=cost_bps,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_return=total_return,
        annualized_return=float(ann_ret),
        annualized_vol=float(abs(ann_vol)),
        corr_with_tsmom=float(corr_with_tsmom),
        deflated_sharpe=float(deflate_result["deflated"]),
        deflated_p_value=float(deflate_result["p_value"]),
        deflated_n_trials=int(deflate_result["n_trials"]),
        deflated_n_observations=int(deflate_result["n_observations"]),
        per_asset_avg_weight=avg_w,
        yearly_sharpe=yearly,
        pass_sharpe=pass_sharpe,
        pass_dsr=pass_dsr,
        pass_corr=pass_corr,
        verdict=verdict,
        verdict_detail=detail,
    )


# ---------------------------------------------------------------------------
def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GFM markdown table (no `tabulate` dep)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def _result_to_csv_rows(result: CarryResult) -> pd.DataFrame:
    base: dict[str, Any] = {
        "n_assets": result.n_assets,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "mode": result.mode,
        "cost_bps": result.cost_bps,
        "sharpe": round(result.sharpe, 4),
        "sortino": round(result.sortino, 4),
        "max_drawdown": round(result.max_drawdown, 4),
        "win_rate": round(result.win_rate, 4),
        "total_return": round(result.total_return, 4),
        "annualized_return": round(result.annualized_return, 4),
        "annualized_vol": round(result.annualized_vol, 4),
        "corr_with_tsmom": round(result.corr_with_tsmom, 4),
        "deflated_sharpe": round(result.deflated_sharpe, 4),
        "deflated_p_value": round(result.deflated_p_value, 6),
        "deflated_n_trials": result.deflated_n_trials,
        "deflated_n_observations": result.deflated_n_observations,
        "verdict": result.verdict,
    }
    return pd.DataFrame([base])


def write_outputs(result: CarryResult, *, out_dir: Path, universe: list[str]) -> tuple[Path, Path]:
    """Write CSV and Markdown reports. Returns (csv_path, md_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _result_to_csv_rows(result)
    csv_path = out_dir / "carry_baseline.csv"
    df.to_csv(csv_path, index=False)
    logger.info("carry baseline CSV written to {}", csv_path)

    md_path = out_dir / "carry_baseline.md"
    lines: list[str] = []
    lines.append("# Carry Baseline Gate (Branch C step 2)")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(UTC).isoformat()}")
    lines.append(f"- Universe: {', '.join(universe)}")
    lines.append(f"- Window: {result.start_date} .. {result.end_date}")
    lines.append(f"- Cost: {result.cost_bps} bps round-trip")
    lines.append(f"- Mode: {result.mode}")
    lines.append("")
    lines.append("## Selection-bias caveat")
    lines.append("")
    lines.append(
        "Carry was chosen *after* the TSMOM sensitivity grid failed (memory: "
        "tsmom_killed_2026-05-07.md). The DSR n_trials=30 figure used here is "
        "the spec value, but post-TSMOM-failure the true trial count is closer "
        "to **30 + the 90-cell TSMOM grid ≈ 120**. If any of the three gate "
        "components is borderline (Sharpe ≈ 0.4, DSR ≈ 0, corr ≈ 0.4), re-run "
        "with `--n-trials 60` (or higher) before declaring PASS."
    )
    lines.append("")
    lines.append("## Gate components")
    lines.append("")
    lines.append(
        f"- Sharpe:           {result.sharpe:.4f}  "
        f"(target ≥ {GATE_SHARPE_MIN} → {'PASS' if result.pass_sharpe else 'FAIL'})"
    )
    lines.append(
        f"- Deflated Sharpe: {result.deflated_sharpe:.4f}  "
        f"(target > 0, n_trials={result.deflated_n_trials} → "
        f"{'PASS' if result.pass_dsr else 'FAIL'})"
    )
    lines.append(
        f"  - p-value: {result.deflated_p_value:.4f}, n_obs: {result.deflated_n_observations}"
    )
    lines.append(
        f"- corr(carry, TSMOM): {result.corr_with_tsmom:.4f}  "
        f"(target < {GATE_CORR_MAX} → {'PASS' if result.pass_corr else 'FAIL'})"
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- **{result.verdict}**: {result.verdict_detail}")
    lines.append("")
    lines.append("## Full metrics")
    lines.append("")
    lines.append(f"- Sortino:           {result.sortino:.4f}")
    lines.append(f"- Max Drawdown:      {result.max_drawdown:.4f}")
    lines.append(f"- Win rate:          {result.win_rate:.4f}")
    lines.append(f"- Total return:      {result.total_return:.4f}")
    lines.append(f"- Annualized return: {result.annualized_return:.4f}")
    lines.append(f"- Annualized vol:    {result.annualized_vol:.4f}")
    lines.append("")
    if result.per_asset_avg_weight:
        lines.append("## Per-asset average |weight|")
        lines.append("")
        for k, v in sorted(result.per_asset_avg_weight.items()):
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")
    if result.yearly_sharpe:
        lines.append("## Per-year Sharpe / total return")
        lines.append("")
        rows: list[dict[str, Any]] = []
        years = sorted(k for k in result.yearly_sharpe if not k.endswith("_ret"))
        for y in years:
            rows.append(
                {
                    "year": y,
                    "sharpe": round(result.yearly_sharpe.get(y, float("nan")), 3),
                    "total_return": round(result.yearly_sharpe.get(f"{y}_ret", float("nan")), 4),
                }
            )
        if rows:
            lines.append(_df_to_markdown(pd.DataFrame(rows)))
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("carry baseline Markdown written to {}", md_path)
    return csv_path, md_path


def _print_human(r: CarryResult) -> None:
    print("\n=== Carry baseline -- Branch C step 2 gate ===")
    print(f"window:      {r.start_date} .. {r.end_date}")
    print(f"assets:      {r.n_assets}  (mode={r.mode})")
    print(f"cost:        {r.cost_bps} bps round-trip")
    print()
    print(
        f"Sharpe:           {r.sharpe:>+8.4f}  "
        f"(PASS target >= {GATE_SHARPE_MIN}; {'PASS' if r.pass_sharpe else 'FAIL'})"
    )
    print(
        f"Deflated SR:      {r.deflated_sharpe:>+8.4f}  "
        f"(PASS target > 0, n_trials={r.deflated_n_trials}, "
        f"n_obs={r.deflated_n_observations}; {'PASS' if r.pass_dsr else 'FAIL'})"
    )
    print(f"  p-value:        {r.deflated_p_value:>+8.4f}")
    print(
        f"Corr vs TSMOM:    {r.corr_with_tsmom:>+8.4f}  "
        f"(PASS target < {GATE_CORR_MAX}; {'PASS' if r.pass_corr else 'FAIL'})"
    )
    print()
    print(f"Sortino:          {r.sortino:>+8.4f}")
    print(f"Max DD:           {r.max_drawdown:>+8.4%}")
    print(f"Win rate:         {r.win_rate:>+8.4%}")
    print(f"Total return:     {r.total_return:>+8.4%}")
    print(f"Ann. return:      {r.annualized_return:>+8.4%}")
    print(f"Ann. vol:         {r.annualized_vol:>+8.4%}")
    print()
    if r.per_asset_avg_weight:
        print("Avg |weight| per asset:")
        for k, v in sorted(r.per_asset_avg_weight.items()):
            print(f"  {k:<5}  {v:>+7.4f}")
    if r.yearly_sharpe:
        print()
        print("Per-year Sharpe / total return:")
        years = sorted(k for k in r.yearly_sharpe if not k.endswith("_ret"))
        for y in years:
            s = r.yearly_sharpe.get(y, float("nan"))
            t = r.yearly_sharpe.get(f"{y}_ret", float("nan"))
            print(f"  {y}:  Sharpe {s:>+6.3f}   return {t:>+7.2%}")
    print()
    print(f"VERDICT: {r.verdict}")
    print(f"  {r.verdict_detail}")


# ---------------------------------------------------------------------------
def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.backtest.carry_baseline",
        description="Run the carry signal baseline gate (Branch C step 2).",
    )
    parser.add_argument("--start", default=None, help="ISO date; default: full OHLCV range")
    parser.add_argument("--end", default=None, help="ISO date; default: full OHLCV range")
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
        "--rebalance",
        choices=("daily", "weekly", "biweekly", "monthly"),
        default="monthly",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"DSR trial count (spec default {DEFAULT_N_TRIALS}; bump to 60+ if borderline)",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--out-dir", default="reports", help="Output directory (default: reports/)")
    args = parser.parse_args(argv)

    universe = _read_universe_file(Path(args.universe_file))
    if not universe:
        logger.error("universe file {} resolved to no tickers", args.universe_file)
        return 2

    if args.start is None or args.end is None:
        with store.get_duckdb_conn(read_only=True) as conn:
            placeholders = ",".join("?" * len(universe))
            row = conn.execute(
                f"SELECT min(ts), max(ts) FROM ohlcv WHERE ticker IN ({placeholders})",
                universe,
            ).fetchone()
        if row is None or row[0] is None:
            logger.error("no OHLCV in DuckDB for universe {}; ingest first", universe)
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

    result = run_carry(
        start=start,
        end=end,
        universe=universe,
        cost_bps=args.cost_bps,
        mode=args.mode,
        n_trials=args.n_trials,
        rebalance=args.rebalance,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        _print_human(result)

    if not args.no_save:
        out_dir = Path(args.out_dir)
        write_outputs(result, out_dir=out_dir, universe=universe)

    return 0 if result.verdict == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
