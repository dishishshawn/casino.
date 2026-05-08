"""Branch C step 1.5 - TSMOM parameter-sensitivity grid.

Runs a 90-cell grid (6 lookbacks x 5 vol-targets x 3 rebal cadences) on the same
strict-OOS universe used by ``tsmom_baseline``. The "chosen" production config
(blend lookbacks (21, 63, 126, 252) at 10% vol-target, monthly rebal) is run
*alongside* the grid as a 91st cell so its Sharpe can be percentile-ranked
against the 90 single-lookback cells.

Acceptance per PRD section 3 hard-restart rule: chosen config must land in the
top quartile of grid Sharpes. Otherwise the signal is overfit and downstream
work (tasks 39, 40, 41, 42) is BLOCKED until Branch C restarts.

Run:
    uv run python -m casino.backtest.tsmom_sensitivity
    uv run python -m casino.backtest.tsmom_sensitivity --no-save
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

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
from casino.signals import ts_momentum

_BDAYS_PER_YEAR = 252
_DEFAULT_COST_BPS = 12.0
_DEFAULT_UNIVERSE_FILE = "universe_tsmom.txt"

# Single-lookback grid points: 1m, 3m, 6m, 9m, 12m, 15m (in business days).
GRID_LOOKBACKS_BDAYS: tuple[tuple[int, ...], ...] = (
    (21,),
    (63,),
    (126,),
    (189,),
    (252,),
    (315,),
)
GRID_LOOKBACK_LABELS: dict[tuple[int, ...], str] = {
    (21,): "1",
    (63,): "3",
    (126,): "6",
    (189,): "9",
    (252,): "12",
    (315,): "15",
}
GRID_VOL_TARGETS: tuple[float, ...] = (0.06, 0.08, 0.10, 0.12, 0.15)
GRID_REBALANCE: tuple[str, ...] = ("weekly", "biweekly", "monthly")

# The production "chosen" point: blend of 1/3/6/12m lookbacks, 10% vol target,
# monthly rebal. Reported separately to be ranked vs the grid.
CHOSEN_LOOKBACKS: tuple[int, ...] = (21, 63, 126, 252)
CHOSEN_VOL_TARGET: float = 0.10
CHOSEN_REBALANCE: str = "monthly"
CHOSEN_LOOKBACK_LABEL: str = "blend(1,3,6,12)"

# n_trials for DSR per cell: the actual grid size (90).
N_TRIALS = 90


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridCell:
    lookback_bdays: tuple[int, ...]
    lookback_label: str
    vol_target: float
    rebalance: str
    sharpe: float
    deflated_sharpe: float
    deflated_p_value: float
    max_drawdown: float
    ann_return: float
    total_return: float
    ann_vol: float

    def to_row(self) -> dict[str, float | str]:
        return {
            "lookback_months": self.lookback_label,
            "vol_target_pct": round(self.vol_target * 100, 2),
            "rebalance": self.rebalance,
            "sharpe": round(self.sharpe, 4),
            "deflated_sharpe": round(self.deflated_sharpe, 4),
            "deflated_p_value": round(self.deflated_p_value, 6),
            "max_drawdown": round(self.max_drawdown, 4),
            "ann_return": round(self.ann_return, 4),
            "total_return": round(self.total_return, 4),
            "ann_vol": round(self.ann_vol, 4),
        }


# ---------------------------------------------------------------------------
def _annualized_return(total_return: float, n_days: int) -> float:
    if n_days <= 0:
        return float("nan")
    years = max(n_days / 365.25, 1e-6)
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def run_cell(
    prices: pd.DataFrame,
    *,
    lookbacks: tuple[int, ...],
    lookback_label: str,
    vol_target: float,
    rebalance: str,
    cost_bps: float,
    n_trials: int = N_TRIALS,
) -> GridCell:
    """Run a single grid cell and return its metrics."""
    weights = ts_momentum.compute_tsmom_panel(
        prices,
        lookbacks=lookbacks,
        target_vol=vol_target,
    )
    rebal: Literal["monthly", "biweekly", "weekly", "daily"] = rebalance  # type: ignore[assignment]
    metrics = _backtest_weights(weights, prices, cost_bps=cost_bps, rebalance=rebal)
    daily_returns = _backtest_weights_returns(weights, prices, cost_bps=cost_bps, rebalance=rebal)
    sharpe = float(metrics.get("sharpe", float("nan")))
    if daily_returns.empty or not np.isfinite(sharpe):
        deflated = float("nan")
        p_value = float("nan")
    else:
        deflate_result = ds.haircut_sharpe(
            sharpe,
            {
                "n_trials": n_trials,
                "n_observations": len(daily_returns),
                "returns": daily_returns.tolist(),
            },
        )
        deflated = float(deflate_result["deflated"])
        p_value = float(deflate_result["p_value"])

    n_days = (prices.index.max() - prices.index.min()).days if not prices.empty else 0
    ann_ret = _annualized_return(float(metrics.get("total_return", 0.0)), n_days)

    return GridCell(
        lookback_bdays=lookbacks,
        lookback_label=lookback_label,
        vol_target=vol_target,
        rebalance=rebalance,
        sharpe=sharpe,
        deflated_sharpe=deflated,
        deflated_p_value=p_value,
        max_drawdown=float(metrics.get("max_drawdown", float("nan"))),
        ann_return=ann_ret,
        total_return=float(metrics.get("total_return", float("nan"))),
        ann_vol=float(metrics.get("ann_vol", float("nan"))),
    )


def run_grid(
    prices: pd.DataFrame,
    *,
    cost_bps: float = _DEFAULT_COST_BPS,
    lookbacks: tuple[tuple[int, ...], ...] = GRID_LOOKBACKS_BDAYS,
    vol_targets: tuple[float, ...] = GRID_VOL_TARGETS,
    rebalances: tuple[str, ...] = GRID_REBALANCE,
    lookback_labels: dict[tuple[int, ...], str] | None = None,
    n_trials: int = N_TRIALS,
) -> list[GridCell]:
    """Run the full sensitivity grid; returns one GridCell per parameter combo."""
    labels = lookback_labels or GRID_LOOKBACK_LABELS
    cells: list[GridCell] = []
    total = len(lookbacks) * len(vol_targets) * len(rebalances)
    i = 0
    for lb in lookbacks:
        for vt in vol_targets:
            for rb in rebalances:
                i += 1
                cell = run_cell(
                    prices,
                    lookbacks=lb,
                    lookback_label=labels.get(lb, str(lb)),
                    vol_target=vt,
                    rebalance=rb,
                    cost_bps=cost_bps,
                    n_trials=n_trials,
                )
                logger.info(
                    "[{}/{}] lb={}m vt={}% rb={} sharpe={:.3f}",
                    i,
                    total,
                    cell.lookback_label,
                    int(round(vt * 100)),
                    rb,
                    cell.sharpe,
                )
                cells.append(cell)
    return cells


def chosen_percentile_rank(grid_cells: list[GridCell], chosen_sharpe: float) -> float:
    """Return the percentile rank in [0, 1] of `chosen_sharpe` vs the grid Sharpes.

    Uses the standard "fraction of cells with sharpe < chosen" definition
    (excluding cells where the chosen point is identical). Top-quartile
    threshold is 0.75.
    """
    valid = [c.sharpe for c in grid_cells if np.isfinite(c.sharpe)]
    if not valid:
        return float("nan")
    n = len(valid)
    less = sum(1 for s in valid if s < chosen_sharpe)
    return float(less) / float(n)


# ---------------------------------------------------------------------------
def grid_to_dataframe(cells: list[GridCell]) -> pd.DataFrame:
    return pd.DataFrame([c.to_row() for c in cells])


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GFM markdown table without `tabulate` dep."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_outputs(
    grid_cells: list[GridCell],
    chosen_cell: GridCell,
    *,
    out_dir: Path,
    cost_bps: float,
    universe: list[str],
) -> tuple[Path, Path]:
    """Write CSV and Markdown reports. Returns (csv_path, md_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df = grid_to_dataframe(grid_cells)
    csv_path = out_dir / "tsmom_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    logger.info("grid CSV written to {}", csv_path)

    sharpes = np.asarray(
        [c.sharpe for c in grid_cells if np.isfinite(c.sharpe)],
        dtype=float,
    )
    if sharpes.size == 0:
        raise RuntimeError("grid produced no finite Sharpes; cannot rank.")

    grid_mean = float(np.mean(sharpes))
    grid_std = float(np.std(sharpes, ddof=1)) if sharpes.size > 1 else float("nan")
    grid_min = float(np.min(sharpes))
    grid_max = float(np.max(sharpes))
    top_q_cutoff = float(np.quantile(sharpes, 0.75))
    pct = chosen_percentile_rank(grid_cells, chosen_cell.sharpe)
    is_top_quartile = chosen_cell.sharpe >= top_q_cutoff
    verdict = "PASS" if is_top_quartile else "FAIL"

    md_path = out_dir / "tsmom_sensitivity.md"
    lines: list[str] = []
    lines.append("# TSMOM Sensitivity Grid (Branch C step 1.5)")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(UTC).isoformat()}")
    lines.append(f"- Universe: {', '.join(universe)}")
    lines.append(f"- Cost: {cost_bps} bps round-trip")
    lines.append(f"- Cells: {len(grid_cells)} (single-lookback grid)")
    lines.append(f"- DSR n_trials: {N_TRIALS}")
    lines.append("")
    lines.append("## Grid summary (Sharpe distribution)")
    lines.append("")
    lines.append(f"- mean: {grid_mean:.4f}")
    lines.append(f"- std:  {grid_std:.4f}")
    lines.append(f"- min:  {grid_min:.4f}")
    lines.append(f"- max:  {grid_max:.4f}")
    lines.append(f"- top-quartile cutoff (75th pct): {top_q_cutoff:.4f}")
    lines.append("")
    lines.append("## Chosen production config")
    lines.append("")
    lines.append(
        f"- lookback: {CHOSEN_LOOKBACK_LABEL}  "
        f"({', '.join(str(x) for x in CHOSEN_LOOKBACKS)} bdays)"
    )
    lines.append(f"- vol-target: {chosen_cell.vol_target * 100:.1f}%")
    lines.append(f"- rebalance: {chosen_cell.rebalance}")
    lines.append(f"- Sharpe: {chosen_cell.sharpe:.4f}")
    lines.append(f"- Deflated Sharpe: {chosen_cell.deflated_sharpe:.4f}")
    lines.append(f"- Max DD: {chosen_cell.max_drawdown:.4f}")
    lines.append(f"- Ann return: {chosen_cell.ann_return:.4f}")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"- chosen_sharpe = {chosen_cell.sharpe:.4f}, "
        f"grid_mean = {grid_mean:.4f}, "
        f"top_quartile_cutoff = {top_q_cutoff:.4f}, "
        f"percentile = {pct:.4f}"
    )
    lines.append(f"- VERDICT: **{verdict}**")
    if is_top_quartile:
        lines.append(
            "- Chosen config lies in top quartile of grid Sharpes; signal is not "
            "obviously overfit. Branch C continues to ensemble step (task 39)."
        )
    else:
        lines.append(
            "- Chosen config is below the top-quartile cutoff. Per PRD section 3 "
            "hard-restart rule, the TSMOM signal is overfit and downstream tasks "
            "(39, 40, 41, 42) are BLOCKED until Branch C restarts."
        )
    lines.append("")
    lines.append("## Top 10 cells by Sharpe")
    lines.append("")
    df_sorted = df.sort_values("sharpe", ascending=False).head(10)
    lines.append(_df_to_markdown(df_sorted))
    lines.append("")
    lines.append("## Bottom 5 cells by Sharpe")
    lines.append("")
    df_bot = df.sort_values("sharpe", ascending=True).head(5)
    lines.append(_df_to_markdown(df_bot))
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("grid Markdown written to {}", md_path)

    return csv_path, md_path


# ---------------------------------------------------------------------------
def _load_prices(*, universe: list[str], db_path: Path | None = None) -> pd.DataFrame:
    """Load OHLCV panel covering full available range for the universe."""
    with store.get_duckdb_conn(db_path, read_only=True) as conn:
        placeholders = ",".join("?" * len(universe))
        row = conn.execute(
            f"SELECT min(ts), max(ts) FROM ohlcv WHERE ticker IN ({placeholders})",
            universe,
        ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(
            f"no OHLCV in DuckDB for universe {universe}. Run ingest_yfinance first."
        )
    start, end = row[0], row[1]
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return ts_momentum.load_ohlcv_panel(start=start, end=end, universe=universe, db_path=db_path)


def _print_human(
    grid_cells: list[GridCell],
    chosen_cell: GridCell,
) -> None:
    sharpes = np.asarray([c.sharpe for c in grid_cells if np.isfinite(c.sharpe)], dtype=float)
    grid_mean = float(np.mean(sharpes))
    top_q = float(np.quantile(sharpes, 0.75))
    pct = chosen_percentile_rank(grid_cells, chosen_cell.sharpe)
    verdict = "PASS" if chosen_cell.sharpe >= top_q else "FAIL"

    print("\n=== TSMOM sensitivity grid ===")
    print(f"cells:            {len(grid_cells)}")
    print(f"grid mean Sharpe: {grid_mean:.4f}")
    print(f"grid Sharpe range: [{float(sharpes.min()):.4f}, {float(sharpes.max()):.4f}]")
    print(f"top-quartile cutoff: {top_q:.4f}")
    print()
    print("Chosen config (blend(1,3,6,12) / 10% vol / monthly):")
    print(f"  Sharpe:           {chosen_cell.sharpe:.4f}")
    print(f"  Deflated Sharpe:  {chosen_cell.deflated_sharpe:.4f}")
    print(f"  Max DD:           {chosen_cell.max_drawdown:.4f}")
    print(f"  Ann return:       {chosen_cell.ann_return:.4f}")
    print()
    print(
        f"VERDICT: chosen_sharpe={chosen_cell.sharpe:.4f}, "
        f"grid_mean={grid_mean:.4f}, "
        f"top_quartile_cutoff={top_q:.4f}, "
        f"percentile={pct:.4f}, "
        f"verdict={verdict}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.backtest.tsmom_sensitivity",
        description="TSMOM 90-cell parameter sensitivity grid (Branch C step 1.5).",
    )
    parser.add_argument(
        "--universe-file",
        default=_DEFAULT_UNIVERSE_FILE,
        help=f"Path to newline-delimited tickers file (default {_DEFAULT_UNIVERSE_FILE}).",
    )
    parser.add_argument("--cost-bps", type=float, default=_DEFAULT_COST_BPS)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing CSV / Markdown to reports/.",
    )
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Run only the 90-cell grid; skip the chosen-config cell.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports",
        help="Output directory for CSV/Markdown (default: reports/).",
    )
    args = parser.parse_args(argv)

    universe = _read_universe_file(Path(args.universe_file))
    if not universe:
        logger.error("universe file {} resolved to no tickers", args.universe_file)
        return 2

    logger.info("loading OHLCV panel for {} tickers...", len(universe))
    prices = _load_prices(universe=universe)
    if prices.empty:
        logger.error("OHLCV panel empty; aborting")
        return 2
    logger.info(
        "panel: {} bars x {} tickers, {} -> {}",
        prices.shape[0],
        prices.shape[1],
        prices.index.min().date(),
        prices.index.max().date(),
    )

    logger.info("running 90-cell grid (this takes ~5-10 min on real data)...")
    grid_cells = run_grid(prices, cost_bps=args.cost_bps)

    if args.grid_only:
        # Use the 12m / 10% / monthly cell as a proxy chosen for human print.
        proxy = next(
            (
                c
                for c in grid_cells
                if c.lookback_label == "12"
                and abs(c.vol_target - CHOSEN_VOL_TARGET) < 1e-9
                and c.rebalance == CHOSEN_REBALANCE
            ),
            None,
        )
        chosen_cell = proxy or grid_cells[0]
    else:
        logger.info("running chosen-config cell (blend lookbacks)...")
        chosen_cell = run_cell(
            prices,
            lookbacks=CHOSEN_LOOKBACKS,
            lookback_label=CHOSEN_LOOKBACK_LABEL,
            vol_target=CHOSEN_VOL_TARGET,
            rebalance=CHOSEN_REBALANCE,
            cost_bps=args.cost_bps,
        )

    _print_human(grid_cells, chosen_cell)

    if not args.no_save:
        out_dir = Path(args.out_dir)
        write_outputs(
            grid_cells,
            chosen_cell,
            out_dir=out_dir,
            cost_bps=args.cost_bps,
            universe=universe,
        )

    sharpes = [c.sharpe for c in grid_cells if np.isfinite(c.sharpe)]
    top_q = float(np.quantile(np.asarray(sharpes), 0.75))
    is_pass = chosen_cell.sharpe >= top_q
    return 0 if is_pass else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
