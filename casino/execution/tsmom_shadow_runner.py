"""TSMOM shadow runner — Branch C, Option B parallel experiment.

Mirrors ``casino.execution.tsmom_runner`` but:

* Computes weights with the **regime-filtered** signal
  (``casino.signals.ts_momentum_regime.compute_tsmom_regime_panel``) — bond
  legs (TLT, IEF) zero out when the 10y-3m yield-curve slope is inverted.
* Routes every order through a **simulated broker**
  (``casino.execution.sim_broker.SimBroker``) — no Alpaca calls. Fills are
  deterministic from DuckDB OHLCV at the next-day open, stops are
  enforced bar-by-bar by the sim's ``mark_to_market``.
* Persists everything keyed by ``run_id="Belfort"`` so the
  live bot's ``DiCaprio`` row in ``paper_clock`` is never touched.

The point of this companion runner is to produce a second day-30 verdict
the operator can compare to the live vanilla TSMOM verdict. **COMMIT/KILL
applies per-run-id** — the operator decides what to do if they diverge.

The shadow does NOT submit live orders, does NOT touch
``casino.execution.book`` (the live bot's internal book), and does NOT
import from any live-trading code path that mutates the standard
``orders``/``positions`` tables.

CLI:

    uv run python -m casino.execution.tsmom_shadow_runner
        # Standard daily-or-rebal step. On a rebal day, computes weights
        # and submits sim orders. Off rebal day, just marks-to-market.

    uv run python -m casino.execution.tsmom_shadow_runner --force
        # Force off-rebal-day rebal (used when starting the shadow, since
        # the start day is likely not a month-end).

    uv run python -m casino.execution.tsmom_shadow_runner --catchup-from 2026-05-08
        # Backfill from a prior date forward through DuckDB OHLCV. Use
        # this if the operator misses a daily mark and wants to catch the
        # sim back up to today.

Hard rules from CLAUDE.md respected:

* All times stored UTC.
* ``data/store.py`` is the only DuckDB path (we go through
  ``ts_momentum.load_ohlcv_panel`` and ``store.load_fred_panel``).
* ``backtest/`` is not imported here (the BT-style baseline lives there
  and is consumed only by the verdict's KS test, not by this runner).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from casino._money import round_money as _round_money
from casino.config import get_config
from casino.data import store
from casino.execution import paper_clock
from casino.execution.sim_broker import SimBroker
from casino.execution.tsmom_runner import (
    DEFAULT_STOP_FRACTION,
    HISTORY_DAYS,
    TargetWeight,
)
from casino.signals.ts_momentum import TSMOM_UNIVERSE, load_ohlcv_panel
from casino.signals.ts_momentum_regime import (
    DEFAULT_BOND_LEGS,
    DEFAULT_LONG_RATE,
    DEFAULT_SHORT_RATE,
    DEFAULT_SLOPE_THRESHOLD,
    compute_tsmom_regime_panel,
)

if TYPE_CHECKING:
    import pandas as pd

# Run id for the shadow. Distinct from the live runner's ``DiCaprio`` so
# both can write to ``paper_clock`` without colliding.
SHADOW_RUN_ID: str = "Belfort"


# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class ShadowRebalAction:
    """One concrete action the shadow rebal step decided to take."""

    kind: str  # "open_long" | "close" | "skip"
    symbol: str
    target_weight: float
    target_dollars: Decimal
    qty: int
    reference_price: Decimal
    stop_price: Decimal
    reason: str


@dataclass
class ShadowRebalRunResult:
    """Structured output of one ``run_shadow_rebal`` invocation."""

    rebal_date_utc: date
    is_rebal_day: bool
    forced: bool
    nav: Decimal
    actions: list[ShadowRebalAction] = field(default_factory=list)
    submitted_order_ids: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    catchup_dates: list[date] = field(default_factory=list)


# ---------------------------------------------------------------------------- helpers


def _today_utc() -> date:
    return datetime.now(tz=UTC).date()


def _load_recent_prices(
    *,
    universe: Sequence[str] = TSMOM_UNIVERSE,
    history_days: int = HISTORY_DAYS,
    duckdb_path: Path | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Pull the most-recent ``history_days`` of adj_close from DuckDB."""
    end_ts = end if end is not None else datetime.now(tz=UTC)
    start_ts = end_ts - timedelta(days=history_days + 60)
    return load_ohlcv_panel(
        start=start_ts,
        end=end_ts,
        universe=list(universe),
        db_path=duckdb_path,
    )


def _load_recent_yields(
    *,
    duckdb_path: Path | None = None,
    end: datetime | None = None,
    series_ids: Sequence[str] = (DEFAULT_LONG_RATE, DEFAULT_SHORT_RATE),
) -> pd.DataFrame:
    """Pull recent FRED yield panel for the slope filter."""
    end_ts = end if end is not None else datetime.now(tz=UTC)
    # Slope only needs ~5 prior observations (forward-filled); pull a year for safety.
    start_ts = end_ts - timedelta(days=400)
    return store.load_fred_panel(
        series_ids=list(series_ids),
        start=start_ts,
        end=end_ts,
        db_path=duckdb_path,
    )


def latest_regime_target_weights(
    prices: pd.DataFrame,
    fred_yields: pd.DataFrame,
    *,
    target_vol: float = 0.10,
    gross_target: float = 1.0,
    bond_legs: tuple[str, ...] = DEFAULT_BOND_LEGS,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
) -> list[TargetWeight]:
    """Compute the regime-filtered TSMOM panel and extract the most-recent row."""
    panel = compute_tsmom_regime_panel(
        prices,
        fred_yields,
        target_vol=target_vol,
        mode="long_only",
        gross_target=gross_target,
        bond_legs=bond_legs,
        slope_threshold=slope_threshold,
    )
    if panel.empty:
        return []
    latest = panel.iloc[-1]
    last_prices = prices.iloc[-1]
    out: list[TargetWeight] = []
    for sym in panel.columns:
        w = float(latest[sym]) if latest[sym] == latest[sym] else 0.0
        if w <= 0.0:
            continue
        ref = last_prices[sym]
        if ref != ref or float(ref) <= 0:
            continue
        out.append(
            TargetWeight(
                symbol=sym,
                weight=w,
                reference_price=Decimal(str(float(ref))),
            )
        )
    return out


def plan_shadow_actions(
    *,
    target_weights: Sequence[TargetWeight],
    nav: Decimal,
    current_positions: dict[str, int],
    stop_fraction: Decimal = DEFAULT_STOP_FRACTION,
) -> list[ShadowRebalAction]:
    """Translate target weights into concrete sim broker actions.

    Mirrors the live runner's ``plan_rebal_actions`` but operates on the
    sim broker's view of held positions (a simple ``dict[symbol, qty]``)
    so we don't need a ``PortfolioState`` shaped exactly like the live
    one.
    """
    cfg = get_config()
    single_name_cap_dollars = nav * Decimal(str(cfg.max_single_name))
    gross_cap_dollars = nav * Decimal(str(cfg.max_gross_exposure))
    target_syms = {tw.symbol.upper() for tw in target_weights}

    actions: list[ShadowRebalAction] = []
    # 1) Close anything held that's not in the new target.
    for sym, qty in current_positions.items():
        if sym.upper() not in target_syms:
            actions.append(
                ShadowRebalAction(
                    kind="close",
                    symbol=sym.upper(),
                    target_weight=0.0,
                    target_dollars=Decimal("0"),
                    qty=int(qty),
                    reference_price=Decimal("0"),
                    stop_price=Decimal("0"),
                    reason=f"{sym} no longer in regime-filtered TSMOM target",
                )
            )

    # 2) Per-symbol target dollars, capped to single-name 10%.
    raw_target: dict[str, Decimal] = {}
    for tw in target_weights:
        sym = tw.symbol.upper()
        td = nav * Decimal(str(tw.weight))
        if td > single_name_cap_dollars:
            td = single_name_cap_dollars
        raw_target[sym] = td

    # 3) Gross-exposure pass.
    total = sum(raw_target.values(), start=Decimal("0"))
    scale = Decimal("1")
    if total > gross_cap_dollars and total > Decimal("0"):
        scale = gross_cap_dollars / total

    # 4) Build open-long actions.
    for tw in target_weights:
        sym = tw.symbol.upper()
        target_dollars = _round_money(raw_target[sym] * scale)
        if target_dollars <= Decimal("0"):
            continue
        ref = tw.reference_price
        qty = int((target_dollars / ref).to_integral_value(rounding=ROUND_DOWN))
        if qty <= 0:
            actions.append(
                ShadowRebalAction(
                    kind="skip",
                    symbol=sym,
                    target_weight=tw.weight,
                    target_dollars=target_dollars,
                    qty=0,
                    reference_price=ref,
                    stop_price=Decimal("0"),
                    reason=f"{sym}: target_dollars={target_dollars} < 1 share at ${ref}",
                )
            )
            continue
        stop_price = _round_money(ref * (Decimal("1") - stop_fraction))
        actions.append(
            ShadowRebalAction(
                kind="open_long",
                symbol=sym,
                target_weight=tw.weight,
                target_dollars=target_dollars,
                qty=qty,
                reference_price=ref,
                stop_price=stop_price,
                reason=f"{sym}: regime-filtered TSMOM weight={tw.weight:.4f}",
            )
        )
    return actions


# ---------------------------------------------------------------------------- runner


def run_shadow_rebal(
    *,
    broker: SimBroker | None = None,
    universe: Sequence[str] = TSMOM_UNIVERSE,
    today: date | None = None,
    force: bool = False,
    db_path: Path | None = None,
    duckdb_path: Path | None = None,
    target_vol: float = 0.10,
    gross_target: float = 1.0,
    stop_fraction: Decimal = DEFAULT_STOP_FRACTION,
    bond_legs: tuple[str, ...] = DEFAULT_BOND_LEGS,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
    run_id: str = SHADOW_RUN_ID,
    catchup_from: date | None = None,
) -> ShadowRebalRunResult:
    """One simulated rebal step for the shadow strategy.

    Steps:

    1. Establish the sim broker (creates the sim_account row on first use).
    2. If ``catchup_from`` is set, mark-to-market every business day from
       that date up to ``today - 1`` BEFORE the rebal logic runs. This
       lets the operator backfill missed days.
    3. Decide if today is a rebal day. If not and not ``force``, just
       mark-to-market and return.
    4. Pull recent OHLCV + FRED panels through ``casino.data.store``.
    5. Apply the regime-filtered signal; build action list.
    6. Submit sim bracket orders (close first, then opens).
    7. Mark-to-market today so fills are executed and NAV is recorded.
    8. Persist a ``rebal_event`` row in ``paper_clock`` (run_id-scoped).
    """
    today = today or _today_utc()
    actual_broker = (
        broker
        if broker is not None
        else SimBroker(run_id=run_id, db_path=db_path, duckdb_path=duckdb_path)
    )

    # ---------------- catchup
    catchup_dates: list[date] = []
    if catchup_from is not None and catchup_from < today:
        cur = catchup_from
        while cur < today:
            # Only mark on business days (Mon-Fri). Sim is tolerant of
            # missing bars (weekends just return empty), but we keep the
            # nav-history clean by skipping weekends.
            if cur.weekday() < 5:
                actual_broker.mark_to_market(cur)
                catchup_dates.append(cur)
            cur = cur + timedelta(days=1)

    # ---------------- rebal-day gate
    is_rebal_day = paper_clock.is_last_business_day_of_month(today)
    nav_now = actual_broker.get_account().equity

    if not is_rebal_day and not force:
        # Daily MTM only — record a NAV row so the dashboard sees life.
        actual_broker.mark_to_market(today)
        return ShadowRebalRunResult(
            rebal_date_utc=today,
            is_rebal_day=False,
            forced=False,
            nav=nav_now,
            skipped_reason=(
                f"{today.isoformat()} is not the last business day of the month; "
                "use --force to override (sim NAV updated)"
            ),
            catchup_dates=catchup_dates,
        )

    # ---------------- ensure paper_clock started for our run_id
    paper_clock.ensure_started(
        run_id=run_id,
        strategy="tsmom_regime_shadow",
        start_nav=nav_now,
        cap_days=paper_clock.PAPER_CAP_DAYS,
        config_json=json.dumps(
            {
                "universe": list(universe),
                "target_vol": target_vol,
                "gross_target": gross_target,
                "stop_fraction": str(stop_fraction),
                "mode": "long_only",
                "bond_legs": list(bond_legs),
                "slope_threshold": slope_threshold,
                "variant": "regime_filtered_shadow",
            },
            sort_keys=True,
        ),
        db_path=db_path,
    )

    # ---------------- compute signal
    end_ts = datetime.combine(today, datetime.min.time(), tzinfo=UTC) + timedelta(days=1)
    prices = _load_recent_prices(universe=universe, duckdb_path=duckdb_path, end=end_ts)
    if prices.empty:
        return ShadowRebalRunResult(
            rebal_date_utc=today,
            is_rebal_day=is_rebal_day,
            forced=force,
            nav=nav_now,
            skipped_reason="no OHLCV history; cannot compute regime-filtered TSMOM",
            catchup_dates=catchup_dates,
        )

    # Freshness gate (mirrors the live runner). Critical for the comparability
    # of the two 30-day samples — a stale rebal day in the shadow but fresh
    # in the live bot would invalidate the head-to-head.
    latest_ts = prices.index.max()
    if hasattr(latest_ts, "tz_convert") and latest_ts.tz is not None:
        latest_ts = latest_ts.tz_convert("UTC")
    latest_date = latest_ts.date() if hasattr(latest_ts, "date") else latest_ts
    if latest_date < today and not force:
        cmd = (
            "uv run python -m casino.data.ingest_yfinance "
            "--tickers-file universe_tsmom.txt --mode ohlcv "
            f"--ohlcv-start {(today - timedelta(days=30)).isoformat()} "
            "--rate-limit-sec 0"
        )
        return ShadowRebalRunResult(
            rebal_date_utc=today,
            is_rebal_day=is_rebal_day,
            forced=force,
            nav=nav_now,
            skipped_reason=(
                f"OHLCV stale: latest bar {latest_date.isoformat()} < today "
                f"{today.isoformat()}. Run yfinance ingest first:\n  {cmd}"
            ),
            catchup_dates=catchup_dates,
        )

    fred = _load_recent_yields(duckdb_path=duckdb_path, end=end_ts)

    targets = latest_regime_target_weights(
        prices,
        fred,
        target_vol=target_vol,
        gross_target=gross_target,
        bond_legs=bond_legs,
        slope_threshold=slope_threshold,
    )

    # Current sim positions as a dict[symbol -> qty].
    current_positions = {p.symbol: p.qty for p in actual_broker.get_positions()}

    actions = plan_shadow_actions(
        target_weights=targets,
        nav=nav_now,
        current_positions=current_positions,
        stop_fraction=stop_fraction,
    )

    result = ShadowRebalRunResult(
        rebal_date_utc=today,
        is_rebal_day=is_rebal_day,
        forced=force,
        nav=nav_now,
        actions=actions,
        catchup_dates=catchup_dates,
    )

    # ---------------- submit sim orders
    for a in actions:
        if a.kind == "close":
            try:
                ord_resp = actual_broker.close_position(a.symbol)
                result.submitted_order_ids.append(ord_resp.id)
                logger.info(
                    "tsmom_shadow_runner: queued close {} qty={}",
                    a.symbol,
                    a.qty,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("tsmom_shadow_runner: close {} failed: {}", a.symbol, e)

    for a in actions:
        if a.kind != "open_long":
            continue
        try:
            ord_resp = actual_broker.submit_bracket_order(
                symbol=a.symbol,
                qty=a.qty,
                side="buy",
                stop_price=a.stop_price,
            )
            result.submitted_order_ids.append(ord_resp.id)
            logger.info(
                "tsmom_shadow_runner: bracket-bought {} qty={} ref=${} stop=${}",
                a.symbol,
                a.qty,
                a.reference_price,
                a.stop_price,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("tsmom_shadow_runner: submit {} failed: {}", a.symbol, e)

    # ---------------- mark-to-market today (fills the just-submitted orders)
    actual_broker.mark_to_market(today)

    # ---------------- record rebal_event keyed by our run_id
    nav_after = actual_broker.get_account().equity
    weights_blob = json.dumps(
        [{"symbol": tw.symbol, "weight": tw.weight} for tw in targets],
        sort_keys=True,
    )
    paper_clock.insert_rebal_event(
        n_orders_submitted=len(result.submitted_order_ids),
        nav_at_rebal=nav_after,
        target_weights_json=weights_blob,
        notes=f"shadow rebal_date_utc={today.isoformat()}",
        run_id=run_id,
        db_path=db_path,
    )
    return result


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.execution.tsmom_shadow_runner",
        description="Regime-filtered TSMOM shadow runner (Option B parallel sim).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run rebal even if today is not the last business day of month.",
    )
    parser.add_argument(
        "--catchup-from",
        type=str,
        default=None,
        help="Backfill mark-to-market from this YYYY-MM-DD up to today.",
    )
    args = parser.parse_args(argv)

    catchup = date.fromisoformat(args.catchup_from) if args.catchup_from else None

    try:
        result = run_shadow_rebal(force=args.force, catchup_from=catchup)
    except Exception as e:  # noqa: BLE001
        logger.exception("tsmom_shadow_runner: unhandled exception: {}", e)
        return 1

    if result.skipped_reason:
        logger.info("tsmom_shadow_runner: {}", result.skipped_reason)
        return 0
    logger.warning(
        "tsmom_shadow_runner: done; rebal_day={} forced={} actions={} submitted={} catchup_days={}",
        result.is_rebal_day,
        result.forced,
        len(result.actions),
        len(result.submitted_order_ids),
        len(result.catchup_dates),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
