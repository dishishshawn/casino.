"""TSMOM **Falcon** runner — aggressive sibling of the live ``DiCaprio`` bot.

Falcon is a deliberately *more aggressive* variant of the vanilla TSMOM
strategy that ``casino.execution.tsmom_runner`` runs live (``run_id="DiCaprio"``).
It exists to answer one question: does cranking the legal aggression knobs
to their limit beat the conservative live configuration over the same
30-day window?

Like ``casino.execution.tsmom_shadow_runner`` (``run_id="Belfort"``), Falcon
runs **entirely in the in-process simulator** (``SimBroker``). This is a
hard safety boundary, not a convenience:

* The live ``DiCaprio`` bot and Falcon would otherwise share a *single*
  Alpaca paper account, a single internal ``book``, and a single
  reconcile/kill-switch path. Two live runners on one account collide on
  positions, gross-exposure accounting, and — critically — the kill
  switch flattens the *entire* account (``risk.flatten_and_disable`` calls
  ``broker.close_all_positions``). Falcon-in-sim cannot touch DiCaprio's
  live positions, NAV, or clock.
* All Falcon state is keyed by ``run_id="Falcon"`` in the ``sim_*`` tables
  and a dedicated ``paper_clock`` row. Nothing Falcon writes is visible to
  the live tables.

How Falcon is "more aggressive" than DiCaprio — and what it is NOT allowed
to do:

* ``target_vol = 0.20`` — double DiCaprio's 0.10 per-asset vol target. In a
  long-only, gross-capped book this keeps Falcon fully invested far more
  often (the gross-normalization scale saturates at 1.0).
* ``lookbacks = (21, 63, 126)`` — drops the 252-bday (12-month) leg the
  live bot uses. Falcon reacts faster to trend changes (higher turnover,
  more "aggressive" momentum-chasing) at the cost of more whipsaw.
* ``stop_fraction = 0.15`` — wider than DiCaprio's 0.10 so the higher-vol
  book is not stopped out on ordinary noise. Wider stop = more downside
  per position = a more aggressive risk posture.

* Falcon does **NOT** relax the CLAUDE.md hard-rule caps. Gross exposure
  stays ≤ 100% NAV and single-name ≤ 10% NAV — enforced by the shared
  ``plan_shadow_actions`` (which reads ``cfg.max_gross_exposure`` /
  ``cfg.max_single_name``). "Aggressive" means pushing the *signal* knobs,
  never the capital-at-risk ceilings.

Unlike Belfort, Falcon uses the **vanilla** TSMOM signal
(``compute_tsmom_panel``), not the regime-filtered variant — the regime
filter zeroes bond legs when the yield curve inverts, which is a
*de-risking* feature and therefore the opposite of what Falcon is for.

CLI (mirrors the shadow runner):

    uv run python -m casino.execution.tsmom_falcon_runner
        # Daily-or-rebal step. On a month-end, computes weights and submits
        # sim orders. Off rebal day, just marks-to-market.

    uv run python -m casino.execution.tsmom_falcon_runner --force
        # Force an off-rebal-day rebal (used to start Falcon mid-month).

    uv run python -m casino.execution.tsmom_falcon_runner --catchup-from 2026-06-01
        # Backfill mark-to-market from a prior date forward through DuckDB OHLCV.

Day-30 verdict: the read-only verdict machinery works for Falcon via

    uv run python -m casino.execution.tsmom_clock_check --verdict --run-id Falcon

The **daily** ``tsmom_clock_check`` kill path is live-account-oriented
(it flattens Alpaca); do NOT schedule it against ``--run-id Falcon``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from casino.execution import paper_clock
from casino.execution.sim_broker import SimBroker
from casino.execution.tsmom_runner import HISTORY_DAYS, TargetWeight
from casino.execution.tsmom_shadow_runner import (
    ShadowRebalAction as FalconRebalAction,  # generic action container; aliased for clarity
)
from casino.execution.tsmom_shadow_runner import (
    plan_shadow_actions as plan_falcon_actions,  # signal-agnostic cap logic; reused verbatim
)
from casino.signals.ts_momentum import (
    TSMOM_UNIVERSE,
    compute_tsmom_panel,
    load_ohlcv_panel,
)

if TYPE_CHECKING:
    import pandas as pd

# Run id for Falcon. Distinct from the live ``DiCaprio`` and the
# regime-filtered shadow ``Belfort`` so all three write to ``paper_clock``
# and the ``sim_*`` tables without colliding.
FALCON_RUN_ID: str = "Falcon"

# ----- Aggression knobs (see module docstring for the rationale) -----
# These are the only differences from the live DiCaprio configuration that
# make Falcon "more aggressive". None of them touch the hard-rule capital
# caps (gross ≤ 100%, single-name ≤ 10%), which stay enforced downstream.
FALCON_TARGET_VOL: float = 0.20  # 2x DiCaprio's 0.10
FALCON_LOOKBACKS: tuple[int, ...] = (21, 63, 126)  # drop the slow 252-bday leg
FALCON_GROSS_TARGET: float = 1.0  # gross still pinned at the 100%-NAV hard cap
FALCON_STOP_FRACTION: Decimal = Decimal("0.15")  # wider than DiCaprio's 0.10

# Re-export so callers can write ``from ...tsmom_falcon_runner import FalconRebalAction``.
__all__ = [
    "FALCON_GROSS_TARGET",
    "FALCON_LOOKBACKS",
    "FALCON_RUN_ID",
    "FALCON_STOP_FRACTION",
    "FALCON_TARGET_VOL",
    "FalconRebalAction",
    "FalconRebalRunResult",
    "latest_falcon_target_weights",
    "plan_falcon_actions",
    "run_falcon_rebal",
]


# ---------------------------------------------------------------------------- types


@dataclass
class FalconRebalRunResult:
    """Structured output of one ``run_falcon_rebal`` invocation."""

    rebal_date_utc: date
    is_rebal_day: bool
    forced: bool
    nav: Decimal
    actions: list[FalconRebalAction] = field(default_factory=list)
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
    """Pull the most-recent ``history_days`` of adj_close from DuckDB.

    Mirrors the live/shadow runners exactly so the three samples are
    computed from an identical data window.
    """
    end_ts = end if end is not None else datetime.now(tz=UTC)
    start_ts = end_ts - timedelta(days=history_days + 60)
    return load_ohlcv_panel(
        start=start_ts,
        end=end_ts,
        universe=list(universe),
        db_path=duckdb_path,
    )


def latest_falcon_target_weights(
    prices: pd.DataFrame,
    *,
    target_vol: float = FALCON_TARGET_VOL,
    gross_target: float = FALCON_GROSS_TARGET,
    lookbacks: tuple[int, ...] = FALCON_LOOKBACKS,
) -> list[TargetWeight]:
    """Compute the aggressive vanilla-TSMOM panel and extract the latest row.

    Same extraction contract as ``tsmom_runner.latest_target_weights`` —
    ``mode="long_only"`` (Alpaca paper-cash can't short, and the live bot
    validates the long-only variant), NaN/non-positive weights and prices
    dropped — but with Falcon's faster ``lookbacks`` and higher
    ``target_vol``.
    """
    panel = compute_tsmom_panel(
        prices,
        lookbacks=lookbacks,
        target_vol=target_vol,
        mode="long_only",
        gross_target=gross_target,
    )
    if panel.empty:
        return []
    latest = panel.iloc[-1]
    last_prices = prices.iloc[-1]
    out: list[TargetWeight] = []
    for sym in panel.columns:
        w = float(latest[sym]) if latest[sym] == latest[sym] else 0.0  # NaN guard
        if w <= 0.0:
            continue
        ref = last_prices[sym]
        if ref != ref or float(ref) <= 0:  # NaN / non-positive
            continue
        out.append(
            TargetWeight(
                symbol=sym,
                weight=w,
                reference_price=Decimal(str(float(ref))),
            )
        )
    return out


# ---------------------------------------------------------------------------- runner


def run_falcon_rebal(
    *,
    broker: SimBroker | None = None,
    universe: Sequence[str] = TSMOM_UNIVERSE,
    today: date | None = None,
    force: bool = False,
    db_path: Path | None = None,
    duckdb_path: Path | None = None,
    target_vol: float = FALCON_TARGET_VOL,
    gross_target: float = FALCON_GROSS_TARGET,
    lookbacks: tuple[int, ...] = FALCON_LOOKBACKS,
    stop_fraction: Decimal = FALCON_STOP_FRACTION,
    run_id: str = FALCON_RUN_ID,
    catchup_from: date | None = None,
) -> FalconRebalRunResult:
    """One simulated rebal step for the aggressive Falcon strategy.

    Structurally identical to ``tsmom_shadow_runner.run_shadow_rebal`` (same
    catchup / freshness / mark-to-market discipline) but swaps the
    regime-filtered signal for the aggressive vanilla one. Steps:

    1. Establish the sim broker (creates the ``sim_account`` row on first use).
    2. If ``catchup_from`` is set, mark-to-market every business day from
       that date up to ``today - 1`` BEFORE the rebal logic runs.
    3. Rebal-day gate. Off rebal day without ``force`` → mark-to-market only.
    4. Pull recent OHLCV through ``casino.data.store``; freshness gate.
    5. Compute the aggressive vanilla-TSMOM weights; build the action list
       (capped to single-name 10% / gross 100% by ``plan_falcon_actions``).
    6. Submit sim bracket orders (close first, then opens).
    7. Mark-to-market today so fills execute and NAV is recorded.
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
            if cur.weekday() < 5:  # business days only; sim tolerates gaps
                actual_broker.mark_to_market(cur)
                catchup_dates.append(cur)
            cur = cur + timedelta(days=1)

    # ---------------- rebal-day gate
    is_rebal_day = paper_clock.is_last_business_day_of_month(today)
    nav_now = actual_broker.get_account().equity

    if not is_rebal_day and not force:
        actual_broker.mark_to_market(today)
        return FalconRebalRunResult(
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
        strategy="tsmom_falcon_aggressive",
        start_nav=nav_now,
        cap_days=paper_clock.PAPER_CAP_DAYS,
        config_json=json.dumps(
            {
                "universe": list(universe),
                "target_vol": target_vol,
                "gross_target": gross_target,
                "lookbacks": list(lookbacks),
                "stop_fraction": str(stop_fraction),
                "mode": "long_only",
                "variant": "aggressive_vanilla_tsmom",
            },
            sort_keys=True,
        ),
        db_path=db_path,
    )

    # ---------------- compute signal
    end_ts = datetime.combine(today, datetime.min.time(), tzinfo=UTC) + timedelta(days=1)
    prices = _load_recent_prices(universe=universe, duckdb_path=duckdb_path, end=end_ts)
    if prices.empty:
        return FalconRebalRunResult(
            rebal_date_utc=today,
            is_rebal_day=is_rebal_day,
            forced=force,
            nav=nav_now,
            skipped_reason="no OHLCV history; cannot compute Falcon TSMOM weights",
            catchup_dates=catchup_dates,
        )

    # Freshness gate (mirrors the live + shadow runners). A stale rebal day
    # in Falcon but fresh in the live bot would invalidate the head-to-head.
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
        return FalconRebalRunResult(
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

    targets = latest_falcon_target_weights(
        prices,
        target_vol=target_vol,
        gross_target=gross_target,
        lookbacks=lookbacks,
    )

    # Current sim positions as a dict[symbol -> qty].
    current_positions = {p.symbol: p.qty for p in actual_broker.get_positions()}

    actions = plan_falcon_actions(
        target_weights=targets,
        nav=nav_now,
        current_positions=current_positions,
        stop_fraction=stop_fraction,
    )

    result = FalconRebalRunResult(
        rebal_date_utc=today,
        is_rebal_day=is_rebal_day,
        forced=force,
        nav=nav_now,
        actions=actions,
        catchup_dates=catchup_dates,
    )

    # ---------------- submit sim orders (close first, then opens)
    for a in actions:
        if a.kind == "close":
            try:
                ord_resp = actual_broker.close_position(a.symbol)
                result.submitted_order_ids.append(ord_resp.id)
                logger.info("tsmom_falcon_runner: queued close {} qty={}", a.symbol, a.qty)
            except Exception as e:  # noqa: BLE001
                logger.error("tsmom_falcon_runner: close {} failed: {}", a.symbol, e)

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
                "tsmom_falcon_runner: bracket-bought {} qty={} ref=${} stop=${}",
                a.symbol,
                a.qty,
                a.reference_price,
                a.stop_price,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("tsmom_falcon_runner: submit {} failed: {}", a.symbol, e)

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
        notes=f"falcon rebal_date_utc={today.isoformat()}",
        run_id=run_id,
        db_path=db_path,
    )
    return result


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.execution.tsmom_falcon_runner",
        description="Aggressive vanilla-TSMOM Falcon runner (parallel sim, run_id=Falcon).",
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
        result = run_falcon_rebal(force=args.force, catchup_from=catchup)
    except Exception as e:  # noqa: BLE001
        logger.exception("tsmom_falcon_runner: unhandled exception: {}", e)
        return 1

    if result.skipped_reason:
        logger.info("tsmom_falcon_runner: {}", result.skipped_reason)
        return 0
    logger.warning(
        "tsmom_falcon_runner: done; rebal_day={} forced={} actions={} submitted={} catchup_days={}",
        result.is_rebal_day,
        result.forced,
        len(result.actions),
        len(result.submitted_order_ids),
        len(result.catchup_dates),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
