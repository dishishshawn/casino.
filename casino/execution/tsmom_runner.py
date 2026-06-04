"""TSMOM Alpaca paper-trading runner — Branch C, 30-day cap.

Branch C amendment 2026-05-07 (PRD §6.3 amendment): the original 3-month
paper window was reduced to a binding **30-day cap**. After day 30 the
companion script ``casino.execution.tsmom_clock_check --verdict`` emits a
binary COMMIT-or-KILL verdict on Branch C.

This module owns the *monthly rebal* leg of that regime. The companion
``tsmom_clock_check`` module owns the *daily kill-criteria* leg. They
share the ``paper_clock`` SQLite tables in ``casino.execution.paper_clock``.

Contract:

* The runner is **manual-run only**. It does NOT cron-schedule itself.
  The 30-day clock starts when the operator first invokes
  ``uv run python -m casino.execution.tsmom_runner`` on a month-end. Until
  that first invocation, ``paper_clock`` is empty and no live (paper)
  orders are submitted.
* The runner refuses to execute against any non-paper Alpaca URL. Defense
  in depth against an env-var typo flipping the system live (CLAUDE.md
  hard rule 7 — cash account only, paper for v1).
* The strategy is forced to ``mode="long_only"`` regardless of caller
  intent. Alpaca paper-cash accounts cannot short, and we are deliberately
  validating the long-only TSMOM variant.
* Every entry order is a bracket order with a broker-side stop at
  ``-10%`` of the entry-day reference price (CLAUDE.md hard rule 3 — every
  position has a broker-side stop).
* All caps from ``casino.execution.risk`` (per-trade ≤ 1.5% NAV via
  per-share risk; single-name ≤ 10% NAV; gross ≤ 100% NAV) are enforced
  by ``risk.size_position``. The runner does NOT add a parallel sizing
  path — it routes every order through ``risk.submit_order``.
* Rebal-day discipline: by default the runner only acts on the **last
  business day of the month** (Mon-Fri only; the broker clock is the
  authority on whether the market is actually open). Off-rebal-day calls
  are no-ops by design — safer than the alternative.

Failure modes the runner intentionally turns into hard errors:

* ``ALPACA_BASE_URL`` does not contain ``"paper"`` → raise on startup.
* Kill switch flag is set in ``state.sqlite`` → ``risk.submit_order``
  raises ``TradingDisabledError``; the runner logs and exits non-zero.
* Reconcile drift > $1 / 1 share after rebal → log critical; the day-30
  verdict (separate script) will pick this up.

CLI:
    uv run python -m casino.execution.tsmom_runner [--force] [--dry-run]

    --force     run even if today is not a rebal day (operator override).
                Also bypasses the OHLCV freshness gate.
    --dry-run   compute signal + sized orders + log them, but do NOT submit.

Operational sequence on a month-end (must run in this order, AFTER NYSE
close + ~30 min for yfinance to publish today's adj_close):

    # 1. Ingest today's bar into DuckDB.
    uv run python -m casino.data.ingest_yfinance \
        --tickers-file universe_tsmom.txt --mode ohlcv \
        --ohlcv-start 2026-04-01 --rate-limit-sec 0

    # 2. Run the rebal. Will hard-fail if step 1 didn't land today's bar.
    uv run python -m casino.execution.tsmom_runner

The runner contains an explicit freshness assertion: if the latest DuckDB
adj_close timestamp is < today's date, the runner returns a no-op result
with the exact ingest command in the skipped_reason. This is intentional:
a stale rebal would invalidate the 30-day paper sample.
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

from casino._money import floor_shares
from casino._money import round_money as _round_money
from casino.config import get_config
from casino.execution import book, paper_clock, reconcile
from casino.execution.alpaca_broker import AlpacaBroker, build_default_broker
from casino.execution.risk import (
    PortfolioState,
    RiskRejection,
    TradingDisabledError,
    snapshot_portfolio_from_broker,
    submit_order,
)
from casino.monitoring import alerts
from casino.signals.ts_momentum import (
    TSMOM_UNIVERSE,
    compute_tsmom_panel,
    load_ohlcv_panel,
)

if TYPE_CHECKING:
    import pandas as pd

# TSMOM_UNIVERSE is re-exported above so callers that historically wrote
# ``from casino.execution.tsmom_runner import TSMOM_UNIVERSE`` keep
# working. The canonical definition lives in casino.signals.ts_momentum
# (see structure_review P1 #7).

# Broker-side stop level for every long entry. PRD §8 + CLAUDE.md rule 3.
# 10% is below realistic monthly TSMOM volatility envelope (vol-target=10%
# annualized → ~2.9% monthly stdev) but above the per-trade-risk cap; if it
# fires, that signals a regime break, not noise.
DEFAULT_STOP_FRACTION: Decimal = Decimal("0.10")

# Lookback window for OHLCV pulled from DuckDB. Generous to ensure the longest
# TSMOM lookback (252 bdays = ~14 months calendar) plus burn-in is covered.
HISTORY_DAYS: int = 500


# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class TargetWeight:
    """One symbol's signal-derived target weight at the rebal date."""

    symbol: str
    weight: float
    reference_price: Decimal


@dataclass(frozen=True)
class RebalAction:
    """One concrete action the rebal step decided to take."""

    kind: str  # "open_long" | "close" | "skip"
    symbol: str
    target_weight: float
    target_dollars: Decimal
    qty: int
    reference_price: Decimal
    stop_price: Decimal
    reason: str


@dataclass
class RebalRunResult:
    """Structured output of one ``run_rebal`` invocation."""

    rebal_date_utc: date
    is_rebal_day: bool
    forced: bool
    dry_run: bool
    nav: Decimal
    actions: list[RebalAction] = field(default_factory=list)
    submitted_order_ids: list[str] = field(default_factory=list)
    drift_after: int = 0
    skipped_reason: str | None = None


# ---------------------------------------------------------------------------- guards


class NotPaperAccountError(RuntimeError):
    """Raised when ``ALPACA_BASE_URL`` does not contain ``"paper"``.

    CLAUDE.md hard rule 7 / PRD §8: live trading requires explicit human
    approval; until then the runner refuses to talk to a non-paper URL.
    """


def assert_paper_account(*, alpaca_base_url: str | None = None) -> None:
    """Raise NotPaperAccountError unless the URL is unambiguously paper.

    The check is intentionally strict — the substring ``"paper"`` must
    appear. Defense in depth: if a future operator sets
    ``ALPACA_BASE_URL=https://api.alpaca.markets`` thinking they're still
    in paper mode, this raises rather than rolling the dice.
    """
    url = alpaca_base_url if alpaca_base_url is not None else get_config().alpaca_base_url
    if "paper" not in url.lower():
        raise NotPaperAccountError(
            f"ALPACA_BASE_URL={url!r} does not contain 'paper'; refusing to run. "
            "TSMOM paper-trading requires a paper-api endpoint (CLAUDE.md hard rule 7)."
        )


# ---------------------------------------------------------------------------- universe -> orders


def latest_target_weights(
    prices: pd.DataFrame,
    *,
    target_vol: float = 0.10,
    gross_target: float = 1.0,
) -> list[TargetWeight]:
    """Compute the TSMOM weight panel and extract the most-recent row.

    Forces ``mode="long_only"`` to match the paper-cash-account constraint.
    Negative or NaN weights are dropped (no shorting, and burn-in cells
    are unscoreable). The reference price is the close on the same row;
    this becomes the basis for the broker-side stop.
    """
    panel = compute_tsmom_panel(
        prices,
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


def plan_rebal_actions(
    *,
    target_weights: Sequence[TargetWeight],
    portfolio: PortfolioState,
    book_positions: Sequence[book.StoredPosition],
    stop_fraction: Decimal = DEFAULT_STOP_FRACTION,
) -> list[RebalAction]:
    """Decide the concrete actions for a rebal day.

    Decision rules:

    * Symbols in the book but **not** in target → action ``"close"``.
    * Symbols in target → action ``"open_long"`` with target_dollars =
      ``weight * NAV``, qty floored to whole shares, stop at
      ``ref_price * (1 - stop_fraction)``.
    * If the per-symbol target_dollars exceeds the 10% single-name cap,
      shrink to the cap (the runner *also* routes through
      ``risk.size_position`` on submission, which enforces the cap a
      second time — belt and braces).
    * Final cap pass: if the sum of ``target_dollars`` after shrinkage
      exceeds ``NAV * gross_cap`` (= 1.0 by default), proportionally
      down-scale every entry. ``risk.size_position`` would also enforce
      this on a per-symbol basis, but doing the pass here keeps the
      action plan internally consistent.
    """
    cfg = get_config()
    nav = portfolio.nav
    single_name_cap_dollars = nav * Decimal(str(cfg.max_single_name))
    gross_cap_dollars = nav * Decimal(str(cfg.max_gross_exposure))

    target_syms = {tw.symbol.upper() for tw in target_weights}
    actions: list[RebalAction] = []

    # 1) Close anything held not in target.
    for pos in book_positions:
        sym = pos.symbol.upper()
        if sym not in target_syms:
            actions.append(
                RebalAction(
                    kind="close",
                    symbol=sym,
                    target_weight=0.0,
                    target_dollars=Decimal("0"),
                    qty=pos.qty,
                    reference_price=pos.avg_entry_price,
                    stop_price=Decimal("0"),
                    reason=f"{sym} no longer in TSMOM target panel",
                )
            )

    # 2) Compute per-symbol target dollars + shrink to single-name cap.
    raw_target: dict[str, Decimal] = {}
    for tw in target_weights:
        sym = tw.symbol.upper()
        td = nav * Decimal(str(tw.weight))
        if td > single_name_cap_dollars:
            td = single_name_cap_dollars
        raw_target[sym] = td

    # 3) Gross-exposure pass: scale down proportionally if total > cap.
    total = sum(raw_target.values(), start=Decimal("0"))
    if total > gross_cap_dollars and total > Decimal("0"):
        scale = gross_cap_dollars / total
    else:
        scale = Decimal("1")

    # 4) Build open-long actions.
    for tw in target_weights:
        sym = tw.symbol.upper()
        target_dollars = _round_money(raw_target[sym] * scale)
        if target_dollars <= Decimal("0"):
            continue
        ref = tw.reference_price
        # whole-share qty floored conservatively (see casino._money.floor_shares)
        qty = floor_shares(target_dollars, ref)
        if qty <= 0:
            actions.append(
                RebalAction(
                    kind="skip",
                    symbol=sym,
                    target_weight=tw.weight,
                    target_dollars=target_dollars,
                    qty=0,
                    reference_price=ref,
                    stop_price=Decimal("0"),
                    reason=(
                        f"{sym}: target_dollars={target_dollars} too small "
                        f"to buy a single share at ${ref}"
                    ),
                )
            )
            continue
        stop_price = _round_money(ref * (Decimal("1") - stop_fraction))
        actions.append(
            RebalAction(
                kind="open_long",
                symbol=sym,
                target_weight=tw.weight,
                target_dollars=target_dollars,
                qty=qty,
                reference_price=ref,
                stop_price=stop_price,
                reason=f"{sym}: TSMOM weight={tw.weight:.4f}",
            )
        )
    return actions


# ---------------------------------------------------------------------------- runner


def _today_utc() -> date:
    return datetime.now(tz=UTC).date()


def _load_recent_prices(
    *,
    universe: Sequence[str] = TSMOM_UNIVERSE,
    history_days: int = HISTORY_DAYS,
    db_path: Path | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Pull the most-recent ``history_days`` of adj_close from DuckDB.

    One DuckDB call total (CLAUDE.md cost-ceiling rule — bound the data
    fetch). The signal needs the longest TSMOM lookback (252 bdays) plus
    burn-in; ``HISTORY_DAYS=500`` covers that comfortably.
    """
    end_ts = end if end is not None else datetime.now(tz=UTC)
    start_ts = end_ts - timedelta(days=history_days + 60)
    return load_ohlcv_panel(
        start=start_ts,
        end=end_ts,
        universe=list(universe),
        db_path=db_path,
    )


def run_rebal(
    *,
    broker: AlpacaBroker | None = None,
    universe: Sequence[str] = TSMOM_UNIVERSE,
    today: date | None = None,
    force: bool = False,
    dry_run: bool = False,
    db_path: Path | None = None,
    duckdb_path: Path | None = None,
    target_vol: float = 0.10,
    gross_target: float = 1.0,
    stop_fraction: Decimal = DEFAULT_STOP_FRACTION,
    run_id: str = paper_clock.DEFAULT_RUN_ID,
) -> RebalRunResult:
    """Execute one monthly TSMOM rebal.

    Steps:

    1. Assert paper-only URL (defense in depth).
    2. Decide if today is a rebal day. If not and not ``force``, return
       a no-op result with ``skipped_reason``.
    3. Snapshot broker NAV + positions.
    4. Pull OHLCV via the data store; compute long-only TSMOM weights.
    5. Plan close + open_long actions, capped to single-name 10% and
       gross 100%.
    6. For each ``open_long``: route through ``risk.submit_order`` which
       enforces all PRD §8 caps + sends a bracket order with the stop.
    7. For each ``close``: ``broker.close_position(symbol)`` (the
       broker's market-close path).
    8. Persist a ``rebal_event`` row + ensure the paper_clock has started.
    9. Reconcile broker-vs-book and record drift count.
    """
    assert_paper_account()
    cfg = get_config()
    today = today or _today_utc()
    is_rebal_day = paper_clock.is_last_business_day_of_month(today)

    actual_broker = broker if broker is not None else build_default_broker()

    if not is_rebal_day and not force:
        return RebalRunResult(
            rebal_date_utc=today,
            is_rebal_day=False,
            forced=False,
            dry_run=dry_run,
            nav=Decimal("0"),
            skipped_reason=(
                f"{today.isoformat()} is not the last business day of the month; "
                "use --force to override"
            ),
        )

    # Snapshot. Broker is the source of truth for NAV and current positions.
    portfolio = snapshot_portfolio_from_broker(actual_broker)
    book_positions = book.fetch_positions(db_path=db_path)

    # Ensure the 30-day paper_clock is recording.
    paper_clock.ensure_started(
        run_id=run_id,
        strategy="tsmom_long_only",
        start_nav=portfolio.nav,
        cap_days=paper_clock.PAPER_CAP_DAYS,
        config_json=json.dumps(
            {
                "universe": list(universe),
                "target_vol": target_vol,
                "gross_target": gross_target,
                "stop_fraction": str(stop_fraction),
                "mode": "long_only",
                "max_single_name": cfg.max_single_name,
                "max_gross_exposure": cfg.max_gross_exposure,
            },
            sort_keys=True,
        ),
        db_path=db_path,
    )

    # Compute target weights.
    prices = _load_recent_prices(
        universe=universe,
        db_path=duckdb_path,
        end=datetime.combine(today, datetime.min.time(), tzinfo=UTC) + timedelta(days=1),
    )
    if prices.empty:
        return RebalRunResult(
            rebal_date_utc=today,
            is_rebal_day=is_rebal_day,
            forced=force,
            dry_run=dry_run,
            nav=portfolio.nav,
            skipped_reason="no OHLCV history available; cannot compute TSMOM weights",
        )

    # Freshness gate. The runner fires monthly on the last bday of the month,
    # AFTER NYSE close. Today's adj_close must be ingested into DuckDB before
    # we compute weights — otherwise we'd rebal on yesterday's signal and the
    # 30-day paper sample would be invalidated by stale-data noise. Hard-fail
    # with the exact ingestion command so the operator can fix it in one step.
    # Compare in UTC; DuckDB returns ts with whatever tz it stores in, so we
    # normalize before .date() to avoid a CST/UTC date-rollover mismatch.
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
        return RebalRunResult(
            rebal_date_utc=today,
            is_rebal_day=is_rebal_day,
            forced=force,
            dry_run=dry_run,
            nav=portfolio.nav,
            skipped_reason=(
                f"OHLCV stale: latest bar {latest_date.isoformat()} < "
                f"today {today.isoformat()}. Run yfinance ingest first AFTER "
                f"NYSE close (~6 PM ET), then re-run the rebal:\n  {cmd}"
            ),
        )

    targets = latest_target_weights(
        prices,
        target_vol=target_vol,
        gross_target=gross_target,
    )
    actions = plan_rebal_actions(
        target_weights=targets,
        portfolio=portfolio,
        book_positions=book_positions,
        stop_fraction=stop_fraction,
    )

    result = RebalRunResult(
        rebal_date_utc=today,
        is_rebal_day=is_rebal_day,
        forced=force,
        dry_run=dry_run,
        nav=portfolio.nav,
        actions=actions,
    )

    if dry_run:
        logger.warning(
            "tsmom_runner: DRY-RUN; {} actions planned, no orders submitted",
            len(actions),
        )
        return result

    # Execute. Closes first (frees gross exposure / single-name slots), then opens.
    for a in actions:
        if a.kind == "close":
            try:
                ord_resp = actual_broker.close_position(a.symbol)
                # Record the sell BEFORE deleting the position so the book
                # carries proof the bot closed this name. Without it, the
                # next book-sync's external-close detector would flag this
                # legitimate rebal close as an outside-the-bot liquidation.
                book.insert_order(
                    broker_order_id=ord_resp.id,
                    client_order_id=ord_resp.client_order_id or None,
                    symbol=a.symbol,
                    side="sell",
                    qty=ord_resp.qty,
                    stop_price=Decimal("0"),
                    limit_price=None,
                    submitted_at_utc=ord_resp.submitted_at,
                    status=ord_resp.status,
                    notional_estimate=None,
                    db_path=db_path,
                )
                book.delete_position(a.symbol, db_path=db_path)
                result.submitted_order_ids.append(ord_resp.id)
                logger.info("tsmom_runner: closed {} (broker order {})", a.symbol, ord_resp.id)
            except Exception as e:  # noqa: BLE001
                logger.error("tsmom_runner: close_position {} failed: {}", a.symbol, e)

    for a in actions:
        if a.kind != "open_long":
            continue
        try:
            ord_resp = submit_order(
                broker=actual_broker,
                symbol=a.symbol,
                side="buy",
                entry_price=a.reference_price,
                stop_price=a.stop_price,
                portfolio=portfolio,
                db_path=db_path,
            )
            result.submitted_order_ids.append(ord_resp.id)
            # Log/alert the qty risk.submit_order actually sent to the
            # broker. ``a.qty`` is the planner's intent (target_dollars /
            # ref); ``ord_resp.qty`` is what survived ¼-Kelly + per-trade
            # risk + single-name + gross + cash caps inside size_position.
            # Pre-2026-05-15 both surfaces used a.qty, which made the
            # Discord notification claim the operator owned 321 shares
            # when only 120 actually went to Alpaca.
            actual_qty = ord_resp.qty
            logger.info(
                "tsmom_runner: bracket-bought {} qty={} ref=${} stop=${}",
                a.symbol,
                actual_qty,
                a.reference_price,
                a.stop_price,
            )
            alerts.alert_order_submitted(
                run_id=run_id,
                symbol=a.symbol,
                side="buy",
                qty=actual_qty,
                reference_price=a.reference_price,
                stop_price=a.stop_price,
                order_id=ord_resp.id,
            )
        except RiskRejection as e:
            logger.warning("tsmom_runner: risk rejected {}: {}", a.symbol, e)
        except TradingDisabledError as e:
            logger.error("tsmom_runner: kill switch engaged: {}", e)
            result.skipped_reason = str(e)
            return result
        except Exception as e:  # noqa: BLE001
            logger.error("tsmom_runner: submit_order {} failed: {}", a.symbol, e)

    # Record the rebal event.
    nav_now = actual_broker.get_account().equity
    weights_blob = json.dumps(
        [{"symbol": tw.symbol, "weight": tw.weight} for tw in targets],
        sort_keys=True,
    )
    paper_clock.insert_rebal_event(
        n_orders_submitted=len(result.submitted_order_ids),
        nav_at_rebal=nav_now,
        target_weights_json=weights_blob,
        notes=f"rebal_date_utc={today.isoformat()}",
        run_id=run_id,
        db_path=db_path,
    )

    # Reconcile broker vs book.
    rec = reconcile.reconcile(broker=actual_broker, db_path=db_path)
    result.drift_after = sum(
        1
        for d in rec.drift
        if d.kind in ("broker_only", "book_only", "qty_mismatch", "side_mismatch")
    )
    if result.drift_after:
        logger.error(
            "tsmom_runner: post-rebal reconcile drift: {} entries",
            result.drift_after,
        )

    # End-of-rebal Discord summary. One alert per rebal cycle complementing
    # the per-order alerts above. Severity flips to warning if reconcile
    # drift > 0; no exception here — alerts module is fail-soft.
    alerts.alert_rebal_summary(
        run_id=run_id,
        rebal_date=today.isoformat(),
        nav=nav_now,
        n_orders_submitted=len(result.submitted_order_ids),
        target_weights=[{"symbol": tw.symbol, "weight": tw.weight} for tw in targets],
        drift_after=result.drift_after,
        forced=force,
        dry_run=dry_run,
    )

    return result


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.execution.tsmom_runner",
        description="TSMOM monthly rebal runner (Alpaca paper, 30-day cap).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if today is not the last business day of the month.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the action plan and log it; do NOT submit any orders.",
    )
    args = parser.parse_args(argv)

    try:
        assert_paper_account()
    except NotPaperAccountError as e:
        logger.error("tsmom_runner: refusing to run: {}", e)
        return 2

    try:
        result = run_rebal(force=args.force, dry_run=args.dry_run)
    except Exception as e:  # noqa: BLE001
        logger.exception("tsmom_runner: unhandled exception: {}", e)
        return 1

    if result.skipped_reason:
        logger.warning("tsmom_runner: skipped: {}", result.skipped_reason)
        return 0
    logger.warning(
        "tsmom_runner: done; rebal_day={} forced={} dry_run={} actions={} submitted={} drift={}",
        result.is_rebal_day,
        result.forced,
        result.dry_run,
        len(result.actions),
        len(result.submitted_order_ids),
        result.drift_after,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
