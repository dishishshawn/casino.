"""Broker-vs-internal-book reconciliation.

The source of truth for "what we actually hold." Any module reasoning about
positions reads from reconcile, not from the broker API directly and not from
local order state alone (CLAUDE.md §4.1).

PRD §8 / §10 alert thresholds: drift > $1 of notional or > 1 share triggers
a critical alert. This module produces structured `Drift` records; pushing
those through the Discord alert path is the alerts module's job.

Money values are `Decimal` end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

from loguru import logger

from casino.execution import book
from casino.execution.alpaca_broker import AlpacaBroker, BrokerPosition

DRIFT_QTY_THRESHOLD: int = 1
DRIFT_NOTIONAL_THRESHOLD: Decimal = Decimal("1.00")

# Default protective-stop distance. Mirrors tsmom_runner.DEFAULT_STOP_FRACTION
# but is duplicated here to avoid importing the runner (which would create an
# execution-layer import cycle: runner → risk → reconcile). The EOD job passes
# the run's actual configured stop_fraction from the paper_clock when available.
DEFAULT_STOP_FRACTION: Decimal = Decimal("0.10")


@dataclass(frozen=True)
class Drift:
    """One symbol's discrepancy between broker and internal book.

    `kind` is one of:
        * ``broker_only``    — broker has position, book does not
        * ``book_only``      — book has position, broker does not
        * ``qty_mismatch``   — both sides hold the symbol but qty differs
        * ``side_mismatch``  — sides disagree (long vs short)
        * ``price_drift``    — qty matches but mark-to-market deviates
                               by > $1 of notional
    """

    symbol: str
    kind: str
    broker_qty: int
    book_qty: int
    broker_side: str
    book_side: str
    broker_notional: Decimal
    book_notional: Decimal
    detail: str


@dataclass(frozen=True)
class ReconciliationResult:
    """Structured output of one reconciliation pass."""

    as_of: datetime
    n_broker_positions: int
    n_book_positions: int
    drift: list[Drift]
    in_sync: bool

    @property
    def has_drift(self) -> bool:
        return len(self.drift) > 0


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _key(p: BrokerPosition) -> str:
    return p.symbol.upper()


def reconcile(
    *,
    broker: AlpacaBroker,
    db_path: Path | None = None,
    qty_threshold: int = DRIFT_QTY_THRESHOLD,
    notional_threshold: Decimal = DRIFT_NOTIONAL_THRESHOLD,
) -> ReconciliationResult:
    """Compare broker positions to the SQLite book; return structured drift.

    Drift detection rules:

    * Symbol in broker but not book → ``broker_only`` drift.
    * Symbol in book but not broker → ``book_only`` drift.
    * Both sides hold but `qty` differs by ≥ ``qty_threshold`` shares →
      ``qty_mismatch``.
    * Sides differ → ``side_mismatch`` (always reported regardless of qty).
    * Same qty + side but broker market value differs from book entry-cost
      by ≥ ``notional_threshold`` → ``price_drift`` (informational; not a
      bug per se, but logged).

    The first three are the ones the alerts module fires on; price_drift
    is informational.
    """
    broker_positions = broker.get_positions()
    book_positions = book.fetch_positions(db_path=db_path)

    by_broker = {p.symbol.upper(): p for p in broker_positions}
    by_book = {p.symbol.upper(): p for p in book_positions}

    all_symbols = sorted(set(by_broker.keys()) | set(by_book.keys()))
    drifts: list[Drift] = []

    for sym in all_symbols:
        b = by_broker.get(sym)
        k = by_book.get(sym)

        if b is not None and k is None:
            drifts.append(
                Drift(
                    symbol=sym,
                    kind="broker_only",
                    broker_qty=b.qty,
                    book_qty=0,
                    broker_side=b.side,
                    book_side="",
                    broker_notional=b.market_value,
                    book_notional=Decimal("0"),
                    detail=(
                        f"Broker shows {b.qty} {b.side} of {sym}, but the "
                        "system has no record of it"
                    ),
                )
            )
            continue
        if k is not None and b is None:
            book_notional = Decimal(k.qty) * k.avg_entry_price
            drifts.append(
                Drift(
                    symbol=sym,
                    kind="book_only",
                    broker_qty=0,
                    book_qty=k.qty,
                    broker_side="",
                    book_side=k.side,
                    broker_notional=Decimal("0"),
                    book_notional=book_notional,
                    detail=(
                        f"System shows {k.qty} {k.side} of {sym}, but the "
                        "broker has no matching position"
                    ),
                )
            )
            continue
        # both present
        assert b is not None and k is not None
        if b.side != k.side:
            drifts.append(
                Drift(
                    symbol=sym,
                    kind="side_mismatch",
                    broker_qty=b.qty,
                    book_qty=k.qty,
                    broker_side=b.side,
                    book_side=k.side,
                    broker_notional=b.market_value,
                    book_notional=Decimal(k.qty) * k.avg_entry_price,
                    detail=(
                        f"Direction mismatch on {sym}: broker shows {b.side}, system shows {k.side}"
                    ),
                )
            )
            continue
        if abs(b.qty - k.qty) >= qty_threshold:
            drifts.append(
                Drift(
                    symbol=sym,
                    kind="qty_mismatch",
                    broker_qty=b.qty,
                    book_qty=k.qty,
                    broker_side=b.side,
                    book_side=k.side,
                    broker_notional=b.market_value,
                    book_notional=Decimal(k.qty) * k.avg_entry_price,
                    detail=(
                        f"Quantity mismatch on {sym}: broker shows {b.qty} "
                        f"shares, system shows {k.qty}"
                    ),
                )
            )
            continue
        # qty + side match → check price drift (informational)
        book_notional = Decimal(k.qty) * k.avg_entry_price
        diff = b.market_value - book_notional
        if abs(diff) >= notional_threshold:
            drifts.append(
                Drift(
                    symbol=sym,
                    kind="price_drift",
                    broker_qty=b.qty,
                    book_qty=k.qty,
                    broker_side=b.side,
                    book_side=k.side,
                    broker_notional=b.market_value,
                    book_notional=book_notional,
                    detail=(
                        f"Mark-to-market difference on {sym}: broker market "
                        f"value=${b.market_value}, system entry-cost="
                        f"${book_notional} (difference ${diff})"
                    ),
                )
            )

    if drifts:
        for d in drifts:
            level = (
                "critical"
                if d.kind in {"broker_only", "book_only", "qty_mismatch", "side_mismatch"}
                else "info"
            )
            log = logger.error if level == "critical" else logger.info
            log("reconcile drift [{}]: {}", d.kind, d.detail)
    else:
        logger.debug(
            "reconcile in sync: {} broker / {} book positions",
            len(broker_positions),
            len(book_positions),
        )

    return ReconciliationResult(
        as_of=_utc_now(),
        n_broker_positions=len(broker_positions),
        n_book_positions=len(book_positions),
        drift=drifts,
        in_sync=len(drifts) == 0,
    )


def critical_drift(result: ReconciliationResult) -> list[Drift]:
    """Filter `result.drift` to entries that warrant a critical alert.

    Price drift is excluded — it's informational only. Quantity, presence,
    and side mismatches are real bugs that need operator attention.
    """
    crit = {"broker_only", "book_only", "qty_mismatch", "side_mismatch"}
    return [d for d in result.drift if d.kind in crit]


def sync_book_from_broker(
    *,
    broker: AlpacaBroker,
    db_path: Path | None = None,
) -> int:
    """Overwrite the internal book to match the broker.

    The escape hatch for resolving drift in the operator workflow (RUNBOOK).
    Returns the number of positions written.
    """
    positions = broker.get_positions()
    # Wipe-and-rewrite. Cheap because positions ≪ 1000 rows.
    book.init_schema(db_path)
    with book.get_book_conn(db_path) as conn:
        conn.execute("DELETE FROM positions")
    for p in positions:
        book.upsert_position(
            symbol=p.symbol,
            side=p.side,
            qty=p.qty,
            avg_entry_price=p.avg_entry_price,
            db_path=db_path,
        )
    logger.warning("reconcile: book overwritten from broker ({} positions)", len(positions))
    return len(positions)


# ---------------------------------------------------------------------------- fills


def record_fills_from_broker(
    *,
    broker: AlpacaBroker,
    db_path: Path | None = None,
) -> int:
    """Advance the internal book from the broker's order history.

    Closes the fill-tracking gap behind the 2026-05/06 reconcile drift: the
    book recorded each order at *submission* (status ``accepted``) and never
    advanced it. The ``fills`` table stayed empty and ``orders`` rows were
    frozen at submit time, so the book was blind to what actually executed.

    For every order the broker reports, this:

    * updates the matching book ``orders`` row's status (no-op if the book
      never saw that order — e.g. an externally-placed close), and
    * records a ``fills`` row for any order that reached a terminal filled
      state and isn't already recorded.

    Idempotent via ``book.has_fill`` — safe to run every EOD. Returns the
    number of newly recorded fills.
    """
    broker_orders = broker.get_orders(status="all")
    recorded = 0
    for o in broker_orders:
        # Sync status onto any book row we own (UPDATE WHERE matches nothing
        # for orders the book never recorded, which is fine).
        book.update_order_status(broker_order_id=o.id, status=o.status, db_path=db_path)
        if (
            o.status == "filled"
            and o.filled_qty > 0
            and o.filled_avg_price is not None
            and not book.has_fill(o.id, db_path=db_path)
        ):
            book.insert_fill(
                broker_order_id=o.id,
                symbol=o.symbol,
                side=o.side,
                qty=o.filled_qty,
                price=o.filled_avg_price,
                filled_at_utc=o.filled_at,
                db_path=db_path,
            )
            recorded += 1
    if recorded:
        logger.info("reconcile: recorded {} new fill(s) from broker", recorded)
    return recorded


# ---------------------------------------------------------------------------- stops


@dataclass(frozen=True)
class StopArmResult:
    """Outcome of the protective-stop guard for one position."""

    symbol: str
    qty: int
    stop_price: Decimal
    already_protected: bool
    armed: bool
    order_id: str | None


def ensure_protective_stops(
    *,
    broker: AlpacaBroker,
    stop_fraction: Decimal = DEFAULT_STOP_FRACTION,
    db_path: Path | None = None,
    dry_run: bool = False,
) -> list[StopArmResult]:
    """Guarantee every open long has a live broker-side stop (hard rule 3).

    The bracket entry attaches a stop leg, but Alpaca forces ``day`` tif on a
    market entry, so that leg expires at the first session close and the
    position is left naked (observed 2026-06-01). This guard runs at EOD: for
    each open long with no live stop order at the broker, it submits a fresh
    GTC sell-stop at ``avg_entry * (1 - stop_fraction)`` (rounded down to the
    cent, the conservative direction for a long's stop) and records it.

    Long-only by design (v1 TSMOM). Returns one result per position so the
    caller can alert on whatever had to be re-armed.
    """
    positions = broker.get_positions()
    open_orders = broker.get_orders(status="open")
    protected = {
        o.symbol.upper() for o in open_orders if o.side == "sell" and o.stop_price is not None
    }

    results: list[StopArmResult] = []
    for p in positions:
        if p.side != "long":
            # v1 is long-only; a short would need a buy-stop. Surface it
            # rather than silently skipping so the operator notices.
            logger.warning(
                "ensure_protective_stops: skipping non-long position {} {}",
                p.side,
                p.symbol,
            )
            continue
        sym = p.symbol.upper()
        stop_px = (p.avg_entry_price * (Decimal("1") - stop_fraction)).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        )
        if sym in protected:
            results.append(
                StopArmResult(
                    symbol=sym,
                    qty=p.qty,
                    stop_price=stop_px,
                    already_protected=True,
                    armed=False,
                    order_id=None,
                )
            )
            continue
        if dry_run:
            logger.warning(
                "ensure_protective_stops: {} UNPROTECTED (would arm GTC stop @ {})",
                sym,
                stop_px,
            )
            results.append(
                StopArmResult(
                    symbol=sym,
                    qty=p.qty,
                    stop_price=stop_px,
                    already_protected=False,
                    armed=False,
                    order_id=None,
                )
            )
            continue
        order = broker.submit_stop_order(
            symbol=sym,
            qty=p.qty,
            side="sell",
            stop_price=stop_px,
            time_in_force="gtc",
        )
        book.insert_order(
            broker_order_id=order.id,
            client_order_id=order.client_order_id or None,
            symbol=sym,
            side="sell",
            qty=p.qty,
            stop_price=stop_px,
            limit_price=None,
            submitted_at_utc=order.submitted_at,
            status=order.status,
            notional_estimate=None,
            db_path=db_path,
        )
        logger.warning(
            "ensure_protective_stops: re-armed {} with GTC stop @ {} (order {})",
            sym,
            stop_px,
            order.id,
        )
        results.append(
            StopArmResult(
                symbol=sym,
                qty=p.qty,
                stop_price=stop_px,
                already_protected=False,
                armed=True,
                order_id=order.id,
            )
        )
    return results


# ---------------------------------------------------------------------------- external-change detection


def detect_external_closes(
    *,
    broker: AlpacaBroker,
    db_path: Path | None = None,
) -> list[str]:
    """Symbols the book holds that vanished at the broker with no bot sell.

    A position the bot closed (rebal or kill switch) leaves a sell order in
    the book. A position that's in the book but gone at the broker *without*
    such a sell order was flattened outside the system — the 2026-06-02/03
    failure mode. ``jobs.sync_book`` calls this BEFORE overwriting the book so
    the disappearance is alerted instead of silently absorbed.

    Returns the list of unexplained symbols (empty on a clean day).
    """
    broker_syms = {p.symbol.upper() for p in broker.get_positions()}
    book_syms = {p.symbol.upper() for p in book.fetch_positions(db_path=db_path)}
    vanished = sorted(book_syms - broker_syms)
    if book.is_trading_disabled(db_path=db_path):
        # Kill switch engaged: a flatten is expected and already alerted.
        return []
    return [s for s in vanished if not book.has_recent_sell_order(s, db_path=db_path)]
