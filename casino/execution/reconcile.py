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
from decimal import Decimal
from pathlib import Path

from loguru import logger

from casino.execution import book
from casino.execution.alpaca_broker import AlpacaBroker, BrokerPosition

DRIFT_QTY_THRESHOLD: int = 1
DRIFT_NOTIONAL_THRESHOLD: Decimal = Decimal("1.00")


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
                    detail=f"broker holds {b.qty} {b.side} of {sym}; book has no row",
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
                    detail=f"book has {k.qty} {k.side} of {sym}; broker has no position",
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
                    detail=f"side mismatch on {sym}: broker={b.side} book={k.side}",
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
                    detail=f"qty mismatch on {sym}: broker={b.qty} book={k.qty}",
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
                        f"price drift on {sym}: broker MV={b.market_value} "
                        f"book entry-notional={book_notional} (Δ={diff})"
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
