"""Post-fill book sync — populate book.positions from the broker.

Runs ~15 min after market open. Closes the gap exposed by the 2026-05-15
incident where bracket-order fills landed at 09:30 ET but book.positions
stayed empty until the 17:30 reconcile, at which point the auto-kill
criterion saw full-notional drift and flattened everything.

Run: ``uv run python -m casino.jobs.sync_book``
Cron: Mon-Fri 09:45 local CT (Casino_Daily_TSMOM_BookSync task).
"""

from __future__ import annotations

import sys

from loguru import logger

from casino.execution import reconcile
from casino.execution.alpaca_broker import build_default_broker
from casino.monitoring import alerts


def main() -> int:
    try:
        broker = build_default_broker()
        # Record any fills first so bot-initiated closes are on the book and
        # don't read as "external" below.
        reconcile.record_fills_from_broker(broker=broker)
        # Detect positions that disappeared at the broker with no bot sell
        # order BEFORE the overwrite silently absorbs them (2026-06-02/03).
        external = reconcile.detect_external_closes(broker=broker)
        if external:
            logger.error("sync_book: positions closed outside the bot: {}", external)
            alerts.alert_external_position_change(symbols=external)
        n = reconcile.sync_book_from_broker(broker=broker)
        logger.info("sync_book: wrote {} broker positions into book.positions", n)
        return 0
    except Exception as e:  # noqa: BLE001
        alerts.alert_unhandled_exception(
            job="sync_book",
            exc_type=type(e).__name__,
            detail=str(e),
        )
        logger.exception("sync_book: unhandled exception")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
