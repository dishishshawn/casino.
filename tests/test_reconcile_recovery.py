"""Tests for the 2026-06 book-recovery fixes in casino.execution.reconcile.

Covers the three failure modes traced from the June 2-3 incident:

* ``record_fills_from_broker`` — orders advance past ``accepted`` and the
  ``fills`` table is populated (the dead audit trail).
* ``ensure_protective_stops`` — every open long gets a live GTC stop, fixing
  the bracket stop-leg expiring at the first session close.
* ``detect_external_closes`` — a position that vanishes at the broker with no
  bot sell order is surfaced instead of being silently overwritten.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from casino.config import get_config
from casino.execution import book, reconcile
from casino.execution.alpaca_broker import (
    AlpacaBroker,
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
)
from tests.test_risk import FakeTradingClient


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path / "state.sqlite"
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(state))
    get_config.cache_clear()
    book.init_schema(state)
    return state


def _account() -> BrokerAccount:
    return BrokerAccount(
        account_number="paper-1",
        status="ACTIVE",
        equity=Decimal("100000"),
        cash=Decimal("100000"),
        buying_power=Decimal("100000"),
        last_equity=Decimal("100000"),
        pattern_day_trader=False,
        trading_blocked=False,
    )


def _long(symbol: str, qty: int, avg: str) -> BrokerPosition:
    px = Decimal(avg)
    return BrokerPosition(
        symbol=symbol,
        qty=qty,
        side="long",
        avg_entry_price=px,
        market_price=px,
        market_value=px * qty,
        unrealized_pl=Decimal("0"),
        cost_basis=px * qty,
    )


def _broker(
    *,
    positions: list[BrokerPosition] | None = None,
    order_history: list[BrokerOrder] | None = None,
) -> tuple[AlpacaBroker, FakeTradingClient]:
    fake = FakeTradingClient(
        account=_account(),
        positions=positions,
        order_history=order_history,
    )
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    broker.set_client(fake)
    return broker, fake


# ---------------------------------------------------------------------------- fills


def test_record_fills_advances_orders_and_populates_fills(isolated_state: Path) -> None:
    # The book has the entry recorded at submit time (status accepted), like
    # the 14 frozen rows on the live book.
    book.insert_order(
        broker_order_id="ord-9",
        client_order_id="casino-9",
        symbol="SPY",
        side="buy",
        qty=4,
        stop_price=Decimal("600"),
        limit_price=None,
        submitted_at_utc=None,
        status="accepted",
        notional_estimate=Decimal("3000"),
        db_path=isolated_state,
    )
    filled = BrokerOrder(
        id="ord-9",
        client_order_id="casino-9",
        symbol="SPY",
        side="buy",
        qty=4,
        filled_qty=4,
        status="filled",
        order_type="market",
        submitted_at=None,
        filled_at=None,
        filled_avg_price=Decimal("750.10"),
        stop_price=None,
        limit_price=None,
    )
    broker, _ = _broker(order_history=[filled])

    n = reconcile.record_fills_from_broker(broker=broker, db_path=isolated_state)

    assert n == 1
    assert book.has_fill("ord-9", db_path=isolated_state) is True
    # Order advanced past "accepted".
    open_orders = book.fetch_open_orders(db_path=isolated_state)
    assert all(o.broker_order_id != "ord-9" for o in open_orders)


def test_record_fills_is_idempotent(isolated_state: Path) -> None:
    filled = BrokerOrder(
        id="ord-1",
        client_order_id="casino-1",
        symbol="QQQ",
        side="buy",
        qty=5,
        filled_qty=5,
        status="filled",
        order_type="market",
        submitted_at=None,
        filled_at=None,
        filled_avg_price=Decimal("400.00"),
        stop_price=None,
        limit_price=None,
    )
    broker, _ = _broker(order_history=[filled])

    first = reconcile.record_fills_from_broker(broker=broker, db_path=isolated_state)
    second = reconcile.record_fills_from_broker(broker=broker, db_path=isolated_state)

    assert first == 1
    assert second == 0  # already recorded; no double-count


# ---------------------------------------------------------------------------- stops


def test_ensure_stops_arms_unprotected_long(isolated_state: Path) -> None:
    broker, fake = _broker(positions=[_long("SPY", 4, "750")])

    results = reconcile.ensure_protective_stops(
        broker=broker,
        stop_fraction=Decimal("0.10"),
        db_path=isolated_state,
    )

    assert len(results) == 1
    r = results[0]
    assert r.armed is True
    assert r.already_protected is False
    # 750 * (1 - 0.10) = 675.00, rounded down to the cent.
    assert r.stop_price == Decimal("675.00")
    # A stop order actually went to the broker, recorded in the book.
    assert any(getattr(req, "stop_price", None) is not None for req in fake.submitted_requests)
    orders = book.fetch_open_orders(db_path=isolated_state)
    assert any(o.symbol == "SPY" and o.side == "sell" for o in orders)


def test_ensure_stops_skips_already_protected(isolated_state: Path) -> None:
    broker, _ = _broker(positions=[_long("SPY", 4, "750")])

    # First pass arms the stop; the fake registers it as an open stop order.
    reconcile.ensure_protective_stops(
        broker=broker, stop_fraction=Decimal("0.10"), db_path=isolated_state
    )
    # Second pass must see it as protected and arm nothing new.
    again = reconcile.ensure_protective_stops(
        broker=broker, stop_fraction=Decimal("0.10"), db_path=isolated_state
    )

    assert len(again) == 1
    assert again[0].already_protected is True
    assert again[0].armed is False


def test_ensure_stops_dry_run_submits_nothing(isolated_state: Path) -> None:
    broker, fake = _broker(positions=[_long("IWM", 12, "260")])

    results = reconcile.ensure_protective_stops(
        broker=broker,
        stop_fraction=Decimal("0.10"),
        db_path=isolated_state,
        dry_run=True,
    )

    assert results[0].armed is False
    assert fake.submitted_requests == []


# ---------------------------------------------------------------------------- external closes


def test_detect_external_close_flags_unexplained_disappearance(isolated_state: Path) -> None:
    # Book holds AAA; broker holds nothing; no sell order on record.
    book.upsert_position(
        symbol="AAA", side="long", qty=10, avg_entry_price=Decimal("50"), db_path=isolated_state
    )
    broker, _ = _broker(positions=[])

    external = reconcile.detect_external_closes(broker=broker, db_path=isolated_state)

    assert external == ["AAA"]


def test_detect_external_close_ignores_bot_initiated_close(isolated_state: Path) -> None:
    book.upsert_position(
        symbol="AAA", side="long", qty=10, avg_entry_price=Decimal("50"), db_path=isolated_state
    )
    # The bot's rebal close recorded a sell order — this disappearance is
    # explained.
    book.insert_order(
        broker_order_id="close-AAA",
        client_order_id=None,
        symbol="AAA",
        side="sell",
        qty=10,
        stop_price=Decimal("0"),
        limit_price=None,
        submitted_at_utc=None,
        status="accepted",
        notional_estimate=None,
        db_path=isolated_state,
    )
    broker, _ = _broker(positions=[])

    external = reconcile.detect_external_closes(broker=broker, db_path=isolated_state)

    assert external == []


def test_detect_external_close_silent_when_kill_engaged(isolated_state: Path) -> None:
    book.upsert_position(
        symbol="AAA", side="long", qty=10, avg_entry_price=Decimal("50"), db_path=isolated_state
    )
    book.set_trading_disabled(True, reason="test", db_path=isolated_state)
    broker, _ = _broker(positions=[])

    external = reconcile.detect_external_closes(broker=broker, db_path=isolated_state)

    assert external == []  # a kill flatten is expected + already alerted
