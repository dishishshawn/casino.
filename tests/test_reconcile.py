"""Tests for casino.execution.reconcile."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from casino.config import get_config
from casino.execution import book, reconcile
from casino.execution.alpaca_broker import (
    AlpacaBroker,
    BrokerAccount,
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


def _broker_with(positions: list[BrokerPosition]) -> AlpacaBroker:
    account = BrokerAccount(
        account_number="paper-1",
        status="ACTIVE",
        equity=Decimal("100000"),
        cash=Decimal("100000"),
        buying_power=Decimal("100000"),
        last_equity=Decimal("100000"),
        pattern_day_trader=False,
        trading_blocked=False,
    )
    fake = FakeTradingClient(account=account, positions=positions)
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    broker.set_client(fake)
    return broker


def test_reconcile_in_sync(isolated_state: Path) -> None:
    book.upsert_position(
        symbol="AAA",
        side="long",
        qty=10,
        avg_entry_price=Decimal("100"),
        db_path=isolated_state,
    )
    broker = _broker_with(
        [
            BrokerPosition(
                symbol="AAA",
                qty=10,
                side="long",
                avg_entry_price=Decimal("100"),
                market_price=Decimal("100"),
                market_value=Decimal("1000"),
                unrealized_pl=Decimal("0"),
                cost_basis=Decimal("1000"),
            )
        ]
    )
    result = reconcile.reconcile(broker=broker, db_path=isolated_state)
    assert result.in_sync is True
    assert result.drift == []


def test_reconcile_broker_only_drift(isolated_state: Path) -> None:
    """Broker holds a position the book doesn't know about → critical."""
    broker = _broker_with(
        [
            BrokerPosition(
                symbol="ZZZ",
                qty=7,
                side="long",
                avg_entry_price=Decimal("50"),
                market_price=Decimal("51"),
                market_value=Decimal("357"),
                unrealized_pl=Decimal("7"),
                cost_basis=Decimal("350"),
            )
        ]
    )
    result = reconcile.reconcile(broker=broker, db_path=isolated_state)
    assert not result.in_sync
    crit = reconcile.critical_drift(result)
    assert len(crit) == 1
    assert crit[0].kind == "broker_only"
    assert crit[0].symbol == "ZZZ"


def test_reconcile_book_only_drift(isolated_state: Path) -> None:
    """Book holds a position the broker doesn't → critical."""
    book.upsert_position(
        symbol="AAA",
        side="long",
        qty=4,
        avg_entry_price=Decimal("10"),
        db_path=isolated_state,
    )
    broker = _broker_with([])
    result = reconcile.reconcile(broker=broker, db_path=isolated_state)
    crit = reconcile.critical_drift(result)
    assert len(crit) == 1
    assert crit[0].kind == "book_only"


def test_reconcile_qty_mismatch_alerts(isolated_state: Path) -> None:
    book.upsert_position(
        symbol="AAA",
        side="long",
        qty=10,
        avg_entry_price=Decimal("100"),
        db_path=isolated_state,
    )
    broker = _broker_with(
        [
            BrokerPosition(
                symbol="AAA",
                qty=12,  # broker has 12, book thinks 10
                side="long",
                avg_entry_price=Decimal("100"),
                market_price=Decimal("100"),
                market_value=Decimal("1200"),
                unrealized_pl=Decimal("0"),
                cost_basis=Decimal("1200"),
            )
        ]
    )
    result = reconcile.reconcile(broker=broker, db_path=isolated_state)
    crit = reconcile.critical_drift(result)
    assert len(crit) == 1
    assert crit[0].kind == "qty_mismatch"


def test_reconcile_side_mismatch(isolated_state: Path) -> None:
    book.upsert_position(
        symbol="AAA",
        side="long",
        qty=10,
        avg_entry_price=Decimal("100"),
        db_path=isolated_state,
    )
    broker = _broker_with(
        [
            BrokerPosition(
                symbol="AAA",
                qty=10,
                side="short",
                avg_entry_price=Decimal("100"),
                market_price=Decimal("100"),
                market_value=Decimal("1000"),
                unrealized_pl=Decimal("0"),
                cost_basis=Decimal("1000"),
            )
        ]
    )
    result = reconcile.reconcile(broker=broker, db_path=isolated_state)
    crit = reconcile.critical_drift(result)
    assert len(crit) == 1
    assert crit[0].kind == "side_mismatch"


def test_reconcile_price_drift_is_informational(isolated_state: Path) -> None:
    """Price drift is reported but excluded from `critical_drift`."""
    book.upsert_position(
        symbol="AAA",
        side="long",
        qty=10,
        avg_entry_price=Decimal("100"),
        db_path=isolated_state,
    )
    broker = _broker_with(
        [
            BrokerPosition(
                symbol="AAA",
                qty=10,
                side="long",
                avg_entry_price=Decimal("100"),
                market_price=Decimal("105"),
                market_value=Decimal("1050"),  # +$50 vs book entry $1000
                unrealized_pl=Decimal("50"),
                cost_basis=Decimal("1000"),
            )
        ]
    )
    result = reconcile.reconcile(broker=broker, db_path=isolated_state)
    assert any(d.kind == "price_drift" for d in result.drift)
    crit = reconcile.critical_drift(result)
    assert crit == []


def test_sync_book_from_broker_overwrites(isolated_state: Path) -> None:
    """Operator escape hatch: rewrite the book from the broker."""
    book.upsert_position(
        symbol="AAA",
        side="long",
        qty=10,
        avg_entry_price=Decimal("100"),
        db_path=isolated_state,
    )
    broker = _broker_with(
        [
            BrokerPosition(
                symbol="ZZZ",
                qty=3,
                side="long",
                avg_entry_price=Decimal("20"),
                market_price=Decimal("20"),
                market_value=Decimal("60"),
                unrealized_pl=Decimal("0"),
                cost_basis=Decimal("60"),
            )
        ]
    )
    n = reconcile.sync_book_from_broker(broker=broker, db_path=isolated_state)
    assert n == 1
    rows = book.fetch_positions(db_path=isolated_state)
    assert [p.symbol for p in rows] == ["ZZZ"]
