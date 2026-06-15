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


def test_sync_book_resolves_post_fill_drift(isolated_state: Path) -> None:
    """Regression for 2026-05-15: post-fill broker holds N positions, book is
    empty, reconcile sees broker_only drift for every entry. After sync_book,
    reconcile must be in_sync. This is the gate that prevents the auto-kill
    from firing on freshly-filled bracket orders (task 50)."""
    broker_positions = [
        BrokerPosition(
            symbol=sym,
            qty=10,
            side="long",
            avg_entry_price=Decimal("100"),
            market_price=Decimal("100"),
            market_value=Decimal("1000"),
            unrealized_pl=Decimal("0"),
            cost_basis=Decimal("1000"),
        )
        for sym in ("SPY", "QQQ", "IWM", "EFA", "EEM", "DBC", "USO")
    ]
    broker = _broker_with(broker_positions)

    pre = reconcile.reconcile(broker=broker, db_path=isolated_state)
    assert not pre.in_sync
    assert len(reconcile.critical_drift(pre)) == len(broker_positions)

    reconcile.sync_book_from_broker(broker=broker, db_path=isolated_state)
    post = reconcile.reconcile(broker=broker, db_path=isolated_state)
    assert post.in_sync
    assert post.drift == []


# --------------------------------------------------------------- protective stops


def _long(symbol: str, entry: str, market: str, qty: int = 10) -> BrokerPosition:
    return BrokerPosition(
        symbol=symbol,
        qty=qty,
        side="long",
        avg_entry_price=Decimal(entry),
        market_price=Decimal(market),
        market_value=Decimal(market) * qty,
        unrealized_pl=Decimal("0"),
        cost_basis=Decimal(entry) * qty,
    )


def test_ensure_stops_arms_normal_position(isolated_state: Path) -> None:
    # entry 100 -> stop 90; market 100 is well above the stop → arm.
    broker = _broker_with([_long("AAA", "100", "100")])
    results = reconcile.ensure_protective_stops(broker=broker, db_path=isolated_state)
    r = next(x for x in results if x.symbol == "AAA")
    assert r.armed is True
    assert r.breached is False
    assert r.liquidated is False


def test_ensure_stops_flags_breached_without_arming(isolated_state: Path) -> None:
    # entry 100 -> stop 90; market 80 is below the stop → cannot arm, must flag.
    broker = _broker_with([_long("USO", "100", "80")])
    results = reconcile.ensure_protective_stops(broker=broker, db_path=isolated_state)
    r = next(x for x in results if x.symbol == "USO")
    assert r.breached is True
    assert r.armed is False
    assert r.liquidated is False
    # no stop order was submitted for the breached symbol
    assert all(
        getattr(req, "symbol", "") != "USO"
        for req in broker._client.submitted_requests  # type: ignore[attr-defined]
    )


def test_ensure_stops_isolates_one_failure(isolated_state: Path) -> None:
    # AAA and BBB both need arming; BBB's broker call raises. AAA must survive.
    broker = _broker_with([_long("AAA", "100", "100"), _long("BBB", "100", "100")])
    broker._client._fail_symbols = {"BBB"}  # type: ignore[attr-defined]
    results = reconcile.ensure_protective_stops(broker=broker, db_path=isolated_state)
    by_sym = {r.symbol: r for r in results}
    assert by_sym["AAA"].armed is True
    assert by_sym["BBB"].armed is False


def test_ensure_stops_liquidates_material_breach_when_open(isolated_state: Path) -> None:
    # entry 100 -> stop 90; market 80 is well past the cushion, market open.
    broker = _broker_with([_long("USO", "100", "80")])  # FakeTradingClient market_open=True
    results = reconcile.ensure_protective_stops(
        broker=broker, db_path=isolated_state, liquidate_breached=True
    )
    r = next(x for x in results if x.symbol == "USO")
    assert r.breached is True
    assert r.liquidated is True
    assert "USO" in broker._client.closed_positions  # type: ignore[attr-defined]


def test_ensure_stops_no_liquidation_when_market_closed(isolated_state: Path) -> None:
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
    fake = FakeTradingClient(
        account=account, positions=[_long("USO", "100", "80")], market_open=False
    )
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    broker.set_client(fake)
    results = reconcile.ensure_protective_stops(
        broker=broker, db_path=isolated_state, liquidate_breached=True
    )
    r = next(x for x in results if x.symbol == "USO")
    assert r.breached is True
    assert r.liquidated is False
    assert fake.closed_positions == []


def test_ensure_stops_cushion_blocks_tiny_breach(isolated_state: Path) -> None:
    # entry 100 -> stop 90; market 89.95 is < 0.25% below the stop → no sale.
    broker = _broker_with([_long("AAA", "100", "89.95")])
    results = reconcile.ensure_protective_stops(
        broker=broker, db_path=isolated_state, liquidate_breached=True
    )
    r = next(x for x in results if x.symbol == "AAA")
    assert r.breached is True
    assert r.liquidated is False
    assert broker._client.closed_positions == []  # type: ignore[attr-defined]
