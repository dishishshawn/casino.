"""Tests for casino.execution.alpaca_broker."""

from __future__ import annotations

from decimal import Decimal

import pytest

from casino.execution.alpaca_broker import (
    AlpacaBroker,
    BrokerAccount,
    BrokerPosition,
)
from tests.test_risk import FakeTradingClient


def _broker(
    *,
    equity: Decimal = Decimal("100000"),
    positions: list[BrokerPosition] | None = None,
) -> tuple[AlpacaBroker, FakeTradingClient]:
    account = BrokerAccount(
        account_number="paper-1",
        status="ACTIVE",
        equity=equity,
        cash=equity,
        buying_power=equity,
        last_equity=equity,
        pattern_day_trader=False,
        trading_blocked=False,
    )
    fake = FakeTradingClient(account=account, positions=positions or [])
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    broker.set_client(fake)
    return broker, fake


def test_submit_bracket_order_includes_stop_loss() -> None:
    broker, fake = _broker()
    order = broker.submit_bracket_order(
        symbol="AAA",
        qty=10,
        side="buy",
        stop_price=Decimal("99"),
    )
    assert order.id == "ord-1"
    req = fake.submitted_requests[0]
    assert req.stop_loss is not None
    assert Decimal(str(req.stop_loss.stop_price)) == Decimal("99")


def test_submit_bracket_order_with_take_profit() -> None:
    broker, fake = _broker()
    broker.submit_bracket_order(
        symbol="AAA",
        qty=10,
        side="buy",
        stop_price=Decimal("99"),
        take_profit_price=Decimal("110"),
    )
    req = fake.submitted_requests[0]
    assert req.stop_loss is not None
    assert req.take_profit is not None


def test_submit_bracket_order_rejects_zero_qty() -> None:
    broker, _ = _broker()
    with pytest.raises(ValueError):
        broker.submit_bracket_order(
            symbol="AAA",
            qty=0,
            side="buy",
            stop_price=Decimal("99"),
        )


def test_submit_bracket_order_rejects_zero_stop() -> None:
    broker, _ = _broker()
    with pytest.raises(ValueError):
        broker.submit_bracket_order(
            symbol="AAA",
            qty=10,
            side="buy",
            stop_price=Decimal("0"),
        )


def test_get_account_returns_decimal_money() -> None:
    broker, _ = _broker(equity=Decimal("12345.67"))
    account = broker.get_account()
    assert isinstance(account.equity, Decimal)
    assert account.equity == Decimal("12345.67")


def test_get_positions_returns_decimal_market_value() -> None:
    pos = BrokerPosition(
        symbol="AAA",
        qty=10,
        side="long",
        avg_entry_price=Decimal("100"),
        market_price=Decimal("101"),
        market_value=Decimal("1010"),
        unrealized_pl=Decimal("10"),
        cost_basis=Decimal("1000"),
    )
    broker, _ = _broker(positions=[pos])
    out = broker.get_positions()
    assert len(out) == 1
    assert isinstance(out[0].market_value, Decimal)


def test_is_market_open_true_in_fake() -> None:
    broker, _ = _broker()
    assert broker.is_market_open() is True


def test_close_all_positions_returns_orders() -> None:
    pos = BrokerPosition(
        symbol="AAA",
        qty=10,
        side="long",
        avg_entry_price=Decimal("100"),
        market_price=Decimal("100"),
        market_value=Decimal("1000"),
        unrealized_pl=Decimal("0"),
        cost_basis=Decimal("1000"),
    )
    broker, fake = _broker(positions=[pos])
    closed = broker.close_all_positions()
    assert len(closed) == 1
    assert "AAA" in fake.closed_positions
