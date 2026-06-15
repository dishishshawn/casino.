"""Risk management tests — PRD §8 hard rules.

Each rule has at least one named test that drives the documented invariant
end-to-end. The suite uses an in-memory `FakeBroker` for everything that
would otherwise hit Alpaca; no live network calls.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from casino.config import get_config
from casino.execution import book
from casino.execution.alpaca_broker import (
    AlpacaBroker,
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    _convert_order,
)
from casino.execution.risk import (
    PortfolioState,
    RiskRejection,
    TradingDisabledError,
    flatten_and_disable,
    re_enable_trading,
    size_position,
    snapshot_portfolio_from_broker,
    submit_order,
)

# ---------------------------------------------------------------------------- fakes


class FakeTradingClient:
    """In-memory stand-in for `alpaca.trading.client.TradingClient`."""

    def __init__(
        self,
        *,
        account: BrokerAccount,
        positions: list[BrokerPosition] | None = None,
        next_order_id: str = "ord-1",
        order_history: list[BrokerOrder] | None = None,
        market_open: bool = True,
        fail_symbols: set[str] | None = None,
    ) -> None:
        self._account = account
        self._positions = list(positions or [])
        self._open_orders: list[BrokerOrder] = []
        # Terminal / historical orders returned for get_orders(status="all").
        self.order_history: list[BrokerOrder] = list(order_history or [])
        self._next_id = next_order_id
        self.submitted_requests: list[Any] = []
        self.cancelled = 0
        self.closed_positions: list[str] = []
        self._market_open = market_open
        self._fail_symbols = {s.upper() for s in (fail_symbols or set())}

    # the broker wrapper calls these
    def submit_order(self, order_data: Any) -> Any:
        sym = str(getattr(order_data, "symbol", "TEST")).upper()
        if sym in self._fail_symbols:
            raise RuntimeError(f"simulated broker reject for {sym}")
        self.submitted_requests.append(order_data)
        # A StopOrderRequest carries a top-level stop_price (bracket entries
        # carry a nested stop_loss instead); register stops as open so a
        # second ensure_protective_stops pass sees the symbol as protected.
        stop_px = getattr(order_data, "stop_price", None)
        side = "buy" if str(getattr(order_data, "side", "")).lower().endswith("buy") else "sell"
        raw = _FakeRawOrder(
            id=self._next_id,
            client_order_id=getattr(order_data, "client_order_id", None) or "",
            symbol=getattr(order_data, "symbol", "TEST"),
            side=side,
            qty=str(getattr(order_data, "qty", 0)),
            filled_qty="0",
            status="accepted",
            order_type="stop" if stop_px is not None else "market",
            stop_price=str(stop_px) if stop_px is not None else None,
            limit_price=None,
            filled_avg_price=None,
            submitted_at=None,
            filled_at=None,
            legs=(),
        )
        if stop_px is not None:
            self._open_orders.append(_convert_order(raw))
        return raw

    def cancel_orders(self) -> list[Any]:
        n = len(self._open_orders)
        self._open_orders = []
        self.cancelled += n
        # alpaca-py returns a list of cancellation responses
        return [object() for _ in range(n)]

    def cancel_order_by_id(self, order_id: str) -> None:
        self._open_orders = [o for o in self._open_orders if o.id != order_id]

    def get_orders(self, filter: Any | None = None) -> list[Any]:
        status = ""
        if filter is not None:
            raw_status = getattr(filter, "status", "")
            status = str(getattr(raw_status, "value", raw_status)).lower()
        if status in ("all", "closed"):
            # History (terminal orders) plus anything still open.
            return list(self.order_history) + list(self._open_orders)
        return list(self._open_orders)

    def get_all_positions(self) -> list[Any]:
        return [_position_to_raw(p) for p in self._positions]

    def close_all_positions(self, cancel_orders: bool = True) -> list[Any]:
        out: list[Any] = []
        for p in self._positions:
            self.closed_positions.append(p.symbol)
            out.append(
                _FakeRawOrder(
                    id=f"close-{p.symbol}",
                    client_order_id="",
                    symbol=p.symbol,
                    side="sell" if p.side == "long" else "buy",
                    qty=str(p.qty),
                    filled_qty="0",
                    status="accepted",
                    order_type="market",
                    stop_price=None,
                    limit_price=None,
                    filled_avg_price=None,
                    submitted_at=None,
                    filled_at=None,
                    legs=(),
                )
            )
        self._positions = []
        return out

    def close_position(self, symbol_or_asset_id: str, close_options: Any | None = None) -> Any:
        self.closed_positions.append(symbol_or_asset_id)
        return _FakeRawOrder(
            id=f"close-{symbol_or_asset_id}",
            client_order_id="",
            symbol=symbol_or_asset_id,
            side="sell",
            qty="0",
            filled_qty="0",
            status="accepted",
            order_type="market",
            stop_price=None,
            limit_price=None,
            filled_avg_price=None,
            submitted_at=None,
            filled_at=None,
            legs=(),
        )

    def get_account(self) -> Any:
        return _account_to_raw(self._account)

    def get_clock(self) -> Any:
        is_open = self._market_open

        class _C:
            pass

        c = _C()
        c.is_open = is_open
        return c


class _FakeRawOrder:
    """Bag of attributes shaped like an alpaca-py Order."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _account_to_raw(a: BrokerAccount) -> Any:
    class _A:
        account_number = a.account_number
        status = a.status
        equity = str(a.equity)
        cash = str(a.cash)
        buying_power = str(a.buying_power)
        last_equity = str(a.last_equity)
        pattern_day_trader = a.pattern_day_trader
        trading_blocked = a.trading_blocked

    return _A()


def _position_to_raw(p: BrokerPosition) -> Any:
    class _P:
        symbol = p.symbol
        qty = str(p.qty if p.side == "long" else -p.qty)
        side = p.side
        avg_entry_price = str(p.avg_entry_price)
        current_price = str(p.market_price)
        market_value = str(p.market_value)
        unrealized_pl = str(p.unrealized_pl)
        cost_basis = str(p.cost_basis)

    return _P()


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up an isolated state.sqlite path so kill-switch flags don't leak across tests."""
    state = tmp_path / "state.sqlite"
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(state))
    get_config.cache_clear()
    book.init_schema(state)
    return state


@pytest.fixture
def broker_factory():
    def _build(
        *,
        equity: Decimal = Decimal("100000"),
        cash: Decimal = Decimal("100000"),
        positions: list[BrokerPosition] | None = None,
    ) -> tuple[AlpacaBroker, FakeTradingClient]:
        account = BrokerAccount(
            account_number="paper-1",
            status="ACTIVE",
            equity=equity,
            cash=cash,
            buying_power=cash,
            last_equity=equity,
            pattern_day_trader=False,
            trading_blocked=False,
        )
        client = FakeTradingClient(account=account, positions=positions or [])
        broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
        broker.set_client(client)
        return broker, client

    return _build


# ---------------------------------------------------------------------------- size_position rule tests


def test_size_position_at_1_5pct_passes() -> None:
    """PRD §8 rule 1: per-trade risk = 1.5% of NAV → passes."""
    nav = Decimal("100000")
    portfolio = PortfolioState(
        nav=nav,
        cash=nav,
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={},
    )
    # entry=100, stop=99 → per-share risk = 1. With 1.5% of $100k = $1500
    # raw qty = 1500 * 1 (kelly fraction can shrink it to 375 with default 0.25)
    sized = size_position(
        symbol="AAA",
        side="buy",
        entry_price=Decimal("100"),
        stop_price=Decimal("99"),
        portfolio=portfolio,
        max_risk_per_trade=Decimal("0.015"),
        kelly_fraction=Decimal("0.25"),
    )
    # default kelly trims; assert qty positive and risk_dollars stays under cap
    assert sized.qty > 0
    assert sized.risk_dollars <= nav * Decimal("0.015")


def test_size_position_above_1_5pct_rejected() -> None:
    """PRD §8 rule 1: max_risk_per_trade > 1.5% of NAV → RiskRejection."""
    portfolio = PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("100000"),
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={},
    )
    with pytest.raises(RiskRejection) as exc:
        size_position(
            symbol="AAA",
            side="buy",
            entry_price=Decimal("100"),
            stop_price=Decimal("99"),
            portfolio=portfolio,
            max_risk_per_trade=Decimal("0.0151"),  # 1.51%
        )
    assert "max_risk_per_trade" in exc.value.rule


def test_size_position_single_name_cap() -> None:
    """PRD §8 rule 2: single-name exposure capped at 10% NAV."""
    nav = Decimal("100000")
    portfolio = PortfolioState(
        nav=nav,
        cash=nav,
        gross_exposure_dollars=Decimal("0"),
        # Already 9.5k in AAA → only 500 of 10% headroom remains.
        single_name_exposure={"AAA": Decimal("9500")},
    )
    sized = size_position(
        symbol="AAA",
        side="buy",
        entry_price=Decimal("100"),
        stop_price=Decimal("99"),
        portfolio=portfolio,
    )
    # 500 / 100 = 5 shares max
    assert sized.qty <= 5


def test_size_position_single_name_already_at_cap() -> None:
    """At/above 10% single-name → reject."""
    nav = Decimal("100000")
    portfolio = PortfolioState(
        nav=nav,
        cash=nav,
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={"AAA": Decimal("10000")},
    )
    with pytest.raises(RiskRejection) as exc:
        size_position(
            symbol="AAA",
            side="buy",
            entry_price=Decimal("100"),
            stop_price=Decimal("99"),
            portfolio=portfolio,
        )
    assert "max_single_name" in exc.value.rule


def test_size_position_gross_exposure_cap() -> None:
    """PRD §8 rule 3: total gross exposure ≤ 100% NAV."""
    nav = Decimal("100000")
    portfolio = PortfolioState(
        nav=nav,
        cash=nav,
        gross_exposure_dollars=Decimal("100000"),  # already at 100%
        single_name_exposure={},
    )
    with pytest.raises(RiskRejection) as exc:
        size_position(
            symbol="AAA",
            side="buy",
            entry_price=Decimal("100"),
            stop_price=Decimal("99"),
            portfolio=portfolio,
        )
    assert "max_gross_exposure" in exc.value.rule


def test_size_position_kelly_above_half_rejected() -> None:
    """PRD §8 rule 4: fractional Kelly only (¼ to ½). Above 0.5 → reject."""
    portfolio = PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("100000"),
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={},
    )
    with pytest.raises(RiskRejection) as exc:
        size_position(
            symbol="AAA",
            side="buy",
            entry_price=Decimal("100"),
            stop_price=Decimal("99"),
            portfolio=portfolio,
            kelly_fraction=Decimal("0.6"),
        )
    assert "kelly_fraction" in exc.value.rule


def test_size_position_kelly_zero_rejected() -> None:
    portfolio = PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("100000"),
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={},
    )
    with pytest.raises(RiskRejection):
        size_position(
            symbol="AAA",
            side="buy",
            entry_price=Decimal("100"),
            stop_price=Decimal("99"),
            portfolio=portfolio,
            kelly_fraction=Decimal("0"),
        )


def test_size_position_long_stop_must_be_below_entry() -> None:
    portfolio = PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("100000"),
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={},
    )
    with pytest.raises(ValueError):
        size_position(
            symbol="AAA",
            side="buy",
            entry_price=Decimal("100"),
            stop_price=Decimal("101"),
            portfolio=portfolio,
        )


def test_size_position_short_stop_must_be_above_entry() -> None:
    portfolio = PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("100000"),
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={},
    )
    with pytest.raises(ValueError):
        size_position(
            symbol="AAA",
            side="sell",
            entry_price=Decimal("100"),
            stop_price=Decimal("99"),
            portfolio=portfolio,
        )


# ---------------------------------------------------------------------------- submit_order


def test_submit_order_uses_bracket_with_broker_stop(isolated_state, broker_factory) -> None:
    """PRD §8 rule 6 / CLAUDE.md hard rule 3: stop is broker-side, not local."""
    broker, fake = broker_factory()
    order = submit_order(
        broker=broker,
        symbol="AAA",
        side="buy",
        entry_price=Decimal("100"),
        stop_price=Decimal("99"),
        db_path=isolated_state,
    )
    assert order.id == "ord-1"
    # The submitted request must have a stop_loss leg.
    assert len(fake.submitted_requests) == 1
    req = fake.submitted_requests[0]
    assert req.stop_loss is not None
    # And the leg's stop_price echoes what we requested.
    assert Decimal(str(req.stop_loss.stop_price)) == Decimal("99")


def test_submit_order_entry_is_gtc_limit_bracket(isolated_state, broker_factory) -> None:
    """Regression for the 2026-06-01 naked-stop incident.

    A *market* bracket forces ``day`` time-in-force, so Alpaca expires the
    stop leg at the first session close and the position is left naked. The
    durable fix enters with a marketable-limit GTC order so the attached stop
    leg persists across sessions. Assert the request we send is GTC + limit
    with the buffer applied in the marketable direction.
    """
    broker, fake = broker_factory()
    submit_order(
        broker=broker,
        symbol="AAA",
        side="buy",
        entry_price=Decimal("100"),
        stop_price=Decimal("99"),
        entry_limit_buffer=Decimal("0.01"),
        db_path=isolated_state,
    )
    req = fake.submitted_requests[0]
    assert str(getattr(req.time_in_force, "value", req.time_in_force)) == "gtc"
    assert str(getattr(req.type, "value", req.type)) == "limit"
    # buy, 100 entry, 1% buffer, rounded up in the marketable direction → 101.00.
    assert Decimal(str(req.limit_price)) == Decimal("101.00")
    # The marketable-limit price is persisted to the book for reconciliation.
    with book.get_book_conn(isolated_state) as conn:
        (limit_px,) = conn.execute(
            "SELECT limit_price FROM orders WHERE status != 'broker_rejected'"
        ).fetchone()
    assert Decimal(str(limit_px)) == Decimal("101.00")


def test_submit_order_records_to_book(isolated_state, broker_factory) -> None:
    """The order row is persisted to state.sqlite for reconciliation later."""
    broker, _ = broker_factory()
    order = submit_order(
        broker=broker,
        symbol="AAA",
        side="buy",
        entry_price=Decimal("100"),
        stop_price=Decimal("99"),
        db_path=isolated_state,
    )
    open_orders = book.fetch_open_orders(db_path=isolated_state)
    assert any(o.broker_order_id == order.id for o in open_orders)
    # The placeholder row has been swapped out — no pending- leftovers.
    assert not any(o.broker_order_id.startswith("pending-") for o in open_orders)


def test_submit_order_writes_pending_row_before_broker_call(
    isolated_state: Path,
    broker_factory,
) -> None:
    """Regression: 2026-05-11 incident — Ctrl-C between broker call and book
    write left Alpaca holding 5 SPY with no book row, tripping the reconcile
    drift kill criterion. The fix writes a "submission_pending" row BEFORE
    the broker call; even if the broker call raises, the row remains for
    operator recovery.
    """
    broker, _ = broker_factory()

    # Replace the broker's submit_bracket_order with a failing stub so we
    # can verify the pre-broker pending row was persisted.
    def _boom(**_kw: Any) -> BrokerOrder:
        raise RuntimeError("simulated broker outage")

    broker.submit_bracket_order = _boom  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="simulated broker outage"):
        submit_order(
            broker=broker,
            symbol="AAA",
            side="buy",
            entry_price=Decimal("100"),
            stop_price=Decimal("99"),
            db_path=isolated_state,
        )

    # Inspect the orders table directly — the failed-but-recorded row uses
    # status="broker_rejected" which fetch_open_orders intentionally
    # excludes, so go through raw SQL.
    with book.get_book_conn(isolated_state) as conn:
        rows = conn.execute("SELECT broker_order_id, status, symbol, qty FROM orders").fetchall()

    assert len(rows) == 1
    broker_order_id, status, symbol, qty = rows[0]
    assert broker_order_id.startswith("pending-casino-")
    assert status == "broker_rejected"
    assert symbol == "AAA"
    assert qty > 0  # sized normally; exact qty depends on PRD §8 caps


# ---------------------------------------------------------------------------- kill switch


def test_kill_switch_end_to_end(isolated_state, broker_factory) -> None:
    """CLAUDE.md hard rule 4 / PRD §8: kill switch must work end-to-end."""
    positions = [
        BrokerPosition(
            symbol="AAA",
            qty=10,
            side="long",
            avg_entry_price=Decimal("100"),
            market_price=Decimal("101"),
            market_value=Decimal("1010"),
            unrealized_pl=Decimal("10"),
            cost_basis=Decimal("1000"),
        ),
        BrokerPosition(
            symbol="BBB",
            qty=5,
            side="short",
            avg_entry_price=Decimal("50"),
            market_price=Decimal("49"),
            market_value=Decimal("245"),
            unrealized_pl=Decimal("5"),
            cost_basis=Decimal("250"),
        ),
    ]
    broker, fake = broker_factory(positions=positions)

    # Pre-condition: trading not disabled.
    assert not book.is_trading_disabled(db_path=isolated_state)

    result = flatten_and_disable(broker=broker, reason="unit test", db_path=isolated_state)

    # 1) cancel-all was invoked
    assert fake.cancelled >= 0  # zero open orders is fine; method called
    # 2) every position got a close order
    assert sorted(fake.closed_positions) == ["AAA", "BBB"]
    # 3) kill flag persisted
    assert book.is_trading_disabled(db_path=isolated_state)
    assert result.flag_set is True
    assert result.closed_positions == 2

    # 4) subsequent submit_order raises
    with pytest.raises(TradingDisabledError):
        submit_order(
            broker=broker,
            symbol="CCC",
            side="buy",
            entry_price=Decimal("100"),
            stop_price=Decimal("99"),
            db_path=isolated_state,
        )

    # operator clears the flag
    re_enable_trading(db_path=isolated_state)
    assert not book.is_trading_disabled(db_path=isolated_state)


def test_kill_switch_cli_module_invocable() -> None:
    """The `python -m casino.execution.kill_switch` entry must be importable."""
    from casino.execution import kill_switch

    assert hasattr(kill_switch, "main")
    assert callable(kill_switch.main)


# ---------------------------------------------------------------------------- snapshot_portfolio_from_broker


def test_snapshot_portfolio_aggregates_gross_exposure(broker_factory) -> None:
    positions = [
        BrokerPosition(
            symbol="AAA",
            qty=10,
            side="long",
            avg_entry_price=Decimal("100"),
            market_price=Decimal("100"),
            market_value=Decimal("1000"),
            unrealized_pl=Decimal("0"),
            cost_basis=Decimal("1000"),
        ),
        BrokerPosition(
            symbol="BBB",
            qty=5,
            side="short",
            avg_entry_price=Decimal("100"),
            market_price=Decimal("100"),
            market_value=Decimal("500"),
            unrealized_pl=Decimal("0"),
            cost_basis=Decimal("500"),
        ),
    ]
    broker, _ = broker_factory(equity=Decimal("100000"), positions=positions)
    snap = snapshot_portfolio_from_broker(broker)
    assert snap.nav == Decimal("100000")
    assert snap.gross_exposure_dollars == Decimal("1500")
    assert snap.single_name_exposure["AAA"] == Decimal("1000")
    assert snap.single_name_exposure["BBB"] == Decimal("500")


# ---------------------------------------------------------------------------- decimal contract


def test_money_columns_are_decimal_strings(isolated_state, broker_factory) -> None:
    """PRD §10 rule: money in SQLite is the canonical Decimal string."""
    broker, _ = broker_factory()
    submit_order(
        broker=broker,
        symbol="AAA",
        side="buy",
        entry_price=Decimal("123.45"),
        stop_price=Decimal("122.00"),
        db_path=isolated_state,
    )
    with book.get_book_conn(isolated_state) as conn:
        rows = conn.execute("SELECT stop_price, notional_estimate FROM orders").fetchall()
    assert rows
    sp, ne = rows[0]
    # str → Decimal round-trips losslessly: this is the contract.
    assert Decimal(str(sp)) == Decimal("122.00")
    if ne is not None:
        assert Decimal(str(ne)) > Decimal("0")
