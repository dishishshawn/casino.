"""Alpaca paper/live broker wrapper.

Thin facade over `alpaca-py`'s `TradingClient` exposing the minimal
surface `casino.execution.risk` and the cron jobs need. The wrapper:

* Hides Alpaca's request-object construction so callers don't import
  `alpaca.trading.requests` directly.
* Returns plain Python types (`Decimal` for prices/notionals, `int` for
  shares) — never raw SDK objects — so downstream code can be tested
  without an SDK dependency.
* Accepts an injected client/factory so tests can stub the SDK without
  hitting the network.

PRD §3 / §8: every position must have a broker-side stop. The
`submit_bracket_order` method is the *only* sanctioned way `risk.py`
talks to Alpaca; it always submits a bracket (entry + stop), never a
naked market order.

Money values are `Decimal` end-to-end (PRD §10, CLAUDE.md "money is
Decimal"). The Alpaca SDK takes/returns strings for prices, so we
convert at the boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from loguru import logger

from casino.config import get_config

OrderSide = Literal["buy", "sell"]
OrderStatus = Literal[
    "new",
    "accepted",
    "pending_new",
    "filled",
    "partially_filled",
    "canceled",
    "expired",
    "rejected",
    "done_for_day",
    "replaced",
    "pending_cancel",
    "pending_replace",
    "stopped",
    "suspended",
    "calculated",
    "held",
    "unknown",
]


# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class BrokerOrder:
    """Plain Python view of an Alpaca order. All money fields are Decimal."""

    id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    qty: int
    filled_qty: int
    status: OrderStatus
    order_type: str
    submitted_at: datetime | None
    filled_at: datetime | None
    filled_avg_price: Decimal | None
    stop_price: Decimal | None
    limit_price: Decimal | None
    legs: tuple[BrokerOrder, ...] = ()


@dataclass(frozen=True)
class BrokerPosition:
    """Plain Python view of an Alpaca position. Money fields are Decimal."""

    symbol: str
    qty: int
    side: Literal["long", "short"]
    avg_entry_price: Decimal
    market_price: Decimal
    market_value: Decimal
    unrealized_pl: Decimal
    cost_basis: Decimal


@dataclass(frozen=True)
class BrokerAccount:
    """Plain Python view of an Alpaca account. Money fields are Decimal."""

    account_number: str
    status: str
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    last_equity: Decimal
    pattern_day_trader: bool
    trading_blocked: bool


# ---------------------------------------------------------------------------- protocol


class _TradingClientLike(Protocol):
    """Subset of `alpaca.trading.client.TradingClient` we use.

    Declared as a Protocol so tests can hand in a stub without inheriting
    from the SDK class.
    """

    def submit_order(self, order_data: Any) -> Any: ...

    def cancel_orders(self) -> Any: ...

    def cancel_order_by_id(self, order_id: str) -> Any: ...

    def get_orders(self, filter: Any | None = ...) -> Any: ...

    def get_all_positions(self) -> Any: ...

    def close_all_positions(self, cancel_orders: bool = ...) -> Any: ...

    def close_position(self, symbol_or_asset_id: str, close_options: Any | None = ...) -> Any: ...

    def get_account(self) -> Any: ...

    def get_clock(self) -> Any: ...


# ---------------------------------------------------------------------------- conversion


def _to_decimal(v: Any) -> Decimal | None:
    """Best-effort conversion to Decimal. `None` / "" / "None" → None."""
    if v is None:
        return None
    if isinstance(v, Decimal):
        return v
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return Decimal(s)
    except Exception:  # noqa: BLE001 — alpaca returns mixed shapes
        return None


def _required_decimal(v: Any) -> Decimal:
    out = _to_decimal(v)
    if out is None:
        raise ValueError(f"required Decimal field was None or unparseable: {v!r}")
    return out


def _to_int_qty(v: Any) -> int:
    """Convert qty (string from Alpaca) to int. Whole-share strategies only."""
    if v is None:
        return 0
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return 0
    # Alpaca returns strings like "10" or "10.0".
    return int(Decimal(s))


def _convert_order(o: Any) -> BrokerOrder:
    legs = getattr(o, "legs", None) or ()
    return BrokerOrder(
        id=str(o.id),
        client_order_id=str(getattr(o, "client_order_id", "") or ""),
        symbol=str(o.symbol),
        side=_normalize_side(o.side),
        qty=_to_int_qty(getattr(o, "qty", 0)),
        filled_qty=_to_int_qty(getattr(o, "filled_qty", 0)),
        status=_normalize_status(getattr(o, "status", "unknown")),
        order_type=str(getattr(o, "order_type", "") or getattr(o, "type", "") or ""),
        submitted_at=getattr(o, "submitted_at", None),
        filled_at=getattr(o, "filled_at", None),
        filled_avg_price=_to_decimal(getattr(o, "filled_avg_price", None)),
        stop_price=_to_decimal(getattr(o, "stop_price", None)),
        limit_price=_to_decimal(getattr(o, "limit_price", None)),
        legs=tuple(_convert_order(leg) for leg in legs),
    )


def _convert_position(p: Any) -> BrokerPosition:
    raw_side = str(getattr(p, "side", "long") or "long").lower()
    side: Literal["long", "short"] = "short" if "short" in raw_side else "long"
    return BrokerPosition(
        symbol=str(p.symbol),
        qty=abs(_to_int_qty(getattr(p, "qty", 0))),
        side=side,
        avg_entry_price=_required_decimal(getattr(p, "avg_entry_price", "0")),
        market_price=_required_decimal(getattr(p, "current_price", "0")),
        market_value=_required_decimal(getattr(p, "market_value", "0")),
        unrealized_pl=_required_decimal(getattr(p, "unrealized_pl", "0")),
        cost_basis=_required_decimal(getattr(p, "cost_basis", "0")),
    )


def _convert_account(a: Any) -> BrokerAccount:
    return BrokerAccount(
        account_number=str(getattr(a, "account_number", "") or ""),
        status=str(getattr(a, "status", "") or ""),
        equity=_required_decimal(getattr(a, "equity", "0")),
        cash=_required_decimal(getattr(a, "cash", "0")),
        buying_power=_required_decimal(getattr(a, "buying_power", "0")),
        last_equity=_required_decimal(getattr(a, "last_equity", "0")),
        pattern_day_trader=bool(getattr(a, "pattern_day_trader", False)),
        trading_blocked=bool(getattr(a, "trading_blocked", False)),
    )


def _normalize_side(side: Any) -> OrderSide:
    s = str(getattr(side, "value", side)).lower()
    return "sell" if s == "sell" else "buy"


def _normalize_status(status: Any) -> OrderStatus:
    s = str(getattr(status, "value", status)).lower()
    valid: tuple[OrderStatus, ...] = (
        "new",
        "accepted",
        "pending_new",
        "filled",
        "partially_filled",
        "canceled",
        "expired",
        "rejected",
        "done_for_day",
        "replaced",
        "pending_cancel",
        "pending_replace",
        "stopped",
        "suspended",
        "calculated",
        "held",
    )
    return s if s in valid else "unknown"


# ---------------------------------------------------------------------------- broker


@dataclass
class AlpacaBroker:
    """Wrapper around alpaca-py's TradingClient.

    `client_factory` lets tests inject a stub. In production the default
    factory builds an `alpaca.trading.client.TradingClient` configured for
    paper trading via `config.alpaca_base_url` (PRD §3 — paper for v1,
    live cash account later).
    """

    api_key: str | None = None
    secret_key: str | None = None
    paper: bool = True
    client_factory: Callable[[str, str, bool], _TradingClientLike] | None = None
    _client: _TradingClientLike | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        cfg = get_config()
        if self.api_key is None:
            self.api_key = cfg.alpaca_api_key
        if self.secret_key is None:
            self.secret_key = cfg.alpaca_secret_key
        # Paper mode is the default (PRD §3 / CLAUDE.md hard rule 7).
        # The presence of "paper" in the configured URL is the signal.
        if "paper" not in cfg.alpaca_base_url.lower():
            logger.warning(
                "AlpacaBroker: configured base URL {} does not look like paper; "
                "ensure live trading is intended",
                cfg.alpaca_base_url,
            )

    # -------------------------------------------------------------- client
    def _ensure_client(self) -> _TradingClientLike:
        if self._client is not None:
            return self._client
        if self.client_factory is not None:
            self._client = self.client_factory(
                self.api_key or "",
                self.secret_key or "",
                self.paper,
            )
            return self._client
        # Default factory: lazy import so tests don't need the SDK installed.
        from alpaca.trading.client import TradingClient  # noqa: PLC0415

        self._client = TradingClient(
            api_key=self.api_key or "",
            secret_key=self.secret_key or "",
            paper=self.paper,
        )
        return self._client

    def set_client(self, client: _TradingClientLike) -> None:
        """Inject a client directly (used by tests + `kill_switch`)."""
        self._client = client

    # -------------------------------------------------------------- orders
    def submit_bracket_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: OrderSide,
        stop_price: Decimal,
        take_profit_price: Decimal | None = None,
        time_in_force: str = "day",
        client_order_id: str | None = None,
    ) -> BrokerOrder:
        """Submit a bracket order: entry + broker-side stop loss.

        CLAUDE.md hard rule 3 / PRD §8: every position must have a
        broker-side stop. This method enforces that — `stop_price` is
        required, and the resulting order has a `stop_loss` leg.

        For now we submit market orders for the entry (the v1 strategy
        is daily/EOD; intraday limit logic lives in a later phase).
        """
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if stop_price <= Decimal("0"):
            raise ValueError(f"stop_price must be positive, got {stop_price}")

        from alpaca.trading.enums import (  # noqa: PLC0415
            OrderClass,
            TimeInForce,
        )
        from alpaca.trading.enums import (
            OrderSide as SDKOrderSide,
        )
        from alpaca.trading.requests import (  # noqa: PLC0415
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        sdk_side = SDKOrderSide.BUY if side == "buy" else SDKOrderSide.SELL
        try:
            tif = TimeInForce(time_in_force.lower())
        except ValueError as e:
            raise ValueError(f"unknown time_in_force {time_in_force!r}") from e

        stop_loss = StopLossRequest(stop_price=str(stop_price))
        take_profit = (
            TakeProfitRequest(limit_price=str(take_profit_price))
            if take_profit_price is not None
            else None
        )
        order_class = OrderClass.BRACKET if take_profit is not None else OrderClass.OTO
        req = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=sdk_side,
            time_in_force=tif,
            order_class=order_class,
            stop_loss=stop_loss,
            take_profit=take_profit,
            client_order_id=client_order_id,
        )
        client = self._ensure_client()
        raw = client.submit_order(order_data=req)
        order = _convert_order(raw)
        logger.info(
            "alpaca: submitted {} {} {} @ market with stop {} (id={})",
            side,
            qty,
            symbol,
            stop_price,
            order.id,
        )
        return order

    def cancel_all(self) -> int:
        """Cancel every open order. Returns the count cancelled (best-effort)."""
        client = self._ensure_client()
        result = client.cancel_orders()
        try:
            return len(list(result))
        except TypeError:
            return 0

    def cancel_order(self, order_id: str) -> None:
        """Cancel a single open order by id."""
        client = self._ensure_client()
        client.cancel_order_by_id(order_id)

    def close_all_positions(self, *, cancel_orders: bool = True) -> list[BrokerOrder]:
        """Submit market-close for every open position.

        Used by the kill switch (`risk.kill_switch`). Returns the order
        objects produced by Alpaca.
        """
        client = self._ensure_client()
        raw_list = client.close_all_positions(cancel_orders=cancel_orders)
        out: list[BrokerOrder] = []
        for r in raw_list or []:
            # close_all_positions returns response wrappers — try to coerce.
            body = getattr(r, "body", None) or r
            try:
                out.append(_convert_order(body))
            except Exception:  # noqa: BLE001
                logger.warning("alpaca: could not convert close response {!r}", r)
        return out

    def close_position(self, symbol: str) -> BrokerOrder:
        """Submit a market-close for one position by symbol."""
        client = self._ensure_client()
        raw = client.close_position(symbol_or_asset_id=symbol.upper())
        return _convert_order(raw)

    def get_orders(self, *, status: str = "open") -> list[BrokerOrder]:
        """Return open (default) or all orders as `BrokerOrder` objects."""
        from alpaca.trading.enums import QueryOrderStatus  # noqa: PLC0415
        from alpaca.trading.requests import GetOrdersRequest  # noqa: PLC0415

        client = self._ensure_client()
        try:
            qstat = QueryOrderStatus(status.lower())
        except ValueError as e:
            raise ValueError(f"unknown order status {status!r}") from e
        req = GetOrdersRequest(status=qstat)
        raw = client.get_orders(filter=req)
        return [_convert_order(o) for o in (raw or [])]

    def get_positions(self) -> list[BrokerPosition]:
        """Return all open positions as `BrokerPosition` objects."""
        client = self._ensure_client()
        raw = client.get_all_positions()
        return [_convert_position(p) for p in (raw or [])]

    def get_account(self) -> BrokerAccount:
        """Return account-level NAV, cash, buying power."""
        client = self._ensure_client()
        return _convert_account(client.get_account())

    # -------------------------------------------------------------- market data
    def is_market_open(self) -> bool:
        """Return True iff the equity market is currently open per Alpaca clock."""
        client = self._ensure_client()
        clock = client.get_clock()
        return bool(getattr(clock, "is_open", False))


# ---------------------------------------------------------------------------- factory


def build_default_broker() -> AlpacaBroker:
    """Construct an AlpacaBroker from the current `casino.config` singleton.

    Convenience factory used by jobs and the kill-switch CLI. Does not
    open a network connection until the first call — `_ensure_client` is
    lazy.
    """
    cfg = get_config()
    return AlpacaBroker(
        api_key=cfg.alpaca_api_key or None,
        secret_key=cfg.alpaca_secret_key or None,
        paper="paper" in cfg.alpaca_base_url.lower(),
    )
