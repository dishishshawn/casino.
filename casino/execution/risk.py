"""Position sizing, stops, and the kill switch.

Authoritative for sizing and exposure caps. Brokers and signals are *inputs*
to risk; risk is not an advisory layer downstream of them.

PRD §8 / CLAUDE.md hard rules — enforced here on every order:
1. Per-trade risk ≤ 1.5% of NAV.
2. Max single-name exposure ≤ 10% of NAV.
3. Total gross exposure ≤ 100% of NAV.
4. Fractional Kelly only (¼ to ½). Never full Kelly.
5. Cash account, not margin.
6. Broker-side stop-loss on every position.
7. Kill switch is a single command and is testable.

Money is `Decimal` end-to-end. Quantity is `int` (whole-share strategy).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Literal

from loguru import logger

from casino.config import get_config
from casino.execution import book
from casino.execution.alpaca_broker import (
    AlpacaBroker,
    BrokerOrder,
    BrokerPosition,
)

OrderSide = Literal["buy", "sell"]


# ---------------------------------------------------------------------------- errors


class RiskRejection(RuntimeError):
    """Raised when an order would violate a hard risk rule.

    The error message identifies the specific PRD §8 rule. Tests assert
    on the rule name.
    """

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail


class TradingDisabledError(RuntimeError):
    """Raised when the kill switch flag is set and an order is attempted."""


# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class SizedOrder:
    """Output of `size_position`: a pre-flight check that fits within all caps.

    `qty` is the share count to send. `notional` is `qty * entry_price`,
    `risk_dollars` is `qty * abs(entry_price - stop_price)` — i.e. the
    capital at risk if the stop is hit.
    """

    symbol: str
    side: OrderSide
    qty: int
    entry_price: Decimal
    stop_price: Decimal
    notional: Decimal
    risk_dollars: Decimal
    nav_used: Decimal


@dataclass(frozen=True)
class PortfolioState:
    """A snapshot of the portfolio used for exposure checks.

    `nav` is account equity. `gross_exposure_dollars` is the sum of
    `|qty * market_price|` over open positions. `single_name_exposure` is
    a per-symbol notional, used for the 10% cap when adding to or
    re-entering a name.
    """

    nav: Decimal
    cash: Decimal
    gross_exposure_dollars: Decimal
    single_name_exposure: dict[str, Decimal]


# ---------------------------------------------------------------------------- helpers


def _zero() -> Decimal:
    return Decimal("0")


def _abs(d: Decimal) -> Decimal:
    return -d if d < _zero() else d


def _to_decimal(v: Decimal | int | float | str) -> Decimal:
    if isinstance(v, Decimal):
        return v
    if isinstance(v, int):
        return Decimal(v)
    return Decimal(str(v))


def snapshot_portfolio_from_broker(broker: AlpacaBroker) -> PortfolioState:
    """Build a `PortfolioState` from broker account + positions."""
    account = broker.get_account()
    positions: list[BrokerPosition] = broker.get_positions()
    gross = _zero()
    by_name: dict[str, Decimal] = {}
    for p in positions:
        notional = _abs(p.market_value)
        gross += notional
        by_name[p.symbol.upper()] = by_name.get(p.symbol.upper(), _zero()) + notional
    return PortfolioState(
        nav=account.equity,
        cash=account.cash,
        gross_exposure_dollars=gross,
        single_name_exposure=by_name,
    )


# ---------------------------------------------------------------------------- sizing


def size_position(
    *,
    symbol: str,
    side: OrderSide,
    entry_price: Decimal,
    stop_price: Decimal,
    portfolio: PortfolioState,
    max_risk_per_trade: Decimal | float | None = None,
    max_single_name: Decimal | float | None = None,
    max_gross_exposure: Decimal | float | None = None,
    kelly_fraction: Decimal | float | None = None,
) -> SizedOrder:
    """Compute the share quantity for a new position, enforcing all caps.

    Algorithm (PRD §8):

    1. Risk-budget sizing: ``risk_dollars_cap = nav * max_risk_per_trade``.
       ``per_share_risk = |entry - stop|``. Raw qty = ``risk_dollars_cap / per_share_risk``.
    2. Apply fractional Kelly: ``qty *= kelly_fraction``.
    3. Cap by single-name notional: ``qty * entry ≤ nav * max_single_name``
       (taking into account existing single-name exposure).
    4. Cap by gross exposure: existing gross + this notional ≤ nav * 100%.
    5. Cap by available cash (rule 5: cash account, no margin).
    6. Floor to whole shares (round down).

    If any binding cap reduces qty to 0, raise `RiskRejection` so the caller
    can log + alert rather than silently submitting nothing.
    """
    cfg = get_config()
    rt = _to_decimal(
        max_risk_per_trade if max_risk_per_trade is not None else cfg.max_risk_per_trade
    )
    sn = _to_decimal(max_single_name if max_single_name is not None else cfg.max_single_name)
    ge = _to_decimal(
        max_gross_exposure if max_gross_exposure is not None else cfg.max_gross_exposure
    )
    kf = _to_decimal(kelly_fraction if kelly_fraction is not None else cfg.kelly_fraction)

    # ---- invariant guards (configuration sanity) ----
    if rt > Decimal("0.015") + Decimal("0.000000001"):
        raise RiskRejection(
            "PRD §8.1 max_risk_per_trade",
            f"max_risk_per_trade {rt} exceeds 1.5% of NAV cap",
        )
    if sn > Decimal("0.10") + Decimal("0.000000001"):
        raise RiskRejection(
            "PRD §8.2 max_single_name",
            f"max_single_name {sn} exceeds 10% of NAV cap",
        )
    if ge > Decimal("1.0") + Decimal("0.000000001"):
        raise RiskRejection(
            "PRD §8.3 max_gross_exposure",
            f"max_gross_exposure {ge} exceeds 100% of NAV cap",
        )
    if kf <= _zero() or kf > Decimal("0.5"):
        raise RiskRejection(
            "PRD §8.4 kelly_fraction",
            f"kelly_fraction {kf} not in (0, 0.5]; full Kelly is forbidden",
        )

    # ---- input validation ----
    entry = _to_decimal(entry_price)
    stop = _to_decimal(stop_price)
    if entry <= _zero():
        raise ValueError(f"entry_price must be positive, got {entry}")
    if stop <= _zero():
        raise ValueError(f"stop_price must be positive, got {stop}")
    if portfolio.nav <= _zero():
        raise RiskRejection(
            "PRD §8 NAV",
            f"NAV is non-positive ({portfolio.nav}); refuse to size new orders",
        )
    if side == "buy" and stop >= entry:
        raise ValueError(f"long stop must be below entry; got stop={stop} entry={entry}")
    if side == "sell" and stop <= entry:
        raise ValueError(f"short stop must be above entry; got stop={stop} entry={entry}")

    per_share_risk = _abs(entry - stop)
    if per_share_risk == _zero():
        raise ValueError("per-share risk is zero (entry == stop); cannot size")

    nav = portfolio.nav

    # 1) risk-budget cap
    risk_dollars_cap = nav * rt
    raw_qty_decimal = risk_dollars_cap / per_share_risk

    # 2) fractional Kelly
    raw_qty_decimal = raw_qty_decimal * kf

    # 3) single-name cap
    sym = symbol.upper()
    existing_single = portfolio.single_name_exposure.get(sym, _zero())
    single_name_dollars_cap = nav * sn - existing_single
    if single_name_dollars_cap <= _zero():
        raise RiskRejection(
            "PRD §8.2 max_single_name",
            f"existing exposure to {sym} ({existing_single}) already at/above 10% NAV cap",
        )
    qty_by_single = single_name_dollars_cap / entry
    raw_qty_decimal = min(raw_qty_decimal, qty_by_single)

    # 4) gross exposure cap
    gross_dollars_cap = nav * ge - portfolio.gross_exposure_dollars
    if gross_dollars_cap <= _zero():
        raise RiskRejection(
            "PRD §8.3 max_gross_exposure",
            f"existing gross exposure ({portfolio.gross_exposure_dollars}) "
            f"already at/above {ge:.0%} NAV cap",
        )
    qty_by_gross = gross_dollars_cap / entry
    raw_qty_decimal = min(raw_qty_decimal, qty_by_gross)

    # 5) cash cap (no margin per PRD §8 rule 5)
    if portfolio.cash > _zero():
        qty_by_cash = portfolio.cash / entry
        raw_qty_decimal = min(raw_qty_decimal, qty_by_cash)

    # 6) floor to whole shares
    qty_int = int(raw_qty_decimal.to_integral_value(rounding=ROUND_DOWN))
    if qty_int <= 0:
        raise RiskRejection(
            "PRD §8 sizing",
            f"computed qty rounds to 0 for {sym} "
            f"(per-share risk={per_share_risk}, risk-budget={risk_dollars_cap})",
        )

    notional = Decimal(qty_int) * entry
    risk_dollars = Decimal(qty_int) * per_share_risk

    return SizedOrder(
        symbol=sym,
        side=side,
        qty=qty_int,
        entry_price=entry,
        stop_price=stop,
        notional=notional,
        risk_dollars=risk_dollars,
        nav_used=nav,
    )


# ---------------------------------------------------------------------------- submit


def submit_order(
    *,
    broker: AlpacaBroker,
    symbol: str,
    side: OrderSide,
    entry_price: Decimal,
    stop_price: Decimal,
    portfolio: PortfolioState | None = None,
    take_profit_price: Decimal | None = None,
    client_order_id: str | None = None,
    db_path: Path | None = None,
) -> BrokerOrder:
    """Size and submit one bracket order. Writes to the internal book.

    This is the *only* sanctioned path for production code to send orders.
    It enforces:

    * Kill-switch: refuses if `book.is_trading_disabled()` is True.
    * Sizing + caps: via `size_position`.
    * Broker-side stop: via `AlpacaBroker.submit_bracket_order`, which
      always sends a stop leg (CLAUDE.md hard rule 3).

    Returns the `BrokerOrder` from the broker; on error, raises
    `TradingDisabledError` or `RiskRejection`.
    """
    if book.is_trading_disabled(db_path=db_path):
        raise TradingDisabledError(
            "trading is disabled (kill switch engaged); refusing to submit orders"
        )
    snap = portfolio if portfolio is not None else snapshot_portfolio_from_broker(broker)

    sized = size_position(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
        portfolio=snap,
    )
    coid = client_order_id or f"casino-{uuid.uuid4().hex[:16]}"
    order = broker.submit_bracket_order(
        symbol=sized.symbol,
        qty=sized.qty,
        side=sized.side,
        stop_price=sized.stop_price,
        take_profit_price=take_profit_price,
        client_order_id=coid,
    )
    book.insert_order(
        broker_order_id=order.id,
        client_order_id=order.client_order_id or coid,
        symbol=sized.symbol,
        side=sized.side,
        qty=sized.qty,
        stop_price=sized.stop_price,
        limit_price=None,
        submitted_at_utc=order.submitted_at,
        status=order.status,
        notional_estimate=sized.notional,
        db_path=db_path,
    )
    logger.info(
        "risk: submitted bracket {} {} {} @ ~{} (stop {}, risk ${}, nav ${})",
        sized.side,
        sized.qty,
        sized.symbol,
        sized.entry_price,
        sized.stop_price,
        sized.risk_dollars,
        sized.nav_used,
    )
    return order


# ---------------------------------------------------------------------------- kill switch


@dataclass(frozen=True)
class KillSwitchResult:
    """Summary of one kill-switch invocation."""

    cancelled_orders: int
    closed_positions: int
    flag_set: bool
    reason: str


def flatten_and_disable(
    *,
    broker: AlpacaBroker | None = None,
    reason: str = "manual",
    db_path: Path | None = None,
) -> KillSwitchResult:
    """The kill switch.

    Behavior (CLAUDE.md hard rule 4 / PRD §8):

    1. Cancel all open orders (broker-side).
    2. Submit market-close for every open position (broker-side).
    3. Set `trading_disabled=1` in the run-state table; future
       `submit_order` calls raise `TradingDisabledError`.

    Returns a structured result so the CLI / test suite can assert on
    each step.
    """
    if broker is None:
        from casino.execution.alpaca_broker import build_default_broker  # noqa: PLC0415

        broker = build_default_broker()

    cancelled = 0
    try:
        cancelled = broker.cancel_all()
    except Exception as e:  # noqa: BLE001 — kill switch must be best-effort
        logger.error("kill_switch: cancel_all failed: {}", e)

    closed = 0
    try:
        closed_orders = broker.close_all_positions(cancel_orders=True)
        closed = len(closed_orders)
    except Exception as e:  # noqa: BLE001
        logger.error("kill_switch: close_all_positions failed: {}", e)

    book.set_trading_disabled(True, reason=reason, db_path=db_path)

    logger.warning(
        "kill_switch engaged ({}); cancelled={}, closed_positions={}",
        reason,
        cancelled,
        closed,
    )
    return KillSwitchResult(
        cancelled_orders=cancelled,
        closed_positions=closed,
        flag_set=True,
        reason=reason,
    )


def re_enable_trading(*, db_path: Path | None = None) -> None:
    """Clear the kill-switch flag (operator action — see RUNBOOK)."""
    book.set_trading_disabled(False, reason=None, db_path=db_path)
    logger.warning("kill_switch flag cleared; trading re-enabled")


# ---------------------------------------------------------------------------- exposure check


def check_gross_exposure(
    portfolio: PortfolioState,
    *,
    candidate_notionals: Iterable[Decimal] = (),
    max_gross_exposure: Decimal | float | None = None,
) -> bool:
    """Return True iff existing + candidate notionals stay under the gross cap.

    Used by the daily job before submitting a basket: lets us reject the
    *entire* basket as a unit when adding all positions would breach the
    100% cap, rather than partial-fill the basket.
    """
    cfg = get_config()
    ge = _to_decimal(
        max_gross_exposure if max_gross_exposure is not None else cfg.max_gross_exposure
    )
    cap = portfolio.nav * ge
    total = portfolio.gross_exposure_dollars + sum(
        (_to_decimal(n) for n in candidate_notionals), start=_zero()
    )
    return total <= cap
