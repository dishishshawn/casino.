# Breached Protective Stop Handling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the protective-stop guard handle a long position that has already fallen past its stop level — never crash, alert the same evening, and auto-sell it at the next market open.

**Architecture:** `ensure_protective_stops` gains per-position fault isolation and a breach classifier. At EOD (market closed) it arms valid stops and *flags* breached ones (the `reconcile_eod` job alerts). At the morning book-sync (market open) the same function is called with `liquidate_breached=True`, so materially-breached positions are market-sold via `broker.close_position`. A 0.25% cushion keeps quote noise from triggering a sale.

**Tech Stack:** Python 3.11, pytest, alpaca-py wrapper (`AlpacaBroker`), loguru, Decimal money.

**Spec:** `docs/superpowers/specs/2026-06-15-breached-stop-handling-design.md`

---

## File Structure

- `casino/execution/reconcile.py` — `StopArmResult` (new fields), `ensure_protective_stops` (breach logic, fault isolation, liquidation). Add a module-level `STOP_BREACH_CUSHION` constant.
- `casino/monitoring/alerts.py` — two new typed alert helpers.
- `casino/jobs/reconcile_eod.py` — alert on breached-but-not-liquidated results.
- `casino/jobs/sync_book.py` — call the guard with `liquidate_breached=True`, alert on liquidations.
- `tests/test_risk.py` — extend `FakeTradingClient` with a market-open knob and a submit-failure knob (test support).
- `tests/test_reconcile.py` — behavior tests for the guard.
- `tests/test_alerts.py` — tests for the two new alert helpers.

---

### Task 1: Test-support knobs on the fake broker client

**Files:**
- Modify: `tests/test_risk.py:42-86` and `:158-162`

- [ ] **Step 1: Add constructor knobs**

In `FakeTradingClient.__init__`, add two parameters and store them. Change the signature and body to include:

```python
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
        self.order_history: list[BrokerOrder] = list(order_history or [])
        self._next_id = next_order_id
        self.submitted_requests: list[Any] = []
        self.cancelled = 0
        self.closed_positions: list[str] = []
        self._market_open = market_open
        self._fail_symbols = {s.upper() for s in (fail_symbols or set())}
```

- [ ] **Step 2: Make `submit_order` raise for `fail_symbols`**

At the top of `submit_order` (before appending to `submitted_requests`), add:

```python
    def submit_order(self, order_data: Any) -> Any:
        sym = str(getattr(order_data, "symbol", "TEST")).upper()
        if sym in self._fail_symbols:
            raise RuntimeError(f"simulated broker reject for {sym}")
        self.submitted_requests.append(order_data)
        ...  # rest unchanged
```

- [ ] **Step 3: Honor `market_open` in `get_clock`**

Replace the hardcoded `is_open = True` clock with one that reflects the knob:

```python
    def get_clock(self) -> Any:
        is_open = self._market_open

        class _C:
            pass

        c = _C()
        c.is_open = is_open
        return c
```

- [ ] **Step 4: Run the existing suite to confirm no regressions**

Run: `uv run pytest tests/test_risk.py tests/test_reconcile.py -q`
Expected: PASS (defaults preserve old behavior).

- [ ] **Step 5: Commit**

```bash
git add tests/test_risk.py
git commit -m "test: add market-open and submit-failure knobs to FakeTradingClient"
```

---

### Task 2: Breach classification + fault isolation (no selling yet)

**Files:**
- Modify: `casino/execution/reconcile.py` — `StopArmResult` (~332-341), `ensure_protective_stops` (~344-449)
- Test: `tests/test_reconcile.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_reconcile.py`:

```python
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
    # entry 100, stop 90; market 100 is well above stop → arm.
    broker = _broker_with([_long("AAA", "100", "100")])
    results = reconcile.ensure_protective_stops(broker=broker, db_path=isolated_state)
    r = next(x for x in results if x.symbol == "AAA")
    assert r.armed is True
    assert r.breached is False
    assert r.liquidated is False


def test_ensure_stops_flags_breached_without_arming(isolated_state: Path) -> None:
    # entry 100, stop 90; market 80 is below stop → cannot arm, must flag.
    broker = _broker_with([_long("USO", "100", "80")])
    results = reconcile.ensure_protective_stops(broker=broker, db_path=isolated_state)
    r = next(x for x in results if x.symbol == "USO")
    assert r.breached is True
    assert r.armed is False
    assert r.liquidated is False
    # no stop order was submitted for the breached symbol
    fake = broker._client  # type: ignore[attr-defined]
    assert all(getattr(req, "symbol", "") != "USO" for req in fake.submitted_requests)


def test_ensure_stops_isolates_one_failure(isolated_state: Path) -> None:
    from tests.test_risk import FakeTradingClient
    account = broker_account = None  # placeholder, replaced below
    broker = _broker_with([_long("AAA", "100", "100"), _long("BBB", "100", "100")])
    # Force BBB's arm to raise; AAA must still be armed.
    broker._client._fail_symbols = {"BBB"}  # type: ignore[attr-defined]
    results = reconcile.ensure_protective_stops(broker=broker, db_path=isolated_state)
    by_sym = {r.symbol: r for r in results}
    assert by_sym["AAA"].armed is True
    assert by_sym["BBB"].armed is False
```

> Note: `_broker_with` sets the client via `broker.set_client(fake)`; access it in tests through `broker._client`. If the attribute name differs, read `casino/execution/alpaca_broker.py` `set_client`/`_ensure_client` and use the real attribute.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_reconcile.py -k ensure_stops -v`
Expected: FAIL — `StopArmResult` has no `breached`/`liquidated`; breached position currently raises or arms.

- [ ] **Step 3: Add result fields + cushion constant**

In `casino/execution/reconcile.py`, extend the dataclass (keep existing fields, add three with defaults so existing keyword construction stays valid):

```python
@dataclass(frozen=True)
class StopArmResult:
    """Outcome of the protective-stop guard for one position."""

    symbol: str
    qty: int
    stop_price: Decimal
    already_protected: bool
    armed: bool
    order_id: str | None
    breached: bool = False
    liquidated: bool = False
    market_price: Decimal | None = None
```

Add near the top of the stops section (just above `ensure_protective_stops`):

```python
# A protective sell-stop is only valid when its price is below the current
# market price. A position that has fallen to/below its stop level is
# "breached" — it cannot be re-armed and must be exited instead. The cushion
# keeps ordinary quote jitter (a cent or two) from counting as a material
# breach worth auto-selling.
STOP_BREACH_CUSHION = Decimal("0.0025")  # 0.25%
```

- [ ] **Step 4: Rewrite the per-position loop with isolation + breach branch**

Replace the body of the `for p in positions:` loop in `ensure_protective_stops` (the non-long skip stays). For each long position, wrap the work in try/except and branch on breach. Use this structure:

```python
    for p in positions:
        if p.side != "long":
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
                    symbol=sym, qty=p.qty, stop_price=stop_px,
                    already_protected=True, armed=False, order_id=None,
                    market_price=p.market_price,
                )
            )
            continue

        # Breached: position has already fallen to/below its stop level. A
        # normal sell-stop would be rejected (stop >= market). Flag it; the
        # caller decides whether to liquidate (only at market open).
        if stop_px >= p.market_price:
            results.append(
                StopArmResult(
                    symbol=sym, qty=p.qty, stop_price=stop_px,
                    already_protected=False, armed=False, order_id=None,
                    breached=True, liquidated=False, market_price=p.market_price,
                )
            )
            logger.warning(
                "ensure_protective_stops: {} BREACHED — stop {} >= market {} "
                "(unprotected; awaiting liquidation)",
                sym, stop_px, p.market_price,
            )
            continue

        if dry_run:
            logger.warning(
                "ensure_protective_stops: {} UNPROTECTED (would arm GTC stop @ {})",
                sym, stop_px,
            )
            results.append(
                StopArmResult(
                    symbol=sym, qty=p.qty, stop_price=stop_px,
                    already_protected=False, armed=False, order_id=None,
                    market_price=p.market_price,
                )
            )
            continue

        try:
            order = broker.submit_stop_order(
                symbol=sym, qty=p.qty, side="sell",
                stop_price=stop_px, time_in_force="gtc",
            )
            book.insert_order(
                broker_order_id=order.id,
                client_order_id=order.client_order_id or None,
                symbol=sym, side="sell", qty=p.qty, stop_price=stop_px,
                limit_price=None, submitted_at_utc=order.submitted_at,
                status=order.status, notional_estimate=None, db_path=db_path,
            )
            logger.warning(
                "ensure_protective_stops: re-armed {} with GTC stop @ {} (order {})",
                sym, stop_px, order.id,
            )
            results.append(
                StopArmResult(
                    symbol=sym, qty=p.qty, stop_price=stop_px,
                    already_protected=False, armed=True, order_id=order.id,
                    market_price=p.market_price,
                )
            )
        except Exception:  # noqa: BLE001 — one symbol must not abort the rest
            logger.exception("ensure_protective_stops: failed to arm {}", sym)
            results.append(
                StopArmResult(
                    symbol=sym, qty=p.qty, stop_price=stop_px,
                    already_protected=False, armed=False, order_id=None,
                    market_price=p.market_price,
                )
            )
    return results
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_reconcile.py -k ensure_stops -v`
Expected: PASS (all three).

- [ ] **Step 6: Commit**

```bash
git add casino/execution/reconcile.py tests/test_reconcile.py
git commit -m "feat(reconcile): flag breached stops and isolate per-position failures"
```

---

### Task 3: Liquidation of materially-breached positions (market-open only)

**Files:**
- Modify: `casino/execution/reconcile.py` — `ensure_protective_stops` signature + breach branch
- Test: `tests/test_reconcile.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_reconcile.py`:

```python
def test_ensure_stops_liquidates_material_breach_when_open(isolated_state: Path) -> None:
    # entry 100 -> stop 90; market 80 is well past the cushion. Market open.
    broker = _broker_with([_long("USO", "100", "80")])  # default market_open=True
    results = reconcile.ensure_protective_stops(
        broker=broker, db_path=isolated_state, liquidate_breached=True
    )
    r = next(x for x in results if x.symbol == "USO")
    assert r.breached is True
    assert r.liquidated is True
    assert "USO" in broker._client.closed_positions  # type: ignore[attr-defined]


def test_ensure_stops_no_liquidation_when_market_closed(isolated_state: Path) -> None:
    from decimal import Decimal as D
    from tests.test_risk import FakeTradingClient
    from casino.execution.alpaca_broker import AlpacaBroker, BrokerAccount
    account = BrokerAccount(
        account_number="paper-1", status="ACTIVE", equity=D("100000"),
        cash=D("100000"), buying_power=D("100000"), last_equity=D("100000"),
        pattern_day_trader=False, trading_blocked=False,
    )
    fake = FakeTradingClient(account=account, positions=[_long("USO", "100", "80")], market_open=False)
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
    # entry 100 -> stop 90; market 89.95 is < 0.25% below stop → no sale.
    broker = _broker_with([_long("AAA", "100", "89.95")])
    results = reconcile.ensure_protective_stops(
        broker=broker, db_path=isolated_state, liquidate_breached=True
    )
    r = next(x for x in results if x.symbol == "AAA")
    assert r.breached is True
    assert r.liquidated is False
    assert broker._client.closed_positions == []  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_reconcile.py -k "liquidat or cushion" -v`
Expected: FAIL — `ensure_protective_stops` has no `liquidate_breached` parameter.

- [ ] **Step 3: Add the parameter and liquidation logic**

Change the signature:

```python
def ensure_protective_stops(
    *,
    broker: AlpacaBroker,
    stop_fraction: Decimal = DEFAULT_STOP_FRACTION,
    db_path: Path | None = None,
    dry_run: bool = False,
    liquidate_breached: bool = False,
) -> list[StopArmResult]:
```

Replace the breached branch from Task 2 (the `if stop_px >= p.market_price:` block) with one that may liquidate. A breach is *material* when the price is below the stop by more than the cushion:

```python
        if stop_px >= p.market_price:
            material = p.market_price <= stop_px * (Decimal("1") - STOP_BREACH_CUSHION)
            liquidated = False
            order_id: str | None = None
            if liquidate_breached and material and not dry_run and broker.is_market_open():
                try:
                    close = broker.close_position(sym)
                    order_id = close.id
                    book.insert_order(
                        broker_order_id=close.id,
                        client_order_id=close.client_order_id or None,
                        symbol=sym, side="sell", qty=p.qty, stop_price=None,
                        limit_price=None, submitted_at_utc=close.submitted_at,
                        status=close.status, notional_estimate=None, db_path=db_path,
                    )
                    liquidated = True
                    logger.warning(
                        "ensure_protective_stops: LIQUIDATED {} at market — was "
                        "below stop {} (market {}), order {}",
                        sym, stop_px, p.market_price, close.id,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "ensure_protective_stops: liquidation of {} failed", sym
                    )
            else:
                logger.warning(
                    "ensure_protective_stops: {} BREACHED — stop {} >= market {} "
                    "(unprotected; material={})",
                    sym, stop_px, p.market_price, material,
                )
            results.append(
                StopArmResult(
                    symbol=sym, qty=p.qty, stop_price=stop_px,
                    already_protected=False, armed=False, order_id=order_id,
                    breached=True, liquidated=liquidated, market_price=p.market_price,
                )
            )
            continue
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_reconcile.py -k "ensure_stops or liquidat or cushion" -v`
Expected: PASS (all guard tests).

- [ ] **Step 5: Commit**

```bash
git add casino/execution/reconcile.py tests/test_reconcile.py
git commit -m "feat(reconcile): auto-liquidate materially-breached positions at market open"
```

---

### Task 4: Alert helpers

**Files:**
- Modify: `casino/monitoring/alerts.py`
- Test: `tests/test_alerts.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_alerts.py` (follow the file's existing capture-transport pattern; if it differs, mirror the existing tests there):

```python
def test_alert_below_stop_unprotected_is_critical() -> None:
    captured = {}

    def transport(url, payload):
        captured["payload"] = payload
        class _R:
            status_code = 204
        return _R()

    res = alerts.alert_below_stop_unprotected(
        symbols=["USO"], webhook_url="http://x", transport=transport
    )
    assert res.sent is True
    embed = captured["payload"]["embeds"][0]
    assert "USO" in embed["description"] or "USO" in str(embed["fields"])
    assert embed["color"] == 0xE74C3C  # critical/red


def test_alert_positions_liquidated_is_critical() -> None:
    captured = {}

    def transport(url, payload):
        captured["payload"] = payload
        class _R:
            status_code = 204
        return _R()

    res = alerts.alert_positions_liquidated(
        details={"USO": "sold at $120.95, ~0.5% below stop"},
        webhook_url="http://x", transport=transport,
    )
    assert res.sent is True
    embed = captured["payload"]["embeds"][0]
    assert embed["color"] == 0xE74C3C
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_alerts.py -k "below_stop or liquidated" -v`
Expected: FAIL — helpers not defined.

- [ ] **Step 3: Implement the helpers**

Add to `casino/monitoring/alerts.py` (near the other typed helpers):

```python
def alert_below_stop_unprotected(
    *,
    symbols: list[str],
    webhook_url: str | None = None,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """EOD: positions that fell past their stop and can't be re-armed.

    The market is closed at EOD so we don't sell here — we warn the operator
    that these positions are unprotected and will be sold at the next open.
    """
    sym_str = ", ".join(symbols) or "(none)"
    return fire(
        title=f"Below protective stop — unprotected: {sym_str}",
        message=(
            f"{len(symbols)} position(s) have fallen past their protective "
            "stop and could not be re-armed (a stop can't sit above the "
            "current price). They are unprotected and will be sold at market "
            "on the next open. Review before then if you disagree."
        ),
        severity="critical",
        fields={"Symbols": sym_str, "Count": str(len(symbols))},
        webhook_url=webhook_url,
        transport=transport,
    )


def alert_positions_liquidated(
    *,
    details: dict[str, str],
    webhook_url: str | None = None,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """Morning: positions sold at market because they were below their stop."""
    sym_str = ", ".join(details) or "(none)"
    return fire(
        title=f"Sold at market — were below stop: {sym_str}",
        message=(
            f"{len(details)} position(s) were below their protective stop and "
            "have been sold at market to honor the risk limit."
        ),
        severity="critical",
        fields=details or {"Positions": "(none)"},
        webhook_url=webhook_url,
        transport=transport,
    )
```

> Confirm `fire(...)` accepts `webhook_url`; it does per `alerts.py`. If the existing `test_alerts.py` calls helpers without `webhook_url`, drop that kwarg here and rely on the injected `transport` only — match the existing tests.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_alerts.py -k "below_stop or liquidated" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add casino/monitoring/alerts.py tests/test_alerts.py
git commit -m "feat(alerts): add below-stop and liquidation critical alerts"
```

---

### Task 5: Wire the EOD reconcile job to alert on breached positions

**Files:**
- Modify: `casino/jobs/reconcile_eod.py:148-163`

- [ ] **Step 1: Update the stop-guard block**

Replace the `try:` block that calls `ensure_protective_stops` so it (a) passes `liquidate_breached=False` explicitly and (b) alerts on breached results:

```python
        stops_rearmed = 0
        try:
            stop_results = reconcile.ensure_protective_stops(
                broker=broker,
                stop_fraction=_resolve_stop_fraction(db_path=db_path),
                db_path=db_path,
                liquidate_breached=False,  # market is closed at EOD
            )
            armed = [r.symbol for r in stop_results if r.armed]
            stops_rearmed = len(armed)
            if armed:
                alerts.alert_stop_rearmed(armed=armed)
            breached = [r.symbol for r in stop_results if r.breached]
            if breached:
                alerts.alert_below_stop_unprotected(symbols=breached)
        except Exception as e:  # noqa: BLE001
            logger.exception("reconcile_eod: stop re-arm failed")
            alerts.alert_unhandled_exception(
                job="reconcile_eod.ensure_stops", exc_type=type(e).__name__, detail=str(e)
            )
```

- [ ] **Step 2: Type/lint check**

Run: `uv run mypy casino` and `uv run ruff check casino/jobs/reconcile_eod.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add casino/jobs/reconcile_eod.py
git commit -m "feat(reconcile_eod): alert on positions left below their stop"
```

---

### Task 6: Wire the morning book-sync job to liquidate breached positions

**Files:**
- Modify: `casino/jobs/sync_book.py:35-37`

- [ ] **Step 1: Add the protection-enforcement step after the book write**

In `main()`, after `n = reconcile.sync_book_from_broker(broker=broker)` and its log line, add:

```python
        n = reconcile.sync_book_from_broker(broker=broker)
        logger.info("sync_book: wrote {} broker positions into book.positions", n)

        # Market is open now (this runs ~09:45 CT). Enforce protection: arm
        # stops on anything naked, and sell anything already past its stop.
        stop_results = reconcile.ensure_protective_stops(
            broker=broker, liquidate_breached=True
        )
        liquidated = {
            r.symbol: (
                f"sold at ~{r.market_price}, stop was {r.stop_price}"
                if r.market_price is not None
                else f"sold; stop was {r.stop_price}"
            )
            for r in stop_results
            if r.liquidated
        }
        if liquidated:
            logger.warning("sync_book: liquidated below-stop positions: {}", list(liquidated))
            alerts.alert_positions_liquidated(details=liquidated)
        return 0
```

- [ ] **Step 2: Type/lint check**

Run: `uv run mypy casino` and `uv run ruff check casino/jobs/sync_book.py`
Expected: clean.

- [ ] **Step 3: Full suite + format check**

Run: `uv run pytest -q` then `uv run ruff format --check .`
Expected: all PASS / clean.

- [ ] **Step 4: Commit**

```bash
git add casino/jobs/sync_book.py
git commit -m "feat(sync_book): arm stops and liquidate below-stop positions at open"
```

---

## Self-Review

- **Spec coverage:** crash-proofing (Task 2), breach flag (Task 2), cushion (Task 3), EOD alert / no-trade (Tasks 3,5), morning liquidation (Tasks 3,6), per-position isolation (Task 2), alert wording (Task 4) — all mapped.
- **Type consistency:** `ensure_protective_stops(..., liquidate_breached: bool)` and `StopArmResult.{breached, liquidated, market_price}` are defined in Tasks 2–3 and used consistently in Tasks 5–6. Alert helpers `alert_below_stop_unprotected(symbols=...)` and `alert_positions_liquidated(details=...)` defined in Task 4, used in Tasks 5–6.
- **Placeholder scan:** test bodies and implementations are complete; the only conditional notes are the two "if the existing test/attr differs, match it" guards, which point at exact files to check.

## Out of scope (separate plan)

Arm the protective stop *immediately on fill* so positions never drift past their stop — the durable cure. Lives in the same morning slot; deferred per the spec.
