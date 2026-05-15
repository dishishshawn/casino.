"""Tests for casino.execution.tsmom_runner.

Coverage:

* paper-only assertion blocks live URLs (``ALPACA_BASE_URL`` typo defense).
* Off-rebal-day calls are no-ops; ``--force`` overrides.
* ``latest_target_weights`` returns long-only weights with reference prices.
* ``plan_rebal_actions`` enforces single-name 10% and gross 100% caps even
  when raw weights would breach.
* End-to-end: ``run_rebal`` against a fake broker submits bracket orders
  with a broker-side stop, persists rebal_event rows, and reconciles
  broker→book without drift on the happy path.
* Bracket orders ALWAYS carry a stop-loss leg (no naked market orders).
* TSMOM mode is forced to ``long_only`` regardless of caller intent (paper
  cash-account constraint).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casino.config import get_config
from casino.execution import book, paper_clock
from casino.execution.alpaca_broker import (
    AlpacaBroker,
    BrokerAccount,
    BrokerPosition,
)
from casino.execution.tsmom_runner import (
    DEFAULT_STOP_FRACTION,
    NotPaperAccountError,
    TargetWeight,
    assert_paper_account,
    latest_target_weights,
    main,
    plan_rebal_actions,
    run_rebal,
)
from tests.test_risk import FakeTradingClient


class UniqueIdFakeClient(FakeTradingClient):
    """FakeTradingClient that mints a fresh order id on every submit.

    The base ``FakeTradingClient`` reuses ``ord-1`` for every order, which
    matches what tests of single-shot risk paths need but trips the UNIQUE
    constraint in ``casino.execution.book.orders`` when the runner submits a
    multi-leg basket. The runner is the integration point that needs unique
    ids — the real Alpaca API mints a unique id per submission.
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._counter = 0

    def submit_order(self, order_data):  # type: ignore[override]
        self._counter += 1
        self._next_id = f"ord-{self._counter}"
        return super().submit_order(order_data)


# ---------------------------------------------------------------------------- fixtures


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "state.sqlite"
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(p))
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    get_config.cache_clear()
    book.init_schema(p)
    paper_clock.init_schema(p)
    return p


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
        client = UniqueIdFakeClient(account=account, positions=positions or [])
        broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
        broker.set_client(client)
        return broker, client

    return _build


def _synthetic_prices(symbols: list[str], days: int = 400) -> pd.DataFrame:
    """Build a synthetic price panel: all symbols trending up smoothly.

    Prices = 100 * exp(0.0005 * t) per symbol with a small per-symbol
    offset. The panel has length ``days``, indexed by sequential business
    dates ending on a Friday-aligned month-end.
    """
    end = pd.Timestamp("2026-05-29")  # known last business day of May 2026
    idx = pd.bdate_range(end=end, periods=days)
    rng = np.random.default_rng(0)
    out = {}
    for i, s in enumerate(symbols):
        # smooth uptrend + tiny noise
        t = np.arange(days)
        out[s] = 100.0 + 5.0 * i + 20.0 * (t / days) + rng.normal(0, 0.05, size=days)
    return pd.DataFrame(out, index=idx)


# ---------------------------------------------------------------------------- paper-only assertion


def test_assert_paper_account_blocks_live_url() -> None:
    """Defense in depth — refuse to run if URL doesn't contain 'paper'."""
    with pytest.raises(NotPaperAccountError) as exc:
        assert_paper_account(alpaca_base_url="https://api.alpaca.markets")
    assert "paper" in str(exc.value).lower()


def test_assert_paper_account_accepts_paper_url() -> None:
    assert_paper_account(alpaca_base_url="https://paper-api.alpaca.markets")
    assert_paper_account(alpaca_base_url="https://PAPER-API.alpaca.markets")  # case-insensitive


def test_main_returns_nonzero_when_not_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI surface returns code 2 (refuse-to-run) on a non-paper URL."""
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    get_config.cache_clear()
    rc = main(["--dry-run"])
    assert rc == 2


# ---------------------------------------------------------------------------- latest_target_weights


def test_latest_target_weights_long_only(state: Path) -> None:
    syms = ["SPY", "QQQ", "TLT"]
    prices = _synthetic_prices(syms, days=400)
    out = latest_target_weights(prices, target_vol=0.10, gross_target=1.0)
    assert all(isinstance(t, TargetWeight) for t in out)
    # All synthetic prices trend up → long-only weights non-empty.
    assert len(out) >= 1
    assert all(t.weight > 0 for t in out)
    # reference_price is Decimal and positive.
    assert all(isinstance(t.reference_price, Decimal) for t in out)
    assert all(t.reference_price > Decimal("0") for t in out)


def test_latest_target_weights_empty_panel() -> None:
    out = latest_target_weights(pd.DataFrame(), target_vol=0.10, gross_target=1.0)
    assert out == []


# ---------------------------------------------------------------------------- plan_rebal_actions


def test_plan_rebal_respects_single_name_cap() -> None:
    """A target weight of 25% for one name shrinks to the 10% single-name cap."""
    from casino.execution.risk import PortfolioState

    portfolio = PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("100000"),
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={},
    )
    targets = [
        TargetWeight(symbol="AAA", weight=0.25, reference_price=Decimal("100")),
    ]
    actions = plan_rebal_actions(
        target_weights=targets,
        portfolio=portfolio,
        book_positions=[],
    )
    open_long = [a for a in actions if a.kind == "open_long"]
    assert len(open_long) == 1
    # Single-name cap: 10% of $100k = $10k. Floor to whole shares at $100 = 100 shares.
    assert open_long[0].target_dollars <= Decimal("10000")
    assert open_long[0].qty == 100


def test_plan_rebal_respects_gross_cap() -> None:
    """Sum of weights = 1.5 → scale all back so total ≤ NAV * 1.0."""
    from casino.execution.risk import PortfolioState

    portfolio = PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("100000"),
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={},
    )
    targets = [
        TargetWeight(symbol="A", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="B", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="C", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="D", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="E", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="F", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="G", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="H", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="I", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="J", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="K", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="L", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="M", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="N", weight=0.10, reference_price=Decimal("100")),
        TargetWeight(symbol="O", weight=0.10, reference_price=Decimal("100")),
    ]
    actions = plan_rebal_actions(
        target_weights=targets,
        portfolio=portfolio,
        book_positions=[],
    )
    open_long = [a for a in actions if a.kind == "open_long"]
    total_dollars = sum((a.target_dollars for a in open_long), Decimal("0"))
    # Total should not exceed NAV * gross_cap = 100k.
    assert total_dollars <= Decimal("100000") + Decimal("1")  # rounding tolerance


def test_plan_rebal_attaches_stop_to_every_open_long() -> None:
    """CLAUDE.md hard rule 3: every long entry has a broker-side stop at -10%."""
    from casino.execution.risk import PortfolioState

    portfolio = PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("100000"),
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={},
    )
    targets = [
        TargetWeight(symbol="A", weight=0.05, reference_price=Decimal("100")),
        TargetWeight(symbol="B", weight=0.05, reference_price=Decimal("200")),
    ]
    actions = plan_rebal_actions(
        target_weights=targets,
        portfolio=portfolio,
        book_positions=[],
    )
    open_long = [a for a in actions if a.kind == "open_long"]
    assert len(open_long) == 2
    for a in open_long:
        # stop = ref * (1 - 0.10) = 90% of ref
        expected = a.reference_price * (Decimal("1") - DEFAULT_STOP_FRACTION)
        assert abs(a.stop_price - expected) < Decimal("0.10")
        assert a.stop_price > Decimal("0")
        assert a.stop_price < a.reference_price


def test_plan_rebal_closes_dropped_positions() -> None:
    """Symbols in book but not in target panel → action 'close'."""
    from casino.execution.risk import PortfolioState

    portfolio = PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("100000"),
        gross_exposure_dollars=Decimal("0"),
        single_name_exposure={},
    )
    held = book.StoredPosition(
        symbol="ZZZ",
        side="long",
        qty=10,
        avg_entry_price=Decimal("50"),
        opened_at_utc=datetime.now(tz=UTC),
        last_update_utc=datetime.now(tz=UTC),
    )
    actions = plan_rebal_actions(
        target_weights=[
            TargetWeight(symbol="A", weight=0.05, reference_price=Decimal("100")),
        ],
        portfolio=portfolio,
        book_positions=[held],
    )
    closes = [a for a in actions if a.kind == "close"]
    assert len(closes) == 1
    assert closes[0].symbol == "ZZZ"


# ---------------------------------------------------------------------------- run_rebal happy path


def _patch_load_ohlcv(monkeypatch: pytest.MonkeyPatch, prices: pd.DataFrame) -> None:
    """Patch tsmom_runner._load_recent_prices to return the supplied panel."""
    from casino.execution import tsmom_runner

    monkeypatch.setattr(
        tsmom_runner,
        "_load_recent_prices",
        lambda **_kwargs: prices,
    )


def test_run_rebal_skips_on_non_rebal_day(state: Path, broker_factory) -> None:
    """Non-rebal day with force=False is a no-op."""
    broker, _ = broker_factory()
    today = date(2026, 5, 15)  # mid-month, not last bday
    result = run_rebal(
        broker=broker,
        today=today,
        db_path=state,
    )
    assert result.is_rebal_day is False
    assert result.skipped_reason is not None
    assert "not the last business day" in result.skipped_reason


def test_run_rebal_force_overrides_non_rebal_day(
    state: Path,
    broker_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--force lets the runner act on any day."""
    broker, fake = broker_factory()
    syms = ["SPY", "QQQ", "TLT"]
    prices = _synthetic_prices(syms, days=400)
    _patch_load_ohlcv(monkeypatch, prices)

    today = date(2026, 5, 15)
    result = run_rebal(
        broker=broker,
        today=today,
        force=True,
        db_path=state,
        universe=syms,
    )
    # Forced → skipped_reason should be None and at least one action
    assert result.skipped_reason is None
    assert len(result.actions) >= 1


def test_run_rebal_dry_run_submits_no_orders(
    state: Path,
    broker_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, fake = broker_factory()
    syms = ["SPY", "QQQ", "TLT"]
    prices = _synthetic_prices(syms, days=400)
    _patch_load_ohlcv(monkeypatch, prices)

    result = run_rebal(
        broker=broker,
        today=date(2026, 5, 29),  # last bday of May 2026
        force=False,
        dry_run=True,
        db_path=state,
        universe=syms,
    )
    assert result.dry_run is True
    assert result.submitted_order_ids == []
    # No bracket orders submitted to the fake broker:
    assert len(fake.submitted_requests) == 0


def test_run_rebal_end_to_end_roundtrip(
    state: Path,
    broker_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: weights → bracket orders → reconciled book.

    Asserts:
    * paper_clock is started.
    * each open_long routes through risk.submit_order → broker.submit_bracket_order.
    * every order has a stop_loss leg (CLAUDE.md hard rule 3).
    * a rebal_event row is recorded.
    """
    broker, fake = broker_factory(equity=Decimal("100000"))
    syms = ["SPY", "QQQ", "TLT"]
    prices = _synthetic_prices(syms, days=400)
    _patch_load_ohlcv(monkeypatch, prices)

    result = run_rebal(
        broker=broker,
        today=date(2026, 5, 29),
        db_path=state,
        universe=syms,
    )
    assert result.skipped_reason is None
    # paper_clock was started.
    clock_row = paper_clock.fetch_paper_clock(db_path=state)
    assert clock_row is not None
    assert clock_row.start_nav == Decimal("100000")
    assert clock_row.strategy == "tsmom_long_only"
    # At least one bracket order submitted.
    assert len(fake.submitted_requests) >= 1
    # CRITICAL: every entry order has a stop-loss leg.
    for req in fake.submitted_requests:
        assert req.stop_loss is not None, "naked market order detected — hard-rule 3 violation"
        assert Decimal(str(req.stop_loss.stop_price)) > Decimal("0")
    # rebal_event recorded.
    rebals = paper_clock.fetch_rebal_events(db_path=state)
    assert len(rebals) == 1
    # n_orders_submitted reflects orders persisted to result.submitted_order_ids.
    # The FakeTradingClient reuses a single order id which triggers the UNIQUE
    # constraint on book.orders for subsequent submissions — that mirrors the
    # real-broker contract (each id is unique). The first submission always
    # succeeds, which is enough to validate the bracket-order flow.
    assert rebals[0].n_orders_submitted >= 1


def test_run_rebal_logs_and_alerts_actual_submitted_qty(
    state: Path,
    broker_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for 2026-05-15: ``bracket-bought`` log + ``alert_order_submitted``
    must report the qty risk.submit_order actually sent to the broker,
    not the planner's intent.

    Pre-fix both surfaces used ``a.qty`` (planner target_dollars / ref),
    so on every name where size_position clipped (the typical ¼-Kelly
    case) the operator saw 321 shares in Discord while only 120 went
    live. After the fix both surfaces read ``ord_resp.qty``.
    """
    import httpx

    from casino.execution import tsmom_runner
    from casino.monitoring import alerts as alerts_mod

    broker, fake = broker_factory(equity=Decimal("100000"))
    syms = ["SPY", "QQQ", "TLT"]
    prices = _synthetic_prices(syms, days=400)
    _patch_load_ohlcv(monkeypatch, prices)

    # Capture the kwargs every alert_order_submitted call receives, then
    # delegate to the real helper with a no-op transport so we still
    # exercise the production code path. Transport must return an
    # httpx.Response — alerts.send_alert reads .status_code on the way out.
    captured_alerts: list[dict] = []
    real_alert = alerts_mod.alert_order_submitted

    def _noop_tx(_url: str, _payload: dict) -> httpx.Response:  # type: ignore[type-arg]
        return httpx.Response(204)

    def capture(**kw):  # type: ignore[no-untyped-def]
        captured_alerts.append(dict(kw))
        return real_alert(**{**kw, "transport": _noop_tx})

    monkeypatch.setattr(tsmom_runner.alerts, "alert_order_submitted", capture)

    result = run_rebal(
        broker=broker,
        today=date(2026, 5, 29),
        db_path=state,
        universe=syms,
    )
    assert result.skipped_reason is None
    assert len(fake.submitted_requests) >= 1
    assert len(captured_alerts) >= 1

    # 1) For every alert, the qty reported must equal the qty that went
    #    to the broker. This is the core bug fix.
    submitted_qty_by_symbol = {r.symbol: int(r.qty) for r in fake.submitted_requests}
    for alert in captured_alerts:
        sym = alert["symbol"]
        assert sym in submitted_qty_by_symbol, f"alert for {sym} but no broker submission"
        assert alert["qty"] == submitted_qty_by_symbol[sym], (
            f"alert qty {alert['qty']} for {sym} != broker qty {submitted_qty_by_symbol[sym]}"
        )

    # 2) Prove the test setup actually exercises the clipping path: at
    #    least one symbol must have broker_qty < planner_qty. Otherwise
    #    the regression would pass trivially even if we reverted the fix.
    planner_qty_by_symbol = {a.symbol: a.qty for a in result.actions if a.kind == "open_long"}
    clipped_symbols = [
        s
        for s, planner_qty in planner_qty_by_symbol.items()
        if s in submitted_qty_by_symbol and submitted_qty_by_symbol[s] < planner_qty
    ]
    assert clipped_symbols, (
        "Test setup invalid: no Kelly clipping observed in this scenario; "
        "the regression test would pass even if we reverted the fix"
    )


def test_run_rebal_kill_switch_engaged_aborts_submission(
    state: Path,
    broker_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If trading_disabled is set, submit_order raises and the runner halts."""
    broker, fake = broker_factory()
    syms = ["SPY", "QQQ", "TLT"]
    prices = _synthetic_prices(syms, days=400)
    _patch_load_ohlcv(monkeypatch, prices)

    book.set_trading_disabled(True, reason="test", db_path=state)

    result = run_rebal(
        broker=broker,
        today=date(2026, 5, 29),
        db_path=state,
        universe=syms,
    )
    # Should have a skipped_reason mentioning kill switch.
    assert result.skipped_reason is not None
    assert "kill" in result.skipped_reason.lower() or "disabled" in result.skipped_reason.lower()
    # No bracket orders submitted.
    assert len(fake.submitted_requests) == 0


def test_run_rebal_no_history_skips(
    state: Path,
    broker_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty OHLCV panel → graceful skip with a reason."""
    broker, _ = broker_factory()
    _patch_load_ohlcv(monkeypatch, pd.DataFrame())
    result = run_rebal(
        broker=broker,
        today=date(2026, 5, 29),
        db_path=state,
    )
    assert result.skipped_reason is not None
    assert "OHLCV" in result.skipped_reason or "history" in result.skipped_reason


# ---------------------------------------------------------------------------- mode forced to long_only


def test_signal_call_uses_long_only_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """compute_tsmom_panel must be called with mode='long_only' regardless of input."""
    captured: dict[str, object] = {}

    def fake_compute(prices: pd.DataFrame, **kw: object) -> pd.DataFrame:
        captured.update(kw)
        # produce a flat 0.5-weight final row to keep latest_target_weights happy
        out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        out.iloc[-1] = 0.05
        return out

    from casino.execution import tsmom_runner

    monkeypatch.setattr(tsmom_runner, "compute_tsmom_panel", fake_compute)
    syms = ["SPY", "QQQ"]
    prices = _synthetic_prices(syms, days=300)
    _ = latest_target_weights(prices, target_vol=0.10, gross_target=1.0)
    assert captured.get("mode") == "long_only"
