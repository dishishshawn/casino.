"""Re-verification of the kill-switch path post-task-36.

CLAUDE.md hard rule 4: "Kill switch must remain a single command that
flattens positions and disables order entry. If you change execution
code, re-verify the kill switch path still works end-to-end."

Task 36 added new execution code (``tsmom_runner``, ``tsmom_clock_check``,
``paper_clock``). This file is the explicit re-verification: simulate a
kill trigger end-to-end via ``tsmom_clock_check.run_daily_check`` and
assert that:

1. ``risk.flatten_and_disable`` is invoked.
2. Every broker position is closed (market-close orders submitted).
3. ``book.is_trading_disabled`` flips to True.
4. Subsequent ``risk.submit_order`` raises ``TradingDisabledError`` —
   the runner cannot bypass.
5. ``tsmom_runner.run_rebal`` halts gracefully (no orders submitted) when
   the kill flag is set.
6. The kill-switch CLI module is importable (single-command guarantee).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from casino.config import get_config
from casino.execution import book, paper_clock
from casino.execution.alpaca_broker import (
    AlpacaBroker,
    BrokerAccount,
    BrokerPosition,
)
from casino.execution.risk import (
    TradingDisabledError,
    flatten_and_disable,
    re_enable_trading,
    submit_order,
)
from casino.execution.tsmom_clock_check import run_daily_check
from tests.test_risk import FakeTradingClient


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "state.sqlite"
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(p))
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    get_config.cache_clear()
    book.init_schema(p)
    paper_clock.init_schema(p)
    return p


def _broker(
    *,
    equity: Decimal,
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


# ---------------------------------------------------------------------------- direct


def test_flatten_and_disable_blocks_subsequent_submit(state: Path) -> None:
    """The post-task-36 sanity check: flatten + flag → submit_order raises."""
    pos = BrokerPosition(
        symbol="SPY",
        qty=10,
        side="long",
        avg_entry_price=Decimal("400"),
        market_price=Decimal("400"),
        market_value=Decimal("4000"),
        unrealized_pl=Decimal("0"),
        cost_basis=Decimal("4000"),
    )
    broker, fake = _broker(equity=Decimal("100000"), positions=[pos])

    assert book.is_trading_disabled(db_path=state) is False
    result = flatten_and_disable(broker=broker, reason="task36-reverify", db_path=state)
    assert result.flag_set is True
    assert result.closed_positions == 1
    assert "SPY" in fake.closed_positions
    assert book.is_trading_disabled(db_path=state) is True

    # Subsequent submit_order via risk.py raises (the runner uses this path).
    with pytest.raises(TradingDisabledError):
        submit_order(
            broker=broker,
            symbol="QQQ",
            side="buy",
            entry_price=Decimal("400"),
            stop_price=Decimal("360"),
            db_path=state,
        )

    re_enable_trading(db_path=state)
    assert book.is_trading_disabled(db_path=state) is False


# ---------------------------------------------------------------------------- via clock_check


def test_kill_via_clock_check_is_end_to_end(
    state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Daily check fires kill criterion → clock_check flattens + disables.

    This is the live integration path the operator will hit in production:
    a cron-invoked ``tsmom_clock_check`` detects drawdown, fires alerts,
    flattens positions, and trips the kill flag.
    """
    paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    # 12% drawdown → exceeds 10% kill threshold.
    book.upsert_daily_pnl(
        book.DailyPnLRow(
            date="2026-05-20",
            equity_open=Decimal("100000"),
            equity_close=Decimal("88000"),
            realized_pl=Decimal("-12000"),
            unrealized_pl=Decimal("0"),
            n_positions=1,
            n_orders=0,
            notes=None,
        ),
        db_path=state,
    )
    pos = BrokerPosition(
        symbol="SPY",
        qty=10,
        side="long",
        avg_entry_price=Decimal("400"),
        market_price=Decimal("352"),
        market_value=Decimal("3520"),
        unrealized_pl=Decimal("-480"),
        cost_basis=Decimal("4000"),
    )
    broker, fake = _broker(equity=Decimal("88000"), positions=[pos])

    # Silence the Discord transport.
    class _Capture:
        def __call__(self, **kw):
            class _R:
                sent = True
                status_code = 204
                reason = "ok"

            return _R()

    monkeypatch.setattr("casino.execution.tsmom_clock_check.alerts.fire", _Capture())

    result = run_daily_check(broker=broker, db_path=state)

    assert result.kill_fired is True
    assert "drawdown" in result.triggered_criteria
    assert result.kill_switch_result is not None
    assert result.kill_switch_result.flag_set is True
    assert "SPY" in fake.closed_positions
    assert book.is_trading_disabled(db_path=state) is True


# ---------------------------------------------------------------------------- runner halts when killed


def test_tsmom_runner_halts_when_kill_flag_set(
    state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner uses ``risk.submit_order`` → kill flag aborts on first attempt."""
    from casino.execution import tsmom_runner

    book.set_trading_disabled(True, reason="pre-existing", db_path=state)

    pos: list[BrokerPosition] = []
    broker, fake = _broker(equity=Decimal("100000"), positions=pos)

    # Build a minimal panel so latest_target_weights returns at least one entry.
    idx = pd.bdate_range(end="2026-05-29", periods=400)
    syms = ["SPY", "QQQ"]
    prices = pd.DataFrame(
        {s: [100.0 + i * 0.05 for i in range(len(idx))] for s in syms},
        index=idx,
    )
    monkeypatch.setattr(tsmom_runner, "_load_recent_prices", lambda **_kw: prices)

    result = tsmom_runner.run_rebal(
        broker=broker,
        today=date(2026, 5, 29),
        db_path=state,
        universe=syms,
    )
    # No bracket orders went out — first submit_order raises TradingDisabledError,
    # which the runner catches and surfaces via skipped_reason.
    assert len(fake.submitted_requests) == 0
    assert result.skipped_reason is not None


# ---------------------------------------------------------------------------- CLI guarantees


def test_kill_switch_cli_module_is_importable_and_callable() -> None:
    """The single-command guarantee: ``python -m casino.execution.kill_switch`` lives."""
    from casino.execution import kill_switch

    assert hasattr(kill_switch, "main")
    assert callable(kill_switch.main)
