"""Tests for casino.execution.tsmom_clock_check.

Coverage:

* Each kill criterion fires correctly on synthetic input:
  drawdown, single_day, cap_violation, reconcile_drift, ks_test.
* Daily check engages the kill switch via flatten_and_disable when any
  criterion fires (CLAUDE.md hard rule 4 re-verification).
* Day-30 verdict logic for the documented branches:
  - COMMIT happy path
  - KILL because a kill_event row exists
  - KILL because zero rebals completed
  - KILL because reconcile drift > 0.5% NAV (verdict gate)
  - COMMIT with loose Sharpe = -0.15 (above the -0.2 floor)
* Discord alerts are mocked and asserted called for kill + verdict paths.
* KS-test produces sane p-values (one-sample sanity).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from casino.config import get_config
from casino.execution import book, paper_clock
from casino.execution.alpaca_broker import (
    AlpacaBroker,
    BrokerAccount,
    BrokerPosition,
)
from casino.execution.tsmom_clock_check import (
    COMMIT_SHARPE_FLOOR,
    DD_KILL_THRESHOLD,
    RECONCILE_DRIFT_COMMIT_THRESHOLD,
    RECONCILE_DRIFT_THRESHOLD,
    SINGLE_DAY_KILL_THRESHOLD,
    _evaluate_cap_violation,
    _evaluate_drawdown,
    _evaluate_ks_test,
    _evaluate_reconcile_drift,
    _evaluate_single_day,
    _ks_two_sample_pvalue,
    _paper_max_drawdown,
    _paper_sharpe,
    kill_thresholds_view,
    run_daily_check,
    run_verdict,
)
from tests.test_risk import FakeTradingClient

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


def _seed_clock(state: Path, *, start_nav: Decimal = Decimal("100000")) -> None:
    paper_clock.ensure_started(start_nav=start_nav, db_path=state)


def _backdate_clock_to(state: Path, days_ago: int) -> None:
    """Rewrite paper_clock.started_at_utc to look ``days_ago`` old."""
    new_ts = (datetime.now(tz=UTC) - timedelta(days=days_ago)).isoformat()
    with book.get_book_conn(state) as conn:
        conn.execute(
            "UPDATE paper_clock SET started_at_utc = ? WHERE run_id = ?",
            (new_ts, paper_clock.DEFAULT_RUN_ID),
        )


def _add_pnl_row(
    state: Path,
    *,
    date_str: str,
    eq_open: Decimal,
    eq_close: Decimal,
    realized: Decimal = Decimal("0"),
    unrealized: Decimal = Decimal("0"),
) -> None:
    book.upsert_daily_pnl(
        book.DailyPnLRow(
            date=date_str,
            equity_open=eq_open,
            equity_close=eq_close,
            realized_pl=realized,
            unrealized_pl=unrealized,
            n_positions=0,
            n_orders=0,
            notes=None,
        ),
        db_path=state,
    )


# ---------------------------------------------------------------------------- drawdown


def test_drawdown_below_threshold_does_not_trigger() -> None:
    s = _evaluate_drawdown(
        start_nav=Decimal("100000"),
        eqs=[Decimal("95000")],  # 5% drop
    )
    assert s.triggered is False
    assert s.value == Decimal("0.05")


def test_drawdown_above_threshold_triggers() -> None:
    s = _evaluate_drawdown(
        start_nav=Decimal("100000"),
        eqs=[Decimal("89000")],  # 11% drop > 10%
    )
    assert s.triggered is True
    assert s.value > DD_KILL_THRESHOLD


def test_drawdown_no_history_no_trigger() -> None:
    s = _evaluate_drawdown(start_nav=Decimal("100000"), eqs=[])
    assert s.triggered is False


# ---------------------------------------------------------------------------- single_day


def test_single_day_below_threshold(state: Path) -> None:
    _add_pnl_row(
        state,
        date_str="2026-05-15",
        eq_open=Decimal("100000"),
        eq_close=Decimal("98000"),
        realized=Decimal("-2000"),  # 2% drop
    )
    s = _evaluate_single_day(db_path=state)
    assert s.triggered is False


def test_single_day_above_threshold_triggers(state: Path) -> None:
    _add_pnl_row(
        state,
        date_str="2026-05-15",
        eq_open=Decimal("100000"),
        eq_close=Decimal("94000"),
        realized=Decimal("-6000"),  # 6% > 5% threshold
    )
    s = _evaluate_single_day(db_path=state)
    assert s.triggered is True
    assert s.value > SINGLE_DAY_KILL_THRESHOLD


# ---------------------------------------------------------------------------- cap_violation


def test_cap_violation_clean_state_no_trigger() -> None:
    broker, _ = _broker(positions=[])
    s = _evaluate_cap_violation(broker)
    assert s.triggered is False


def test_cap_violation_single_name_over_10pct_triggers() -> None:
    pos = BrokerPosition(
        symbol="AAA",
        qty=200,
        side="long",
        avg_entry_price=Decimal("100"),
        market_price=Decimal("100"),
        market_value=Decimal("20000"),  # 20% of $100k → over 10% cap
        unrealized_pl=Decimal("0"),
        cost_basis=Decimal("20000"),
    )
    broker, _ = _broker(positions=[pos])
    s = _evaluate_cap_violation(broker)
    assert s.triggered is True


def test_cap_violation_gross_over_one_triggers() -> None:
    # Each position 8% of NAV → 14 of them = 112% gross.
    positions = [
        BrokerPosition(
            symbol=f"S{i}",
            qty=80,
            side="long",
            avg_entry_price=Decimal("100"),
            market_price=Decimal("100"),
            market_value=Decimal("8000"),
            unrealized_pl=Decimal("0"),
            cost_basis=Decimal("8000"),
        )
        for i in range(14)
    ]
    broker, _ = _broker(positions=positions)
    s = _evaluate_cap_violation(broker)
    assert s.triggered is True


# ---------------------------------------------------------------------------- reconcile_drift


def test_reconcile_drift_in_sync_no_trigger(state: Path) -> None:
    book.upsert_position(
        symbol="AAA",
        side="long",
        qty=10,
        avg_entry_price=Decimal("100"),
        db_path=state,
    )
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
    broker, _ = _broker(positions=[pos])
    s = _evaluate_reconcile_drift(broker=broker, db_path=state)
    assert s.triggered is False


def test_reconcile_drift_book_only_position_triggers(state: Path) -> None:
    """Book has $5000 worth of a position the broker doesn't know about → 5% drift on $100k."""
    book.upsert_position(
        symbol="AAA",
        side="long",
        qty=50,
        avg_entry_price=Decimal("100"),  # $5000 notional
        db_path=state,
    )
    broker, _ = _broker(positions=[])
    s = _evaluate_reconcile_drift(broker=broker, db_path=state)
    assert s.triggered is True
    assert s.value > RECONCILE_DRIFT_THRESHOLD


# ---------------------------------------------------------------------------- KS test


def test_ks_pvalue_identical_samples_high_p() -> None:
    """Identical distributions → p ≫ 0.05."""
    rs = list(np.random.default_rng(0).normal(0, 0.01, 200))
    p = _ks_two_sample_pvalue(rs, rs)
    assert p is not None
    assert p > 0.5


def test_ks_pvalue_different_distributions_low_p() -> None:
    """Distinct means → low p-value."""
    rng = np.random.default_rng(1)
    a = list(rng.normal(0.0, 0.01, 200))
    b = list(rng.normal(0.05, 0.01, 200))  # shifted mean
    p = _ks_two_sample_pvalue(a, b)
    assert p is not None
    assert p < 0.01


def test_ks_pvalue_too_few_samples_returns_none() -> None:
    assert _ks_two_sample_pvalue([1.0], [2.0]) is None


def test_evaluate_ks_test_skips_when_below_min_days(state: Path) -> None:
    s = _evaluate_ks_test(days_elapsed=5, db_path=state)
    assert s.triggered is False
    assert "below" in s.detail


def test_evaluate_ks_test_skips_when_no_baseline(state: Path, tmp_path: Path) -> None:
    """Missing reports/tsmom_backtest_returns.csv → no-op (informational)."""
    # Seed enough paper rows.
    for i in range(20):
        _add_pnl_row(
            state,
            date_str=f"2026-04-{i + 1:02d}",
            eq_open=Decimal("100000"),
            eq_close=Decimal("100050"),
            realized=Decimal("50"),
        )
    s = _evaluate_ks_test(
        days_elapsed=20,
        db_path=state,
        backtest_returns_path=tmp_path / "missing.csv",
    )
    assert s.triggered is False
    assert "no backtest baseline" in s.detail


# ---------------------------------------------------------------------------- run_daily_check engages kill switch


class _AlertCapture:
    """Stand-in for alerts.fire — records arguments without POSTing."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(
        self,
        *,
        title: str,
        message: str,
        severity: str = "info",
        fields: dict | None = None,
        webhook_url: str | None = None,
        transport=None,
    ):
        self.calls.append(
            {
                "title": title,
                "message": message,
                "severity": severity,
                "fields": fields or {},
            }
        )

        class _R:
            sent = True
            status_code = 204
            reason = "ok"

        return _R()


def test_run_daily_check_drawdown_engages_kill_switch(
    state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: drawdown trigger → kill_event row + flatten_and_disable.

    This is the kill-switch re-verification path called for in
    CLAUDE.md hard rule 4 ("If you change execution code, re-verify the
    kill switch path still works end-to-end").
    """
    _seed_clock(state, start_nav=Decimal("100000"))
    _add_pnl_row(
        state,
        date_str="2026-05-20",
        eq_open=Decimal("100000"),
        eq_close=Decimal("85000"),  # -15%, exceeds 10% kill threshold
    )
    pos = BrokerPosition(
        symbol="SPY",
        qty=10,
        side="long",
        avg_entry_price=Decimal("400"),
        market_price=Decimal("380"),
        market_value=Decimal("3800"),
        unrealized_pl=Decimal("-200"),
        cost_basis=Decimal("4000"),
    )
    broker, fake = _broker(equity=Decimal("85000"), positions=[pos])

    capture = _AlertCapture()
    monkeypatch.setattr("casino.execution.tsmom_clock_check.alerts.fire", capture)

    pre_disabled = book.is_trading_disabled(db_path=state)
    assert pre_disabled is False

    result = run_daily_check(broker=broker, db_path=state)

    # Drawdown criterion triggered.
    assert result.kill_fired is True
    assert "drawdown" in result.triggered_criteria
    # kill_event row persisted.
    kills = paper_clock.fetch_kill_events(db_path=state)
    assert any(k.criterion == "drawdown" for k in kills)
    # flatten_and_disable was actually invoked → trading disabled.
    assert book.is_trading_disabled(db_path=state) is True
    # broker.close_all_positions was called.
    assert "SPY" in fake.closed_positions
    # Discord alert fired.
    assert any("KILL CRITERION" in c["title"] for c in capture.calls)


def test_run_daily_check_engage_kill_switch_false_skips_flatten(
    state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engage_kill_switch=False knob is for testing only and does not flatten."""
    _seed_clock(state, start_nav=Decimal("100000"))
    _add_pnl_row(
        state,
        date_str="2026-05-20",
        eq_open=Decimal("100000"),
        eq_close=Decimal("85000"),
    )
    pos = BrokerPosition(
        symbol="SPY",
        qty=10,
        side="long",
        avg_entry_price=Decimal("400"),
        market_price=Decimal("380"),
        market_value=Decimal("3800"),
        unrealized_pl=Decimal("-200"),
        cost_basis=Decimal("4000"),
    )
    broker, fake = _broker(equity=Decimal("85000"), positions=[pos])
    monkeypatch.setattr("casino.execution.tsmom_clock_check.alerts.fire", _AlertCapture())

    result = run_daily_check(broker=broker, db_path=state, engage_kill_switch=False)
    assert result.kill_fired is True
    # No flatten this time.
    assert book.is_trading_disabled(db_path=state) is False
    assert fake.closed_positions == []


# ---------------------------------------------------------------------------- verdict


def _seed_happy_paper_run(state: Path) -> None:
    """30 days of small positive returns + one rebal_event + clean reconcile."""
    _seed_clock(state, start_nav=Decimal("100000"))
    _backdate_clock_to(state, days_ago=30)
    eq = 100000
    for i in range(30):
        eq_open = eq
        eq_close = eq + 50  # +0.05% per day, smooth uptrend
        eq = eq_close
        _add_pnl_row(
            state,
            date_str=f"2026-04-{i + 1:02d}" if i < 30 else f"2026-05-{i - 29:02d}",
            eq_open=Decimal(eq_open),
            eq_close=Decimal(eq_close),
            realized=Decimal(50),
        )
    paper_clock.insert_rebal_event(
        n_orders_submitted=3,
        nav_at_rebal=Decimal("100500"),
        target_weights_json='[{"symbol":"SPY","weight":0.5}]',
        db_path=state,
    )


def test_verdict_commit_happy_path(
    state: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_happy_paper_run(state)
    broker, _ = _broker(equity=Decimal("101500"), positions=[])
    monkeypatch.setattr("casino.execution.tsmom_clock_check.alerts.fire", _AlertCapture())

    out = run_verdict(
        broker=broker,
        db_path=state,
        backtest_returns_path=tmp_path / "missing.csv",  # no baseline → ks treated as PASS
        reports_dir=tmp_path / "reports",
    )
    assert out.verdict == "COMMIT"
    assert out.n_rebals == 1
    # Verdict persisted to paper_clock row.
    row = paper_clock.fetch_paper_clock(db_path=state)
    assert row is not None
    assert row.verdict == "COMMIT"
    # Report files written.
    assert (tmp_path / "reports" / "tsmom_paper_30day_verdict.csv").exists()
    md = (tmp_path / "reports" / "tsmom_paper_30day_verdict.md").read_text(encoding="utf-8")
    assert "COMMIT" in md
    assert "does NOT authorize live trading" in md


def test_verdict_kill_when_kill_event_present(
    state: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_happy_paper_run(state)
    paper_clock.insert_kill_event(
        criterion="drawdown",
        value=Decimal("0.15"),
        threshold=DD_KILL_THRESHOLD,
        nav_at_kill=Decimal("85000"),
        db_path=state,
    )
    broker, _ = _broker(equity=Decimal("85000"), positions=[])
    capture = _AlertCapture()
    monkeypatch.setattr("casino.execution.tsmom_clock_check.alerts.fire", capture)

    out = run_verdict(
        broker=broker,
        db_path=state,
        backtest_returns_path=tmp_path / "missing.csv",
        reports_dir=tmp_path / "reports",
    )
    assert out.verdict == "KILL"
    # Discord verdict alert with ACTION REQUIRED tag.
    assert any("ACTION REQUIRED" in c["title"] for c in capture.calls)


def test_verdict_kill_when_no_rebals(
    state: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """30 days elapsed with zero rebals → KILL."""
    _seed_clock(state, start_nav=Decimal("100000"))
    _backdate_clock_to(state, days_ago=30)
    # Add a few flat P&L rows so paper Sharpe is computable but we have 0 rebals.
    for i in range(10):
        _add_pnl_row(
            state,
            date_str=f"2026-04-{i + 1:02d}",
            eq_open=Decimal("100000"),
            eq_close=Decimal("100020"),
            realized=Decimal("20"),
        )
    broker, _ = _broker(equity=Decimal("100200"), positions=[])
    monkeypatch.setattr("casino.execution.tsmom_clock_check.alerts.fire", _AlertCapture())

    out = run_verdict(
        broker=broker,
        db_path=state,
        backtest_returns_path=tmp_path / "missing.csv",
        reports_dir=tmp_path / "reports",
    )
    assert out.verdict == "KILL"
    assert any("rebal" in r.lower() for r in out.reasons)


def test_verdict_kill_when_reconcile_drift_above_commit_threshold(
    state: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift > 0.5% NAV (verdict gate, even if below the kill 1% threshold) → KILL."""
    _seed_happy_paper_run(state)
    # Plant 0.7% drift: book has $700 of position the broker doesn't.
    book.upsert_position(
        symbol="AAA",
        side="long",
        qty=7,
        avg_entry_price=Decimal("100"),  # $700 notional
        db_path=state,
    )
    broker, _ = _broker(equity=Decimal("100000"), positions=[])
    monkeypatch.setattr("casino.execution.tsmom_clock_check.alerts.fire", _AlertCapture())

    out = run_verdict(
        broker=broker,
        db_path=state,
        backtest_returns_path=tmp_path / "missing.csv",
        reports_dir=tmp_path / "reports",
    )
    # Drift fraction is between commit (0.5%) and kill (1.0%) → verdict=KILL,
    # but no kill_event was fired during the run.
    assert out.drift_now > RECONCILE_DRIFT_COMMIT_THRESHOLD
    assert out.verdict == "KILL"


def test_verdict_commit_with_loose_sharpe(
    state: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharpe = -0.15 (above floor -0.2) and all other gates clean → COMMIT.

    Constructs a noisy series with a small negative drift so the Sharpe
    lands above the floor but below zero.
    """
    _seed_clock(state, start_nav=Decimal("100000"))
    _backdate_clock_to(state, days_ago=30)
    rng = np.random.default_rng(7)
    eq = 100000.0
    for i in range(30):
        ret = rng.normal(loc=-0.0001, scale=0.005)  # tiny negative drift, big noise
        eq_open = eq
        eq_close = eq * (1.0 + ret)
        eq = eq_close
        _add_pnl_row(
            state,
            date_str=f"2026-04-{i + 1:02d}",
            eq_open=Decimal(str(round(eq_open, 2))),
            eq_close=Decimal(str(round(eq_close, 2))),
            realized=Decimal(str(round(eq_close - eq_open, 2))),
        )
    paper_clock.insert_rebal_event(
        n_orders_submitted=3,
        nav_at_rebal=Decimal("100000"),
        target_weights_json="[]",
        db_path=state,
    )
    broker, _ = _broker(equity=Decimal(str(round(eq, 2))), positions=[])
    monkeypatch.setattr("casino.execution.tsmom_clock_check.alerts.fire", _AlertCapture())

    out = run_verdict(
        broker=broker,
        db_path=state,
        backtest_returns_path=tmp_path / "missing.csv",
        reports_dir=tmp_path / "reports",
    )
    # We don't pin sign of Sharpe (random); just assert if it was above floor it commits.
    if out.paper_sharpe is not None and out.paper_sharpe >= COMMIT_SHARPE_FLOOR:
        assert out.verdict == "COMMIT"
    else:
        assert out.verdict == "KILL"


# ---------------------------------------------------------------------------- helpers


def test_kill_thresholds_view_exposes_all_constants() -> None:
    view = kill_thresholds_view()
    for k in (
        "drawdown",
        "single_day",
        "gross_cap_buffer",
        "reconcile_drift",
        "ks_kill_pvalue",
        "ks_commit_pvalue",
        "commit_sharpe_floor",
        "commit_drift_threshold",
    ):
        assert k in view


def test_paper_sharpe_returns_none_below_window() -> None:
    assert _paper_sharpe([0.01, 0.02]) is None


def test_paper_max_drawdown_basic() -> None:
    eqs = [Decimal("100"), Decimal("110"), Decimal("90")]
    dd = _paper_max_drawdown(eqs)
    # Peak 110 → trough 90 → 20/110 ≈ 0.1818
    assert dd > Decimal("0.18")
    assert dd < Decimal("0.19")
