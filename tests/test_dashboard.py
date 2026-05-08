"""Tests for casino.monitoring.dashboard data-prep functions.

The Streamlit `render` function is intentionally untested — per the
Phase-3 instructions, we cover only the pure data-prep layer.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from casino.config import get_config
from casino.execution import book
from casino.execution.alpaca_broker import BrokerPosition
from casino.monitoring import dashboard


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "state.sqlite"
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(p))
    get_config.cache_clear()
    book.init_schema(p)
    return p


def _row(
    date_str: str, *, eo: int = 100000, ec: int = 100000, r: int = 0, u: int = 0
) -> book.DailyPnLRow:
    return book.DailyPnLRow(
        date=date_str,
        equity_open=Decimal(eo),
        equity_close=Decimal(ec),
        realized_pl=Decimal(r),
        unrealized_pl=Decimal(u),
        n_positions=0,
        n_orders=0,
        notes=None,
    )


def test_compute_pnl_summary_aggregates_today_mtd_ytd() -> None:
    today = date(2026, 5, 3)
    history = [
        _row("2026-05-03", ec=100500, r=300, u=200),  # today
        _row("2026-05-02", ec=100000, r=100, u=-100),
        _row("2026-04-15", ec=99800, r=-50, u=-150),  # prior month
        _row("2026-01-10", ec=99000, r=-300, u=-700),  # YTD
        _row("2025-12-15", ec=98000, r=-1000, u=0),  # prior year
    ]
    summary = dashboard.compute_pnl_summary(history, today=today)
    assert summary.today == Decimal("500")
    assert summary.mtd == Decimal("500")  # only May rows: 500 + 0
    # YTD: today (500) + 2026-05-02 (0) + 2026-04-15 (-200) + 2026-01-10 (-1000)
    assert summary.ytd == Decimal("-700")
    assert summary.equity_today == Decimal("100500")


def test_compute_pnl_drawdown_from_history() -> None:
    today = date(2026, 5, 3)
    history = [
        _row("2026-05-03", ec=90000),
        _row("2026-05-02", ec=100000),  # high-water mark
    ]
    summary = dashboard.compute_pnl_summary(history, today=today)
    assert summary.high_water_mark == Decimal("100000")
    assert summary.drawdown == Decimal("0.10")


def test_rolling_sharpe_returns_none_below_window() -> None:
    history = [_row(f"2026-01-{i:02d}", ec=100000 + i * 10, r=10) for i in range(1, 5)]
    assert dashboard.rolling_sharpe(history, window=60) is None


def test_rolling_sharpe_finite_with_full_window() -> None:
    history = [
        book.DailyPnLRow(
            date=f"2026-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
            equity_open=Decimal("100000"),
            equity_close=Decimal("100000"),
            realized_pl=Decimal(10 + (i % 5) * 7),
            unrealized_pl=Decimal("0"),
            n_positions=0,
            n_orders=0,
            notes=None,
        )
        for i in range(60)
    ]
    sharpe = dashboard.rolling_sharpe(history, window=60)
    assert sharpe is not None
    assert sharpe > 0


def test_merge_positions_in_sync(state: Path) -> None:
    book_pos = [
        book.StoredPosition(
            symbol="AAA",
            side="long",
            qty=10,
            avg_entry_price=Decimal("100"),
            opened_at_utc=__import__("datetime").datetime(
                2026, 5, 1, tzinfo=__import__("datetime").UTC
            ),
            last_update_utc=__import__("datetime").datetime(
                2026, 5, 1, tzinfo=__import__("datetime").UTC
            ),
        )
    ]
    broker_pos = [
        BrokerPosition(
            symbol="AAA",
            qty=10,
            side="long",
            avg_entry_price=Decimal("100"),
            market_price=Decimal("101"),
            market_value=Decimal("1010"),
            unrealized_pl=Decimal("10"),
            cost_basis=Decimal("1000"),
        )
    ]
    rows = dashboard.merge_positions(
        book_positions=book_pos,
        broker_positions=broker_pos,
    )
    assert len(rows) == 1
    assert rows[0].in_sync is True


def test_merge_positions_broker_only_flagged_out_of_sync() -> None:
    broker_pos = [
        BrokerPosition(
            symbol="ZZZ",
            qty=5,
            side="long",
            avg_entry_price=Decimal("20"),
            market_price=Decimal("20"),
            market_value=Decimal("100"),
            unrealized_pl=Decimal("0"),
            cost_basis=Decimal("100"),
        )
    ]
    rows = dashboard.merge_positions(
        book_positions=[],
        broker_positions=broker_pos,
    )
    assert len(rows) == 1
    assert rows[0].in_sync is False
    assert rows[0].book_qty == 0


def test_format_llm_calls_handles_naive_timestamps() -> None:
    rows = [
        {
            "timestamp_utc": "2026-05-03T12:00:00",  # naive
            "model": "claude-haiku-4-5",
            "mode": "live",
            "cost_usd": 0.001,
            "input_tokens": 100,
            "cache_read_tokens": 50,
            "cache_creation_tokens": 0,
            "output_tokens": 200,
            "latency_ms": 200,
            "success": 1,
            "schema_name": "HeadlineClassification",
        }
    ]
    out = dashboard.format_llm_calls(rows)
    assert len(out) == 1
    assert "2026" in out[0].timestamp_local
    assert out[0].success is True


def test_build_snapshot_smoke(state: Path) -> None:
    """End-to-end: build a snapshot with empty book + no broker."""
    snap = dashboard.build_snapshot(broker=None, db_path=state)
    assert snap.pnl.today == Decimal("0")
    assert snap.positions == []
    assert snap.recent_calls == []
    assert snap.trading_disabled is False


# ---------------------------------------------------------------------------- 30-day-cap panel


def test_paper_clock_panel_awaiting_first_rebal(state: Path) -> None:
    """Before the runner has been invoked, the panel reports ``started=False``."""
    panel = dashboard.build_paper_clock_panel(broker=None, db_path=state)
    assert panel.started is False
    assert panel.days_elapsed is None
    assert panel.cap_days == 30
    html = dashboard.paper_clock_panel_html(panel)
    assert "awaiting first rebal" in html


def test_paper_clock_panel_renders_after_clock_started(state: Path) -> None:
    """After ``ensure_started`` runs, the panel surfaces day count + criteria."""
    from casino.execution import paper_clock as _pc

    _pc.ensure_started(start_nav=Decimal("100000"), db_path=state)
    panel = dashboard.build_paper_clock_panel(broker=None, db_path=state)
    assert panel.started is True
    assert panel.days_elapsed is not None
    assert panel.start_nav == Decimal("100000")
    # Drawdown / single-day / KS criteria evaluated even without a broker.
    assert any(c.name == "drawdown" for c in panel.criteria)
    assert any(c.name == "single_day" for c in panel.criteria)
    html = dashboard.paper_clock_panel_html(panel)
    assert "30-day cap" in html


def test_paper_clock_panel_drift_red_above_half_pct(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconcile drift > 0.5% of NAV must render the verdict-gate flag in red."""
    from casino.execution import paper_clock as _pc
    from casino.execution.alpaca_broker import (
        AlpacaBroker,
        BrokerAccount,
    )
    from tests.test_risk import FakeTradingClient

    _pc.ensure_started(start_nav=Decimal("100000"), db_path=state)
    # Plant a 0.7% book-only drift: $700 of position the broker doesn't see.
    book.upsert_position(
        symbol="AAA",
        side="long",
        qty=7,
        avg_entry_price=Decimal("100"),
        db_path=state,
    )
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
    fake = FakeTradingClient(account=account, positions=[])
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    broker.set_client(fake)

    panel = dashboard.build_paper_clock_panel(broker=broker, db_path=state)
    assert panel.reconcile_drift_fraction > 0.005
    assert panel.reconcile_drift_red is True


def test_paper_clock_panel_html_pending_verdict(state: Path) -> None:
    """A started clock with no verdict should render 'PENDING'."""
    from casino.execution import paper_clock as _pc

    _pc.ensure_started(start_nav=Decimal("100000"), db_path=state)
    panel = dashboard.build_paper_clock_panel(broker=None, db_path=state)
    html = dashboard.paper_clock_panel_html(panel)
    assert "PENDING" in html


def test_paper_clock_panel_html_commit_verdict(state: Path) -> None:
    from casino.execution import paper_clock as _pc

    _pc.ensure_started(start_nav=Decimal("100000"), db_path=state)
    _pc.set_verdict(verdict="COMMIT", db_path=state)
    panel = dashboard.build_paper_clock_panel(broker=None, db_path=state)
    html = dashboard.paper_clock_panel_html(panel)
    assert "COMMIT" in html
