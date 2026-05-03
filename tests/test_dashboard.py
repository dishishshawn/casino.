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
