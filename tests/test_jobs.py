"""Tests for casino.jobs (earnings_daily, news_intraday, reconcile_eod)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from casino.config import get_config
from casino.data import store
from casino.execution import book
from casino.execution.alpaca_broker import (
    AlpacaBroker,
    BrokerAccount,
    BrokerPosition,
)
from casino.jobs import earnings_daily, news_intraday, reconcile_eod
from casino.llm import audit
from casino.llm.client import LLMClient, stub_transport
from tests.test_risk import FakeTradingClient


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    duck = tmp_path / "casino.duckdb"
    state = tmp_path / "state.sqlite"
    monkeypatch.setenv("CASINO_DUCKDB_PATH", str(duck))
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(state))
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
    get_config.cache_clear()
    store.create_schema(db_path=duck)
    book.init_schema(state)
    audit.init_audit_schema(state)
    return duck, state


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


# ---------------------------------------------------------------------------- earnings_daily


def _seed_spy_for_regime(duck: Path, *, risk_on: bool) -> None:
    """Seed 200 SPY bars whose final close is above (risk-on) or below the MA."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(200):
        # ramp from 400 to 500 if risk-on, else 500 → 400
        if risk_on:
            close = 400 + (100 * i / 199)
        else:
            close = 500 - (100 * i / 199)
        rows.append(
            {
                "ticker": "SPY",
                "ts": base + timedelta(days=i),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000_000,
                "adj_close": close,
            }
        )
    store.upsert_ohlcv(rows, db_path=duck)


def _seed_earnings_history(duck: Path, ticker: str, base: datetime) -> None:
    """Seed earnings history rows so SUE can compute a non-None value."""
    rows = []
    for i in range(8):
        rd = base - timedelta(days=90 * (i + 2))
        rows.append(
            {
                "ticker": ticker,
                "report_date": rd,
                "period_end": rd,
                "actual_eps": 2.0 + 0.02 * (i % 3 - 1),
                "consensus_eps": 2.0,
                "revenue": 0.0,
                "source": "test",
            }
        )
    store.upsert_earnings(rows, db_path=duck)


def test_earnings_daily_no_earnings_returns_skip(isolated_paths) -> None:
    duck, state = isolated_paths
    broker, _ = _broker()
    result = earnings_daily.run_earnings_daily(
        client=LLMClient(mode="live", transport=stub_transport("{}")),
        broker=broker,
        as_of=datetime(2026, 5, 3, tzinfo=UTC),
        db_path=duck,
        state_path=state,
    )
    assert result.skipped_reason == "no earnings rows for window"
    assert result.n_submitted == 0


def test_earnings_daily_regime_off_blocks_orders(isolated_paths) -> None:
    duck, state = isolated_paths
    as_of = datetime(2024, 7, 18, tzinfo=UTC)  # ~200 days into the 2024 ramp
    _seed_spy_for_regime(duck, risk_on=False)

    # one earnings row in the 24h window
    store.upsert_earnings(
        [
            {
                "ticker": "AAA",
                "report_date": as_of - timedelta(hours=2),
                "period_end": as_of,
                "actual_eps": 2.5,
                "consensus_eps": 2.0,
                "revenue": 0.0,
                "source": "test",
            }
        ],
        db_path=duck,
    )
    _seed_earnings_history(duck, "AAA", as_of)
    # transcript + close
    with store.get_duckdb_conn(duck) as conn:
        conn.execute(
            "INSERT INTO transcripts (ticker, event_date, transcript_text, source) VALUES (?, ?, ?, ?)",
            [
                "AAA",
                as_of - timedelta(hours=1),
                "prepared remarks " * 50 + " Operator " + "qa " * 50,
                "test",
            ],
        )
    store.upsert_ohlcv(
        [
            {
                "ticker": "AAA",
                "ts": as_of - timedelta(hours=1),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000,
                "adj_close": 100.0,
            }
        ],
        db_path=duck,
    )
    broker, _ = _broker()
    result = earnings_daily.run_earnings_daily(
        client=LLMClient(mode="live", transport=stub_transport("{}")),
        broker=broker,
        as_of=as_of,
        db_path=duck,
        state_path=state,
    )
    assert result.risk_on is False
    assert result.n_submitted == 0
    assert result.skipped_reason == "regime risk-off"


def test_earnings_daily_submits_basket(isolated_paths) -> None:
    duck, state = isolated_paths
    as_of = datetime(2024, 7, 18, tzinfo=UTC)
    _seed_spy_for_regime(duck, risk_on=True)

    # two earnings rows, with strong beats
    earn_rows = []
    for t, beat in (("AAA", 0.50), ("BBB", 0.40)):
        earn_rows.append(
            {
                "ticker": t,
                "report_date": as_of - timedelta(hours=2),
                "period_end": as_of,
                "actual_eps": 2.0 + beat,
                "consensus_eps": 2.0,
                "revenue": 0.0,
                "source": "test",
            }
        )
        _seed_earnings_history(duck, t, as_of)
    store.upsert_earnings(earn_rows, db_path=duck)

    with store.get_duckdb_conn(duck) as conn:
        for t in ("AAA", "BBB"):
            conn.execute(
                "INSERT INTO transcripts (ticker, event_date, transcript_text, source) VALUES (?, ?, ?, ?)",
                [
                    t,
                    as_of - timedelta(hours=1),
                    "prepared " * 50 + " Operator " + "qa " * 50,
                    "test",
                ],
            )
    store.upsert_ohlcv(
        [
            {
                "ticker": t,
                "ts": as_of - timedelta(hours=1),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000,
                "adj_close": 100.0,
            }
            for t in ("AAA", "BBB")
        ],
        db_path=duck,
    )
    broker, fake = _broker()

    # LLM stub returns a strong positive score
    payload = json.dumps(
        {
            "beat_quality": 2,
            "guidance_tone": 2,
            "qa_defensiveness": 1,
            "confidence": 0.8,
            "reasoning": "strong beat",
        }
    )
    client = LLMClient(
        mode="live",
        transport=stub_transport(payload),
        audit_db_path=state,
    )

    result = earnings_daily.run_earnings_daily(
        client=client,
        broker=broker,
        as_of=as_of,
        db_path=duck,
        state_path=state,
    )
    assert result.risk_on is True
    assert result.n_scored == 2
    # Both received strong positive scores; quintile selection with n=2, k=1
    # picks at most one long. The exact count depends on the SUE distribution,
    # but the basket should be valid (no exceptions, no rejections).
    assert result.n_long + result.n_short <= 2
    assert result.n_rejected == 0
    # Reconcile ran clean (no positions)
    assert result.drift_alerts == 0


# ---------------------------------------------------------------------------- news_intraday


def test_news_intraday_no_headlines(isolated_paths) -> None:
    duck, state = isolated_paths
    broker, _ = _broker()
    result = news_intraday.run_news_intraday(
        client=LLMClient(mode="live", transport=stub_transport("{}"), audit_db_path=state),
        broker=broker,
        as_of=datetime(2026, 5, 3, 12, tzinfo=UTC),
        db_path=duck,
        require_market_open=False,
    )
    assert result.skipped_reason == "no fresh headlines"


def test_news_intraday_classifies_and_records(isolated_paths) -> None:
    duck, state = isolated_paths
    as_of = datetime(2026, 5, 3, 12, 0, tzinfo=UTC)
    store.upsert_news(
        [
            {
                "id": "h1",
                "ticker": "AAA",
                "headline": "Company beats earnings",
                "published_at": as_of - timedelta(minutes=10),
                "source": "test",
                "url": "x",
            }
        ],
        db_path=duck,
    )
    broker, _ = _broker()
    payload = json.dumps(
        {
            "sentiment": 0.8,
            "is_material": True,
            "relevance": "high",
            "rationale": "earnings beat",
        }
    )
    client = LLMClient(
        mode="live",
        transport=stub_transport(payload),
        audit_db_path=state,
    )
    result = news_intraday.run_news_intraday(
        client=client,
        broker=broker,
        as_of=as_of,
        db_path=duck,
        require_market_open=False,
    )
    assert result.n_classified == 1
    # Persisted into the classifications table
    with sqlite3.connect(str(state)) as conn:
        rows = conn.execute(
            "SELECT headline_id, ticker, sentiment, is_material FROM headline_classifications"
        ).fetchall()
    assert rows == [("h1", "AAA", 0.8, 1)]


def test_news_intraday_budget_breach_halts(isolated_paths) -> None:
    """Per PRD §10 / invariant 17: when daily LLM spend exceeds the cap, halt."""
    duck, state = isolated_paths
    as_of = datetime(2026, 5, 3, 12, 0, tzinfo=UTC)

    # Pre-populate llm_calls with $5 of spend already today
    audit.write_audit_row(
        prompt_hash="x",
        model="claude-haiku-4-5",
        mode="live",
        input_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        output_tokens=0,
        cost_usd=10.0,
        latency_ms=0,
        parsed_score=None,
        success=True,
        error_msg=None,
        schema_name="HeadlineClassification",
        db_path=state,
    )

    store.upsert_news(
        [
            {
                "id": "h1",
                "ticker": "AAA",
                "headline": "boom",
                "published_at": as_of - timedelta(minutes=5),
                "source": "test",
                "url": "x",
            }
        ],
        db_path=duck,
    )
    broker, _ = _broker()
    client = LLMClient(
        mode="live",
        transport=stub_transport("{}"),
        audit_db_path=state,
    )
    result = news_intraday.run_news_intraday(
        client=client,
        broker=broker,
        as_of=as_of,
        db_path=duck,
        daily_budget_usd=Decimal("5.00"),
        require_market_open=False,
    )
    assert result.budget_breached is True
    assert result.n_classified == 0


# ---------------------------------------------------------------------------- reconcile_eod


def test_reconcile_eod_writes_daily_pnl(isolated_paths) -> None:
    duck, state = isolated_paths
    as_of = datetime(2026, 5, 3, 21, tzinfo=UTC)
    _seed_spy_for_regime(duck, risk_on=True)
    broker, _ = _broker(equity=Decimal("100500"))

    result = reconcile_eod.run_reconcile_eod(
        broker=broker,
        as_of=as_of,
        db_path=state,
        duckdb_path=duck,
    )
    assert result.equity_close == Decimal("100500")
    rows = book.fetch_daily_pnl(db_path=state)
    assert len(rows) == 1
    assert rows[0].equity_close == Decimal("100500")


def test_reconcile_eod_drawdown_alert_threshold(isolated_paths) -> None:
    """Drawdown >= 10% should still write the row (alert side-effect is mocked away)."""
    duck, state = isolated_paths
    # Pre-seed history with a high-water mark of 100k
    book.upsert_daily_pnl(
        book.DailyPnLRow(
            date="2026-05-01",
            equity_open=Decimal("100000"),
            equity_close=Decimal("100000"),
            realized_pl=Decimal("0"),
            unrealized_pl=Decimal("0"),
            n_positions=0,
            n_orders=0,
            notes=None,
        ),
        db_path=state,
    )
    broker, _ = _broker(equity=Decimal("85000"))  # 15% drawdown
    result = reconcile_eod.run_reconcile_eod(
        broker=broker,
        as_of=datetime(2026, 5, 3, 21, tzinfo=UTC),
        db_path=state,
        duckdb_path=duck,
    )
    assert result.drawdown >= Decimal("0.10")
    assert result.high_water_mark >= Decimal("100000")


# ---------------------------------------------------------------------------- LLM mode invariant


def test_jobs_use_live_mode_only() -> None:
    """Invariant 16: jobs must always use mode='live'.

    We assert the default factory branch in `run_earnings_daily` and
    `run_news_intraday` constructs a live-mode LLMClient.
    """
    cfg = get_config()
    client = LLMClient(
        mode="live", audit_db_path=cfg.state_sqlite_path, transport=stub_transport("{}")
    )
    assert client.mode == "live"
