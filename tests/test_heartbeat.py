"""Tests for casino.jobs.heartbeat."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from casino.config import get_config
from casino.data import store
from casino.execution import book, paper_clock
from casino.execution.alpaca_broker import AlpacaBroker, BrokerAccount
from casino.jobs import heartbeat
from casino.monitoring import alerts
from tests.test_risk import FakeTradingClient


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    duck = tmp_path / "casino.duckdb"
    state = tmp_path / "state.sqlite"
    monkeypatch.setenv("CASINO_DUCKDB_PATH", str(duck))
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(state))
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    get_config.cache_clear()
    store.create_schema(db_path=duck)
    book.init_schema(state)
    paper_clock.init_schema(db_path=state)
    return duck, state


def _seed_recent_ohlcv(duck: Path, *, latest: datetime) -> None:
    """Seed one-bar-per-ticker rows so the freshness query has data."""
    rows = [
        {
            "ticker": t,
            "ts": latest,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1_000_000,
            "adj_close": 100.0,
        }
        for t in heartbeat.TSMOM_UNIVERSE
    ]
    store.upsert_ohlcv(rows, db_path=duck)


def _broker(*, equity: Decimal = Decimal("100000")) -> AlpacaBroker:
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
    fake = FakeTradingClient(account=account, positions=[])
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    broker.set_client(fake)
    return broker


def _capture() -> tuple[list[dict[str, Any]], alerts.WebhookTransport]:
    payloads: list[dict[str, Any]] = []

    def tx(_url: str, payload: dict[str, Any]) -> httpx.Response:
        payloads.append(payload)
        return httpx.Response(204)

    return payloads, tx


def test_pre_clock_heartbeat_is_info_with_clock_not_started(isolated: tuple[Path, Path]) -> None:
    duck, state = isolated
    today = datetime(2026, 5, 7, tzinfo=UTC)
    _seed_recent_ohlcv(duck, latest=today)

    summary = heartbeat.build_summary(
        broker=_broker(),
        db_path=state,
        duckdb_path=duck,
        as_of=today,
    )

    assert summary.has_clock is False
    assert summary.severity == "info"
    assert summary.fields["30-day paper run"].startswith("not started yet")
    assert summary.fields["Latest market data"] == "2026-05-07"
    assert summary.ohlcv_gap_days == 0
    assert summary.n_kill_events_today == 0


def test_stale_ohlcv_promotes_to_warning(isolated: tuple[Path, Path]) -> None:
    duck, state = isolated
    today = datetime(2026, 5, 7, tzinfo=UTC)
    # Latest bar is 7 calendar days behind -> over OHLCV_STALE_DAYS (5).
    _seed_recent_ohlcv(duck, latest=today - timedelta(days=7))

    summary = heartbeat.build_summary(
        broker=_broker(),
        db_path=state,
        duckdb_path=duck,
        as_of=today,
    )

    assert summary.severity == "warning"
    assert summary.ohlcv_gap_days == 7
    assert "stale" in summary.message.lower()


def test_kill_event_today_promotes_to_critical(isolated: tuple[Path, Path]) -> None:
    duck, state = isolated
    # Use real UTC now so the kill_event timestamp the production code
    # stamps with _utc_now() falls on the same UTC date as our as_of.
    now = datetime.now(tz=UTC)
    _seed_recent_ohlcv(duck, latest=now)
    paper_clock.ensure_started(
        run_id=paper_clock.DEFAULT_RUN_ID,
        strategy="tsmom_long_only",
        start_nav=Decimal("100000"),
        config_json="{}",
        db_path=state,
    )
    paper_clock.insert_kill_event(
        criterion="reconcile_drift",
        value=Decimal("0.02"),
        threshold=Decimal("0.01"),
        nav_at_kill=Decimal("100000"),
        detail="seeded",
        run_id=paper_clock.DEFAULT_RUN_ID,
        db_path=state,
    )

    summary = heartbeat.build_summary(
        broker=_broker(),
        db_path=state,
        duckdb_path=duck,
        as_of=now,
    )

    assert summary.severity == "critical"
    assert summary.n_kill_events_today == 1
    assert "halted trading" in summary.message.lower()


def test_run_heartbeat_dispatches_one_embed(isolated: tuple[Path, Path]) -> None:
    duck, state = isolated
    today = datetime(2026, 5, 7, tzinfo=UTC)
    _seed_recent_ohlcv(duck, latest=today)
    payloads, tx = _capture()

    summary, result = heartbeat.run_heartbeat(
        broker=_broker(),
        db_path=state,
        duckdb_path=duck,
        transport=tx,
        as_of=today,
    )

    assert result.sent is True
    assert len(payloads) == 1
    embed = payloads[0]["embeds"][0]
    assert "Daily status" in embed["title"]
    assert summary.severity == "info"
