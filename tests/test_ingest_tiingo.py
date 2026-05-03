"""Tests for casino.data.ingest_tiingo. HTTP layer mocked via `responses`/respx-style.

We use `respx`-free pure-httpx mocks via httpx.MockTransport.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from casino.config import get_config
from casino.data import store
from casino.data.ingest_tiingo import TiingoClient, _parse_iso_to_utc


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    get_config.cache_clear()
    p = tmp_path / "casino.duckdb"
    store.create_schema(db_path=p)
    return p


def _make_client(handler: httpx.MockTransport, raw_dir: Path) -> TiingoClient:
    httpx_client = httpx.Client(transport=handler)
    return TiingoClient(api_key="test-key", raw_dir=raw_dir, client=httpx_client)


def test_fetch_ohlcv_parses_and_normalizes(tmp_path: Path) -> None:
    payload = [
        {
            "date": "2024-01-05T00:00:00.000Z",
            "open": 100.0,
            "high": 101.5,
            "low": 99.0,
            "close": 100.5,
            "volume": 12345,
            "adjClose": 100.5,
        },
        {
            "date": "2024-01-08T00:00:00.000Z",
            "open": 101.0,
            "high": 102.0,
            "low": 100.0,
            "close": 101.5,
            "volume": 22222,
            "adjClose": 101.5,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/tiingo/daily/aapl/prices" in str(request.url)
        assert request.headers.get("Authorization") == "Token test-key"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    with _make_client(transport, tmp_path / "raw") as c:
        rows = c.fetch_ohlcv("AAPL", start_date=date(2024, 1, 1), end_date=date(2024, 1, 10))

    assert len(rows) == 2
    assert rows[0]["ticker"] == "AAPL"
    assert isinstance(rows[0]["ts"], datetime)
    assert rows[0]["ts"].tzinfo is not None
    assert rows[0]["close"] == pytest.approx(100.5)
    # raw archive written
    raw_files = list((tmp_path / "raw" / "ohlcv").glob("AAPL_*.json"))
    assert raw_files, "raw archive json should be written"


def test_fetch_news(tmp_path: Path) -> None:
    payload = [
        {
            "id": 42,
            "title": "Apple beats EPS",
            "publishedDate": "2024-05-01T13:00:00.000Z",
            "source": "Reuters",
            "url": "https://example.com/n42",
        }
    ]
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    with _make_client(transport, tmp_path / "raw") as c:
        rows = c.fetch_news("AAPL")
    assert rows[0]["id"] == "42"
    assert rows[0]["headline"] == "Apple beats EPS"
    assert rows[0]["ticker"] == "AAPL"


def test_fetch_ohlcv_retries_on_429(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429, json={"error": "rate"})
        return httpx.Response(200, json=[])

    monkeypatch.setattr("casino.data.ingest_tiingo.time.sleep", lambda _s: None)
    transport = httpx.MockTransport(handler)
    with _make_client(transport, tmp_path / "raw") as c:
        rows = c.fetch_ohlcv("AAPL", days=5)
    assert rows == []
    assert calls["n"] == 2


def test_fetch_ohlcv_5xx_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={})
        return httpx.Response(200, json=[])

    monkeypatch.setattr("casino.data.ingest_tiingo.time.sleep", lambda _s: None)
    transport = httpx.MockTransport(handler)
    with _make_client(transport, tmp_path / "raw") as c:
        c.fetch_ohlcv("AAPL", days=5)
    assert calls["n"] == 2


def test_ingest_round_trips_to_store(tmp_path: Path, db_path: Path) -> None:
    payload: list[dict[str, Any]] = [
        {
            "date": "2024-01-05T00:00:00.000Z",
            "open": 100.0,
            "high": 101.5,
            "low": 99.0,
            "close": 100.5,
            "volume": 12345,
            "adjClose": 100.5,
        },
    ]
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    with _make_client(transport, tmp_path / "raw") as c:
        rows = c.fetch_ohlcv("AAPL", days=5)
    n = store.upsert_ohlcv(rows, db_path=db_path)
    assert n == 1
    # idempotent
    n2 = store.upsert_ohlcv(rows, db_path=db_path)
    assert n2 == 1
    with store.get_duckdb_conn(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE ticker='AAPL'").fetchone()
    assert count is not None
    assert count[0] == 1  # dedup by PK


def test_parse_iso_to_utc_handles_z_suffix() -> None:
    dt = _parse_iso_to_utc("2024-01-05T00:00:00.000Z")
    assert dt.tzinfo == UTC
    assert dt.year == 2024 and dt.month == 1 and dt.day == 5
