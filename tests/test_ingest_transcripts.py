"""Tests for casino.data.ingest_transcripts (task 7). HTTP fully mocked."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from casino.config import get_config
from casino.data import store
from casino.data.ingest_transcripts import (
    FMPClient,
    clean_transcript_text,
    split_into_sections,
    upsert_transcripts,
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    get_config.cache_clear()
    monkeypatch.setenv("CASINO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CASINO_DUCKDB_PATH", str(tmp_path / "casino.duckdb"))
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    get_config.cache_clear()
    p = tmp_path / "casino.duckdb"
    store.create_schema(db_path=p)
    return p


# -------------------------------------------------------------- text helpers
def test_clean_transcript_strips_ads_and_normalizes() -> None:
    raw = "Q1 results\r\n\r\n\r\nadvertisement   \nfoo  \n  \nbar"
    cleaned = clean_transcript_text(raw)
    assert "advertisement" not in cleaned.lower() or cleaned.lower().count("advertisement") == 0
    assert "\n\n\n" not in cleaned


def test_split_into_sections_finds_qa() -> None:
    text = (
        "Prepared remarks: revenue grew.\n"
        "We expect strong Q4.\n\n"
        "Question and Answer\n\n"
        "Analyst: how is China? Answer: stable."
    )
    sections = split_into_sections(text)
    assert "revenue grew" in sections.prepared_remarks
    assert "Analyst" in sections.qa_session
    assert "Question and Answer" in sections.qa_session


def test_split_into_sections_falls_back_to_full_text() -> None:
    text = "No QA marker here, just remarks."
    sections = split_into_sections(text)
    assert sections.prepared_remarks == text
    assert sections.qa_session == ""


# -------------------------------------------------------------- HTTP mocking
def _make_mock_transport(
    payload: list[dict[str, Any]] | dict[str, Any],
    *,
    status: int = 200,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def test_fetch_transcript_normalizes_fmp_payload(env: Path) -> None:
    payload = [
        {
            "symbol": "AAPL",
            "quarter": 4,
            "year": 2024,
            "date": "2024-10-31",
            "content": (
                "Welcome to the call. Revenue rose 8%.\n"
                "Question and Answer\nAnalyst: outlook? CFO: strong."
            ),
        }
    ]
    transport = _make_mock_transport(payload)
    client = httpx.Client(transport=transport)
    fmp = FMPClient(api_key="test", client=client, raw_dir=env.parent / "raw", rate_limit_sec=0.0)
    rows = fmp.fetch_transcript("AAPL")
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "AAPL"
    assert isinstance(row["event_date"], datetime)
    assert row["event_date"].tzinfo is not None
    assert "Question and Answer" in row["transcript_text"]
    assert row["source"] == "FMP"


def test_fetch_transcript_handles_404_as_empty(env: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fmp = FMPClient(api_key="test", client=client, raw_dir=env.parent / "raw", rate_limit_sec=0.0)
    assert fmp.fetch_transcript("AAPL") == []


def test_fetch_transcript_retries_on_429(env: Path) -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] < 2:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(
            200,
            json=[
                {
                    "symbol": "AAPL",
                    "quarter": 1,
                    "year": 2024,
                    "date": "2024-01-31",
                    "content": "ok\n",
                }
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fmp = FMPClient(
        api_key="test",
        client=client,
        raw_dir=env.parent / "raw",
        rate_limit_sec=0.0,
    )
    rows = fmp.fetch_transcript("AAPL")
    assert state["n"] >= 2
    assert len(rows) == 1


def test_upsert_transcripts_round_trip(env: Path) -> None:
    rows = [
        {
            "ticker": "AAPL",
            "event_date": datetime(2024, 10, 31, tzinfo=UTC),
            "transcript_text": "Hello world",
            "source": "FMP",
        }
    ]
    n = upsert_transcripts(rows, db_path=env)
    assert n == 1
    with store.get_duckdb_conn(env) as conn:
        out = conn.execute("SELECT ticker, transcript_text, source FROM transcripts").fetchall()
    assert out == [("AAPL", "Hello world", "FMP")]


def test_upsert_transcripts_empty_no_op(env: Path) -> None:
    assert upsert_transcripts([], db_path=env) == 0
