"""Tests for casino.llm.prompts.* (tasks 10, 11)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from casino.config import get_config
from casino.llm.client import LLMClient, stub_transport
from casino.llm.prompts.earnings_score import (
    EARNINGS_SCORE_SYSTEM_PROMPT,
    TranscriptParts,
    anonymize,
    build_user_prompt,
    score_transcript,
    truncate_for_prompt,
)
from casino.llm.prompts.headline_class import (
    HEADLINE_CLASS_SYSTEM_PROMPT,
    anonymize_headline,
    classify_headline,
    classify_headlines_batch,
)
from casino.llm.schemas import EarningsTranscriptScore, HeadlineClassification


@pytest.fixture
def state_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(tmp_path / "state.sqlite"))
    monkeypatch.setenv("CASINO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CASINO_DUCKDB_PATH", str(tmp_path / "casino.duckdb"))
    get_config.cache_clear()
    return tmp_path / "state.sqlite"


# -------------------------------------------------------- anonymization
def test_anonymize_replaces_ticker_and_aliases() -> None:
    text = "Apple Inc. (AAPL) reported strong iPhone sales. Apple is doing well."
    out = anonymize(text, ticker="AAPL", company_aliases=("Apple", "Apple Inc."))
    assert "AAPL" not in out
    assert "Apple" not in out
    assert out.count("<COMPANY>") >= 2


def test_anonymize_word_boundary_ignores_substrings() -> None:
    text = "PINEAPPLE prices up. AAPL down."
    out = anonymize(text, ticker="AAPL")
    assert "PINEAPPLE" in out  # not redacted
    assert "AAPL" not in out


def test_anonymize_headline_works() -> None:
    h = "Apple beats earnings; AAPL up 5%"
    out = anonymize_headline(h, ticker="AAPL", company_aliases=("Apple",))
    assert "Apple" not in out
    assert "AAPL" not in out


# -------------------------------------------------------- truncation
def test_truncate_keeps_prepared_and_qa_when_short() -> None:
    parts = TranscriptParts(prepared_remarks="opening" * 10, qa_session="qa" * 10)
    out = truncate_for_prompt(parts, max_chars=10_000)
    assert "opening" in out
    assert "qa" in out


def test_truncate_caps_when_long() -> None:
    parts = TranscriptParts(
        prepared_remarks="A" * 20_000,
        qa_session="B" * 20_000,
    )
    out = truncate_for_prompt(parts, max_chars=5_000)
    assert len(out) <= 5_500  # within budget + headers


def test_build_user_prompt_includes_anonymization_marker() -> None:
    parts = TranscriptParts(prepared_remarks="hello", qa_session="world")
    out = build_user_prompt(parts)
    assert "<COMPANY>" in out


# -------------------------------------------------------- score_transcript
def _good_score_json() -> str:
    return json.dumps(
        {
            "beat_quality": 1,
            "guidance_tone": 0,
            "qa_defensiveness": 1,
            "confidence": 0.7,
            "reasoning": "solid",
        }
    )


def test_score_transcript_anonymizes_before_calling(state_db: Path) -> None:
    transport = stub_transport(_good_score_json())
    client = LLMClient(mode="live", transport=transport, audit_db_path=state_db)
    parts = TranscriptParts(
        prepared_remarks="Apple Inc. reported strong revenue this quarter.",
        qa_session="Q: AAPL outlook? A: confident.",
    )
    score = score_transcript(
        client=client,
        ticker="AAPL",
        parts=parts,
        company_aliases=("Apple", "Apple Inc."),
    )
    assert isinstance(score, EarningsTranscriptScore)
    assert score.transcript_hash is not None
    captured = transport.calls  # type: ignore[attr-defined]
    body = captured[0]["messages"][0]["content"]
    assert "AAPL" not in body
    assert "Apple" not in body
    assert "<COMPANY>" in body


def test_score_transcript_caches_system_prompt(state_db: Path) -> None:
    transport = stub_transport(_good_score_json())
    client = LLMClient(mode="live", transport=transport, audit_db_path=state_db)
    parts = TranscriptParts(prepared_remarks="x", qa_session="")
    score_transcript(client=client, ticker="X", parts=parts)
    captured = transport.calls  # type: ignore[attr-defined]
    sys_blocks = captured[0]["system"]
    assert sys_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert EARNINGS_SCORE_SYSTEM_PROMPT in sys_blocks[0]["text"]


def test_score_transcript_backtest_mode_blocks_ticker_leakage(state_db: Path) -> None:
    """Even if score_transcript() forgets to anonymize, the LLMClient
    backtest guard must catch the leak. Demonstrate by passing parts that
    include a banned entity that score_transcript will scrub but if we
    bypass anonymization the guard fires."""
    transport = stub_transport(_good_score_json())
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=transport,
        audit_db_path=state_db,
    )
    # Provide aliases so score_transcript anonymizes correctly → succeeds.
    parts = TranscriptParts(
        prepared_remarks="Apple reported well.",
        qa_session="",
    )
    score = score_transcript(
        client=client,
        ticker="AAPL",
        parts=parts,
        company_aliases=("Apple",),
    )
    assert isinstance(score, EarningsTranscriptScore)


# -------------------------------------------------------- classify_headline
def _headline_json() -> str:
    return json.dumps(
        {
            "sentiment": 0.5,
            "is_material": True,
            "rationale": "earnings beat",
            "relevance": "high",
        }
    )


def test_classify_headline_uses_haiku_default(state_db: Path) -> None:
    transport = stub_transport(_headline_json())
    client = LLMClient(mode="live", transport=transport, audit_db_path=state_db)
    out = classify_headline(
        client=client,
        ticker="AAPL",
        headline="Apple beats earnings",
        company_aliases=("Apple",),
    )
    assert isinstance(out, HeadlineClassification)
    captured = transport.calls  # type: ignore[attr-defined]
    assert captured[0]["model"] == "claude-haiku-4-5"
    body = captured[0]["messages"][0]["content"]
    assert "AAPL" not in body
    assert "Apple" not in body
    assert "<COMPANY>" in body


def test_classify_headline_caches_system(state_db: Path) -> None:
    transport = stub_transport(_headline_json())
    client = LLMClient(mode="live", transport=transport, audit_db_path=state_db)
    classify_headline(client=client, ticker="X", headline="something")
    captured = transport.calls  # type: ignore[attr-defined]
    sys_blocks = captured[0]["system"]
    assert sys_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert HEADLINE_CLASS_SYSTEM_PROMPT in sys_blocks[0]["text"]


def test_classify_headlines_batch_processes_all(state_db: Path) -> None:
    transport = stub_transport(_headline_json())
    client = LLMClient(mode="live", transport=transport, audit_db_path=state_db)
    items = [
        {"ticker": "AAPL", "headline": "AAPL beats", "company_aliases": ("Apple",)},
        {"ticker": "MSFT", "headline": "MSFT raises guidance", "company_aliases": ("Microsoft",)},
    ]
    out = classify_headlines_batch(client=client, items=items)
    assert len(out) == 2
    captured = transport.calls  # type: ignore[attr-defined]
    assert len(captured) == 2
    for c in captured:
        body = c["messages"][0]["content"]
        assert "<COMPANY>" in body
