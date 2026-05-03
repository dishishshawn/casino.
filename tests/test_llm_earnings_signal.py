"""Tests for casino.signals.llm_earnings (task 13)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from casino.config import get_config
from casino.data import store
from casino.llm.client import LLMClient, stub_transport
from casino.llm.prompts.earnings_score import TranscriptParts
from casino.signals import llm_earnings
from casino.signals.llm_earnings import (
    clear_llm_cache,
    combined_earnings_signal,
    combined_score,
    normalize_sue,
    passes_trade_filter,
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CASINO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CASINO_DUCKDB_PATH", str(tmp_path / "casino.duckdb"))
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(tmp_path / "state.sqlite"))
    get_config.cache_clear()
    db = tmp_path / "casino.duckdb"
    store.create_schema(db_path=db)
    clear_llm_cache()
    return db


def _seed_history(db: Path, ticker: str, n: int = 8) -> None:
    """Seed lookback history so SUE doesn't fall back to industry std."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "ticker": ticker,
                "report_date": datetime(2023, 1 + i, 15, tzinfo=UTC),
                "period_end": datetime(2023, 1 + i, 15, tzinfo=UTC),
                "actual_eps": 1.00 + 0.01 * i,
                "consensus_eps": 1.00,
                "revenue": 1_000_000.0,
                "source": "TEST",
            }
        )
    store.upsert_earnings(rows, db_path=db)


def _good_score_payload(beat: int = 1, guidance: int = 1, qa: int = 1) -> str:
    return json.dumps(
        {
            "beat_quality": beat,
            "guidance_tone": guidance,
            "qa_defensiveness": qa,
            "confidence": 0.8,
            "reasoning": "ok",
        }
    )


# ---------------------------------------------------------- pure helpers
def test_normalize_sue_default_divisor() -> None:
    assert normalize_sue(6.0) == 2.0
    assert normalize_sue(-3.0) == -1.0


def test_combined_score_50_50_weighting() -> None:
    assert combined_score(sue=2.0, llm_composite=0.0) == pytest.approx(1.0)
    assert combined_score(sue=0.0, llm_composite=1.0) == pytest.approx(0.5)
    assert combined_score(sue=1.0, llm_composite=1.0) == pytest.approx(1.0)


def test_trade_filter_requires_sign_agreement() -> None:
    assert passes_trade_filter(sue=1.5, llm_composite=0.5) is True
    assert passes_trade_filter(sue=1.5, llm_composite=-0.5) is False
    assert passes_trade_filter(sue=-1.5, llm_composite=-0.5) is True


def test_trade_filter_requires_both_thresholds() -> None:
    assert passes_trade_filter(sue=0.5, llm_composite=0.5) is False  # SUE too small
    assert passes_trade_filter(sue=1.5, llm_composite=0.1) is False  # LLM too small


def test_trade_filter_returns_false_when_sue_missing() -> None:
    assert passes_trade_filter(sue=None, llm_composite=0.9) is False


def test_trade_filter_zero_either_side_blocks() -> None:
    assert passes_trade_filter(sue=0.0, llm_composite=0.5) is False
    assert passes_trade_filter(sue=0.5, llm_composite=0.0) is False


# ---------------------------------------------------- combined signal
def test_combined_signal_strong_beat_and_positive_transcript_traded(env: Path) -> None:
    _seed_history(env, "AAPL")
    transport = stub_transport(_good_score_payload(2, 2, 2))  # composite = 1.0
    client = LLMClient(
        mode="live",
        transport=transport,
        audit_db_path=env.parent / "state.sqlite",
    )
    sig = combined_earnings_signal(
        client=client,
        ticker="AAPL",
        actual_eps=Decimal("2.0"),  # massive beat
        consensus_eps=Decimal("1.0"),
        transcript_parts=TranscriptParts(prepared_remarks="strong quarter", qa_session="q"),
        as_of_date=datetime(2024, 1, 1, tzinfo=UTC),
        company_aliases=("Apple",),
        db_path=env,
    )
    assert sig.sue is not None and sig.sue > 0
    assert sig.llm_composite == pytest.approx(1.0)
    assert sig.combined > 0
    assert sig.traded is True


def test_combined_signal_mixed_signs_not_traded(env: Path) -> None:
    _seed_history(env, "AAPL")
    transport = stub_transport(_good_score_payload(-2, -2, -2))  # composite = -1.0
    client = LLMClient(
        mode="live",
        transport=transport,
        audit_db_path=env.parent / "state.sqlite",
    )
    sig = combined_earnings_signal(
        client=client,
        ticker="AAPL",
        actual_eps=Decimal("2.0"),
        consensus_eps=Decimal("1.0"),  # SUE positive
        transcript_parts=TranscriptParts(prepared_remarks="x", qa_session=""),
        as_of_date=datetime(2024, 1, 1, tzinfo=UTC),
        db_path=env,
    )
    assert sig.sue is not None and sig.sue > 0
    assert sig.llm_composite < 0
    assert sig.traded is False


def test_combined_signal_caches_by_transcript_hash(env: Path) -> None:
    _seed_history(env, "AAPL")
    transport = stub_transport(_good_score_payload())
    client = LLMClient(
        mode="live",
        transport=transport,
        audit_db_path=env.parent / "state.sqlite",
    )
    parts = TranscriptParts(prepared_remarks="same body", qa_session="same qa")
    combined_earnings_signal(
        client=client,
        ticker="AAPL",
        actual_eps=Decimal("1.5"),
        consensus_eps=Decimal("1.0"),
        transcript_parts=parts,
        as_of_date=datetime(2024, 1, 1, tzinfo=UTC),
        db_path=env,
    )
    n_first = len(transport.calls)  # type: ignore[attr-defined]
    combined_earnings_signal(
        client=client,
        ticker="AAPL",
        actual_eps=Decimal("1.5"),
        consensus_eps=Decimal("1.0"),
        transcript_parts=parts,
        as_of_date=datetime(2024, 1, 1, tzinfo=UTC),
        db_path=env,
    )
    n_second = len(transport.calls)  # type: ignore[attr-defined]
    # Second call hits the cache → no new transport call.
    assert n_second == n_first


def test_combined_signal_no_consensus_returns_no_trade(env: Path) -> None:
    _seed_history(env, "AAPL")
    transport = stub_transport(_good_score_payload(2, 2, 2))
    client = LLMClient(
        mode="live",
        transport=transport,
        audit_db_path=env.parent / "state.sqlite",
    )
    sig = combined_earnings_signal(
        client=client,
        ticker="AAPL",
        actual_eps=Decimal("1.5"),
        consensus_eps=None,
        transcript_parts=TranscriptParts(prepared_remarks="x", qa_session=""),
        as_of_date=datetime(2024, 1, 1, tzinfo=UTC),
        db_path=env,
    )
    assert sig.sue is None
    assert sig.traded is False


def test_combined_signal_batch_path_anonymizes_every_call(env: Path) -> None:
    _seed_history(env, "AAPL")
    _seed_history(env, "MSFT")
    transport = stub_transport(_good_score_payload())
    client = LLMClient(
        mode="live",
        transport=transport,
        audit_db_path=env.parent / "state.sqlite",
    )
    items = [
        {
            "ticker": "AAPL",
            "actual_eps": Decimal("1.5"),
            "consensus_eps": Decimal("1.0"),
            "transcript_parts": TranscriptParts(
                prepared_remarks="Apple strong quarter",
                qa_session="",
            ),
            "as_of_date": datetime(2024, 1, 1, tzinfo=UTC),
            "company_aliases": ("Apple",),
        },
        {
            "ticker": "MSFT",
            "actual_eps": Decimal("2.5"),
            "consensus_eps": Decimal("2.0"),
            "transcript_parts": TranscriptParts(
                prepared_remarks="Microsoft strong cloud quarter",
                qa_session="",
            ),
            "as_of_date": datetime(2024, 1, 1, tzinfo=UTC),
            "company_aliases": ("Microsoft",),
        },
    ]
    out = llm_earnings.combined_earnings_signal_batch(
        client=client,
        items=items,
        db_path=env,
    )
    assert len(out) == 2
    captured = transport.calls  # type: ignore[attr-defined]
    for c in captured:
        body = c["messages"][0]["content"]
        assert "Apple" not in body
        assert "Microsoft" not in body
        assert "<COMPANY>" in body
