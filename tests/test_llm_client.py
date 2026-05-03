"""Tests for casino.llm.client (task 8) and casino.llm.audit.

All tests use a stub transport — no live Anthropic calls.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from casino.config import get_config
from casino.llm import audit
from casino.llm.client import (
    BacktestLeakageError,
    LLMClient,
    Usage,
    cost_usd_for_usage,
    extract_dates,
    stub_transport,
)
from casino.llm.schemas import EarningsTranscriptScore, HeadlineClassification


# -------------------------------------------------------------------- fixtures
@pytest.fixture
def state_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    get_config.cache_clear()
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(tmp_path / "state.sqlite"))
    monkeypatch.setenv("CASINO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CASINO_DUCKDB_PATH", str(tmp_path / "casino.duckdb"))
    get_config.cache_clear()
    return tmp_path / "state.sqlite"


def _good_score_json() -> str:
    return json.dumps(
        {
            "beat_quality": 1,
            "guidance_tone": 0,
            "qa_defensiveness": 1,
            "confidence": 0.8,
            "reasoning": "solid quarter",
        }
    )


# -------------------------------------------------------------------- pricing
def test_cost_calculation_known_models() -> None:
    # Sonnet: 1M input @ $3 = $3.00; cached 1M @ $0.30; output 1M @ $15.
    cost = cost_usd_for_usage(
        "claude-sonnet-4-5",
        Usage(
            input_tokens=1_000_000, cache_creation_tokens=0, cache_read_tokens=0, output_tokens=0
        ),
    )
    assert cost == pytest.approx(3.0)
    cost = cost_usd_for_usage(
        "claude-sonnet-4-6",
        Usage(
            input_tokens=0, cache_creation_tokens=0, cache_read_tokens=1_000_000, output_tokens=0
        ),
    )
    assert cost == pytest.approx(0.30)
    # Haiku
    cost = cost_usd_for_usage(
        "claude-haiku-4-5-20251001",
        Usage(
            input_tokens=1_000_000,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            output_tokens=1_000_000,
        ),
    )
    assert cost == pytest.approx(1.0 + 5.0)


def test_unknown_model_falls_back_to_sonnet() -> None:
    cost = cost_usd_for_usage(
        "claude-novel-9000",
        Usage(
            input_tokens=1_000_000, cache_creation_tokens=0, cache_read_tokens=0, output_tokens=0
        ),
    )
    assert cost == pytest.approx(3.0)


# --------------------------------------------------------------- date extractor
def test_extract_dates_iso_and_us() -> None:
    dates = extract_dates("Reported on 2025-04-15 and again 04/16/2025 and 17 April 2025.")
    assert date(2025, 4, 15) in dates
    assert date(2025, 4, 16) in dates
    assert date(2025, 4, 17) in dates


def test_extract_dates_ignores_garbage() -> None:
    assert extract_dates("revenue grew 4.5%") == []
    # 99/99/9999 — not a real date, but pattern matches; safe_date returns None.
    assert extract_dates("99/99/9999") == []


# --------------------------------------------------------------- live mode call
def test_call_structured_live_mode_writes_audit_row(state_db: Path) -> None:
    transport = stub_transport(_good_score_json(), input_tokens=200, output_tokens=80)
    client = LLMClient(mode="live", transport=transport, audit_db_path=state_db)

    resp = client.call_structured(
        system="You are an expert.",
        user="Score this quarter for AAPL: ...",
        schema=EarningsTranscriptScore,
        model="claude-sonnet-4-5",
    )
    assert isinstance(resp.parsed, EarningsTranscriptScore)
    assert resp.parsed.beat_quality == 1
    assert resp.audit_row_id > 0
    rows = audit.fetch_recent_calls(limit=5, db_path=state_db)
    assert len(rows) == 1
    assert rows[0]["success"] == 1
    assert rows[0]["model"] == "claude-sonnet-4-5"
    assert rows[0]["input_tokens"] == 200


def test_call_structured_sets_cache_control_on_system(state_db: Path) -> None:
    transport = stub_transport(_good_score_json())
    client = LLMClient(mode="live", transport=transport, audit_db_path=state_db)
    client.call_structured(
        system="STABLE SYSTEM PROMPT",
        user="some user content",
        schema=EarningsTranscriptScore,
        model="claude-sonnet-4-5",
    )
    captured = transport.calls  # type: ignore[attr-defined]
    assert len(captured) == 1
    sys_blocks = captured[0]["system"]
    assert isinstance(sys_blocks, list)
    assert sys_blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_validation_failure_retries_then_raises(state_db: Path) -> None:
    transport = stub_transport("not json at all")
    client = LLMClient(
        mode="live",
        transport=transport,
        audit_db_path=state_db,
        validation_retries=1,
    )
    with pytest.raises(ValueError, match="schema validation"):
        client.call_structured(
            system="sys",
            user="usr",
            schema=EarningsTranscriptScore,
            model="claude-sonnet-4-5",
        )
    rows = audit.fetch_recent_calls(limit=5, db_path=state_db)
    assert rows[0]["success"] == 0
    assert rows[0]["error_msg"] is not None


def test_transport_error_retries_with_backoff(state_db: Path) -> None:
    attempts = {"n": 0}

    class _FlakyTransport:
        calls: list[dict] = []  # type: ignore[type-arg]

        def messages_create(self, **kwargs):  # type: ignore[no-untyped-def]
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("transient")
            return stub_transport(_good_score_json()).messages_create(**kwargs)  # type: ignore[attr-defined]

    client = LLMClient(
        mode="live",
        transport=_FlakyTransport(),  # type: ignore[arg-type]
        audit_db_path=state_db,
        backoff_base_sec=0.01,
    )
    resp = client.call_structured(
        system="s",
        user="u",
        schema=EarningsTranscriptScore,
        model="claude-sonnet-4-5",
    )
    assert resp.parsed.beat_quality == 1
    assert attempts["n"] == 2


# --------------------------------------------------------------- backtest mode
def test_backtest_mode_requires_window() -> None:
    with pytest.raises(ValueError, match="backtest_window"):
        LLMClient(mode="backtest")


def test_backtest_rejects_missing_anonymization(state_db: Path) -> None:
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=stub_transport(_good_score_json()),
        audit_db_path=state_db,
        banned_entities=("AAPL",),
    )
    with pytest.raises(BacktestLeakageError, match="anonymization"):
        client.call_structured(
            system="sys",
            user="please score this user",  # no <COMPANY>
            schema=EarningsTranscriptScore,
            model="claude-sonnet-4-5",
        )


def test_backtest_rejects_banned_ticker_even_if_company_marker_present(state_db: Path) -> None:
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=stub_transport(_good_score_json()),
        audit_db_path=state_db,
        banned_entities=("AAPL",),
    )
    with pytest.raises(BacktestLeakageError, match="leaks banned entity"):
        client.call_structured(
            system="sys",
            user="<COMPANY> formerly known as AAPL...",
            schema=EarningsTranscriptScore,
            model="claude-sonnet-4-5",
        )


def test_backtest_rejects_in_window_iso_date(state_db: Path) -> None:
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=stub_transport(_good_score_json()),
        audit_db_path=state_db,
    )
    with pytest.raises(BacktestLeakageError, match="in-window date"):
        client.call_structured(
            system="sys",
            user="<COMPANY> reported on 2025-04-15.",
            schema=EarningsTranscriptScore,
            model="claude-sonnet-4-5",
        )


def test_backtest_rejects_in_window_english_date(state_db: Path) -> None:
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=stub_transport(_good_score_json()),
        audit_db_path=state_db,
    )
    with pytest.raises(BacktestLeakageError):
        client.call_structured(
            system="sys",
            user="<COMPANY> reported on April 15, 2025.",
            schema=EarningsTranscriptScore,
            model="claude-sonnet-4-5",
        )


def test_backtest_accepts_pre_window_date(state_db: Path) -> None:
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=stub_transport(_good_score_json()),
        audit_db_path=state_db,
    )
    resp = client.call_structured(
        system="sys",
        user="<COMPANY> historically reported on 2024-04-15.",
        schema=EarningsTranscriptScore,
        model="claude-sonnet-4-5",
    )
    assert resp.parsed.beat_quality == 1


def test_live_mode_skips_anonymization_check(state_db: Path) -> None:
    client = LLMClient(
        mode="live",
        transport=stub_transport(_good_score_json()),
        audit_db_path=state_db,
    )
    # Live: AAPL and 2025-04-15 are both fine.
    resp = client.call_structured(
        system="sys",
        user="AAPL reported on 2025-04-15.",
        schema=EarningsTranscriptScore,
        model="claude-sonnet-4-5",
    )
    assert resp.audit_row_id > 0


# ------------------------------------------------------------------ batch path
def test_batch_dispatch_runs_each_with_audit(state_db: Path) -> None:
    transport = stub_transport(
        json.dumps(
            {
                "sentiment": 0.5,
                "is_material": True,
                "rationale": "ok",
            }
        )
    )
    client = LLMClient(mode="live", transport=transport, audit_db_path=state_db)
    items = [{"system": "s", "user": f"item {i}"} for i in range(3)]
    results = client.call_structured_batch(
        items=items,
        schema=HeadlineClassification,
        model="claude-haiku-4-5",
    )
    assert len(results) == 3
    rows = audit.fetch_recent_calls(limit=10, db_path=state_db)
    assert len(rows) == 3
    assert {r["model"] for r in rows} == {"claude-haiku-4-5"}
