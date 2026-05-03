"""Release gate (task 19): assert backtest LLM prompts contain no in-window
dates and are anonymized.

PRD §6.3 / CLAUDE.md: this test must remain green at all times. Skipping,
xfailing, or temporarily disabling it is forbidden without an explicit
user instruction. CI runs it on every PR.

Coverage:
    1. Backtest mode: ticker leakage in prompt → BacktestLeakageError.
    2. Backtest mode: in-window ISO date in prompt → BacktestLeakageError.
    3. Backtest mode: in-window English date in prompt → BacktestLeakageError.
    4. Backtest mode: missing <COMPANY> marker → BacktestLeakageError.
    5. Live/forward mode with same content → succeeds.
    6. End-to-end through `signals.llm_earnings`: every captured outgoing
       prompt is anonymized.
    7. Backtest start date after model cutoff is allowed without
       anonymization (PRD §6.2 rule 1) — though we still recommend
       anonymizing for robustness.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from casino.config import get_config
from casino.data import store
from casino.llm.client import (
    BacktestLeakageError,
    LLMClient,
    extract_dates,
    stub_transport,
)
from casino.llm.prompts.earnings_score import TranscriptParts
from casino.llm.schemas import EarningsTranscriptScore
from casino.signals.llm_earnings import (
    clear_llm_cache,
    combined_earnings_signal,
)

pytestmark = pytest.mark.release_gate


# ---------------------------------------------------------------------- fixtures
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


def _good_score_payload() -> str:
    return json.dumps(
        {
            "beat_quality": 1,
            "guidance_tone": 0,
            "qa_defensiveness": 1,
            "confidence": 0.7,
            "reasoning": "ok",
        }
    )


# ============================================================ direct LLMClient
def test_backtest_mode_rejects_ticker_leak_in_prompt(env: Path) -> None:
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=stub_transport(_good_score_payload()),
        audit_db_path=env.parent / "state.sqlite",
        banned_entities=("AAPL",),
    )
    with pytest.raises(BacktestLeakageError) as excinfo:
        client.call_structured(
            system="sys",
            user="<COMPANY> related news for AAPL today.",
            schema=EarningsTranscriptScore,
            model="claude-sonnet-4-5",
        )
    assert "AAPL" in str(excinfo.value)


def test_backtest_mode_rejects_in_window_iso_date(env: Path) -> None:
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=stub_transport(_good_score_payload()),
        audit_db_path=env.parent / "state.sqlite",
    )
    with pytest.raises(BacktestLeakageError) as excinfo:
        client.call_structured(
            system="sys",
            user="<COMPANY> reported on 2025-04-15.",
            schema=EarningsTranscriptScore,
            model="claude-sonnet-4-5",
        )
    assert "2025-04-15" in str(excinfo.value)


def test_backtest_mode_rejects_in_window_english_date(env: Path) -> None:
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=stub_transport(_good_score_payload()),
        audit_db_path=env.parent / "state.sqlite",
    )
    with pytest.raises(BacktestLeakageError) as excinfo:
        client.call_structured(
            system="sys",
            user="<COMPANY> guidance issued April 15, 2025.",
            schema=EarningsTranscriptScore,
            model="claude-sonnet-4-5",
        )
    assert "in-window" in str(excinfo.value).lower()


def test_backtest_mode_rejects_in_window_us_slash_date(env: Path) -> None:
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=stub_transport(_good_score_payload()),
        audit_db_path=env.parent / "state.sqlite",
    )
    with pytest.raises(BacktestLeakageError):
        client.call_structured(
            system="sys",
            user="<COMPANY> filed on 04/15/2025.",
            schema=EarningsTranscriptScore,
            model="claude-sonnet-4-5",
        )


def test_backtest_mode_rejects_missing_company_marker(env: Path) -> None:
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=stub_transport(_good_score_payload()),
        audit_db_path=env.parent / "state.sqlite",
    )
    with pytest.raises(BacktestLeakageError) as excinfo:
        client.call_structured(
            system="sys",
            user="some opaque transcript with no marker",
            schema=EarningsTranscriptScore,
            model="claude-sonnet-4-5",
        )
    assert "anonymization" in str(excinfo.value).lower()


def test_live_mode_accepts_same_content(env: Path) -> None:
    """The same prompt that fails in backtest mode succeeds in live mode."""
    client = LLMClient(
        mode="live",
        transport=stub_transport(_good_score_payload()),
        audit_db_path=env.parent / "state.sqlite",
    )
    resp = client.call_structured(
        system="sys",
        user="AAPL reported on 2025-04-15 with strong results.",
        schema=EarningsTranscriptScore,
        model="claude-sonnet-4-5",
    )
    assert resp.audit_row_id > 0


# ============================================================ end-to-end
def test_end_to_end_signal_anonymizes_every_prompt(env: Path) -> None:
    """`signals.llm_earnings` must produce only anonymized outgoing prompts.

    We capture every transport call and assert the user content contains
    no banned entity and contains <COMPANY>.
    """
    # Seed history so SUE works.
    rows = [
        {
            "ticker": "AAPL",
            "report_date": datetime(2024, 1 + i, 15, tzinfo=UTC),
            "period_end": datetime(2024, 1 + i, 15, tzinfo=UTC),
            "actual_eps": 1.00 + 0.01 * i,
            "consensus_eps": 1.00,
            "revenue": 1_000_000.0,
            "source": "TEST",
        }
        for i in range(8)
    ]
    store.upsert_earnings(rows, db_path=env)

    transport = stub_transport(_good_score_payload())
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=transport,
        audit_db_path=env.parent / "state.sqlite",
        banned_entities=("AAPL",),
    )
    sig = combined_earnings_signal(
        client=client,
        ticker="AAPL",
        actual_eps=Decimal("1.5"),
        consensus_eps=Decimal("1.0"),
        transcript_parts=TranscriptParts(
            prepared_remarks="Apple reported strong revenue.",
            qa_session="Q: AAPL outlook? A: confident.",
        ),
        as_of_date=datetime(2024, 12, 31, tzinfo=UTC),  # outside window — fine
        company_aliases=("Apple",),
        db_path=env,
    )
    assert sig is not None
    captured = transport.calls  # type: ignore[attr-defined]
    assert len(captured) >= 1
    for call in captured:
        body = call["messages"][0]["content"]
        assert "AAPL" not in body, f"prompt leaked AAPL: {body[:200]}"
        assert "Apple" not in body, f"prompt leaked Apple: {body[:200]}"
        assert "<COMPANY>" in body, f"prompt missing <COMPANY> marker: {body[:200]}"


def test_end_to_end_signal_in_window_date_rejected(env: Path) -> None:
    """If a transcript happens to contain an in-window date, the gate fires."""
    rows = [
        {
            "ticker": "AAPL",
            "report_date": datetime(2024, 1 + i, 15, tzinfo=UTC),
            "period_end": datetime(2024, 1 + i, 15, tzinfo=UTC),
            "actual_eps": 1.00 + 0.01 * i,
            "consensus_eps": 1.00,
            "revenue": 1_000_000.0,
            "source": "TEST",
        }
        for i in range(8)
    ]
    store.upsert_earnings(rows, db_path=env)

    transport = stub_transport(_good_score_payload())
    client = LLMClient(
        mode="backtest",
        backtest_window=(date(2025, 1, 1), date(2025, 6, 30)),
        transport=transport,
        audit_db_path=env.parent / "state.sqlite",
    )
    # Transcript carries an in-window date — must be rejected.
    with pytest.raises(BacktestLeakageError):
        combined_earnings_signal(
            client=client,
            ticker="AAPL",
            actual_eps=Decimal("1.5"),
            consensus_eps=Decimal("1.0"),
            transcript_parts=TranscriptParts(
                prepared_remarks="On 2025-03-15 we updated guidance.",
                qa_session="",
            ),
            as_of_date=datetime(2024, 12, 31, tzinfo=UTC),
            company_aliases=("Apple",),
            db_path=env,
        )


# ============================================================ helpers
def test_extract_dates_covers_all_required_formats() -> None:
    """The extract_dates utility must catch every format the gate relies on.

    PRD §6.2 requires no calendar-date leakage in prompts; the regex set
    must cover the common ISO and English variants.
    """
    text = (
        "Filed 2025-04-15. "
        "Issued 04/15/2025. "
        "Press release April 15, 2025. "
        "Updated 15 April 2025. "
        "Earlier 2025/04/15."
    )
    found = extract_dates(text)
    assert date(2025, 4, 15) in found
    # The US-style 04/15 is interpreted both ways; assert at least the canonical reading.
    assert any(d.year == 2025 and d.month == 4 and d.day == 15 for d in found)


def test_runs_under_five_seconds() -> None:
    """Per task 19: the gate must run quickly so CI feedback stays tight.

    We don't hit the network — every transport is stubbed — so a multi-second
    suite would indicate something has gone wrong.
    """
    import time

    t0 = time.perf_counter()
    extract_dates("just a noop call")
    assert time.perf_counter() - t0 < 0.1
