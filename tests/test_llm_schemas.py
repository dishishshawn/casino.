"""Tests for casino.llm.schemas (task 9)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from casino.llm.schemas import (
    EarningsTranscriptScore,
    HeadlineClassification,
    parse_response_json,
)


def test_earnings_score_field_constraints() -> None:
    EarningsTranscriptScore(
        beat_quality=2,
        guidance_tone=-2,
        qa_defensiveness=0,
        confidence=0.7,
        reasoning="ok",
    )
    with pytest.raises(ValidationError):
        EarningsTranscriptScore(
            beat_quality=3,  # out of range
            guidance_tone=0,
            qa_defensiveness=0,
            confidence=0.5,
            reasoning="ok",
        )
    with pytest.raises(ValidationError):
        EarningsTranscriptScore(
            beat_quality=0,
            guidance_tone=0,
            qa_defensiveness=0,
            confidence=1.5,  # out of range
            reasoning="ok",
        )


def test_earnings_composite_score_range() -> None:
    perfect = EarningsTranscriptScore(
        beat_quality=2,
        guidance_tone=2,
        qa_defensiveness=2,
        confidence=1.0,
        reasoning="all positive",
    )
    assert perfect.composite_score == pytest.approx(1.0)
    worst = EarningsTranscriptScore(
        beat_quality=-2,
        guidance_tone=-2,
        qa_defensiveness=-2,
        confidence=1.0,
        reasoning="all negative",
    )
    assert worst.composite_score == pytest.approx(-1.0)
    mid = EarningsTranscriptScore(
        beat_quality=1,
        guidance_tone=-1,
        qa_defensiveness=0,
        confidence=0.5,
        reasoning="mixed",
    )
    assert mid.composite_score == pytest.approx(0.0)


def test_earnings_to_audit_dict_includes_composite_and_hash() -> None:
    s = EarningsTranscriptScore(
        beat_quality=1,
        guidance_tone=1,
        qa_defensiveness=0,
        confidence=0.5,
        reasoning="ok",
        transcript_hash="deadbeef" * 8,
    )
    d = s.to_audit_dict()
    assert d["composite_score"] == pytest.approx(2.0 / 6.0)
    assert d["transcript_hash"] == "deadbeef" * 8


def test_headline_classification_validates() -> None:
    HeadlineClassification(
        sentiment=0.5,
        is_material=True,
        rationale="earnings beat",
    )
    with pytest.raises(ValidationError):
        HeadlineClassification(
            sentiment=2.0,
            is_material=False,
            rationale="bad",
        )


def test_parse_response_json_strips_code_fences() -> None:
    body = json.dumps(
        {
            "beat_quality": 1,
            "guidance_tone": 0,
            "qa_defensiveness": 1,
            "confidence": 0.8,
            "reasoning": "ok",
        }
    )
    fenced = f"```json\n{body}\n```"
    parsed = parse_response_json(fenced, EarningsTranscriptScore)
    assert isinstance(parsed, EarningsTranscriptScore)
    assert parsed.beat_quality == 1


def test_parse_response_json_extracts_json_from_prose() -> None:
    body = json.dumps(
        {
            "sentiment": 0.2,
            "is_material": True,
            "rationale": "ok",
        }
    )
    wrapped = f"Sure, here is the result:\n{body}\nHope that helps."
    parsed = parse_response_json(wrapped, HeadlineClassification)
    assert isinstance(parsed, HeadlineClassification)
    assert parsed.is_material is True


def test_parse_response_json_raises_on_garbage() -> None:
    with pytest.raises(ValueError, match="no JSON object found"):
        parse_response_json("the model said no", EarningsTranscriptScore)


def test_round_trip_via_model_dump_json() -> None:
    s = EarningsTranscriptScore(
        beat_quality=1,
        guidance_tone=-1,
        qa_defensiveness=0,
        confidence=0.4,
        reasoning="ok",
    )
    j = s.model_dump_json()
    s2 = EarningsTranscriptScore.model_validate_json(j)
    assert s2.composite_score == pytest.approx(s.composite_score)
