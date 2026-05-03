"""Pydantic schemas for structured LLM outputs.

Phase 2 (task 9): hardened with computed properties, transcript correlation
hash, and JSON round-trip helpers. These models are the contract between
`casino/llm/prompts/*` and `casino/signals/*` — no signal logic ever
consumes free-form LLM text.

PRD §5.3 mandates the v1 transcript schema: beat_quality, guidance_tone,
qa_defensiveness, confidence, reasoning. PRD §6.1 rule 2: every prompt
validates against a schema here.

Floats are acceptable here per CLAUDE.md "Conventions" — these are
research-only scores, not money. Money values stay Decimal.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


class EarningsTranscriptScore(BaseModel):
    """Structured output for the earnings-call transcript scorer.

    Mirrors PRD §5.3. The combined signal in `signals/llm_earnings.py`
    consumes `composite_score` after dividing the unbounded LLM scores onto
    the same numeric scale as the normalized SUE.

    `transcript_hash` is set by the prompt wrapper (sha256 of the original
    pre-anonymization transcript text); the audit log uses it to correlate
    repeated calls on the same artifact (Monte-Carlo over LLM stochasticity,
    PRD §6.2 rule 5).
    """

    beat_quality: float = Field(..., ge=-2.0, le=2.0)
    guidance_tone: float = Field(..., ge=-2.0, le=2.0)
    qa_defensiveness: float = Field(..., ge=-2.0, le=2.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1, max_length=2000)
    transcript_hash: str | None = Field(default=None, max_length=64)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def composite_score(self) -> float:
        """Average of the three −2..+2 axes, mapped to [-1, 1].

        Sum range is [-6, +6]; divide by 6 to land on [-1, +1] which matches
        the post-normalization SUE scale used in `signals/llm_earnings.py`.
        """
        total = self.beat_quality + self.guidance_tone + self.qa_defensiveness
        return total / 6.0

    def to_audit_dict(self) -> dict[str, Any]:
        """Flat dict suitable for the SQLite audit row's parsed_score column."""
        return {
            "beat_quality": self.beat_quality,
            "guidance_tone": self.guidance_tone,
            "qa_defensiveness": self.qa_defensiveness,
            "confidence": self.confidence,
            "composite_score": self.composite_score,
            "transcript_hash": self.transcript_hash,
        }


class HeadlineClassification(BaseModel):
    """Structured output for the Haiku-based headline classifier (task 11).

    Two coupled outputs: a sentiment score and a materiality flag. Headlines
    that aren't material to trading are dropped at the signal layer
    regardless of sentiment.
    """

    sentiment: float = Field(..., ge=-1.0, le=1.0)
    is_material: bool
    rationale: str = Field(..., min_length=1, max_length=500)
    relevance: Literal["high", "medium", "low"] = "medium"

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "sentiment": self.sentiment,
            "is_material": self.is_material,
            "relevance": self.relevance,
        }


def parse_response_json(text: str, schema: type[BaseModel]) -> BaseModel:
    """Parse a model's text response into `schema`, tolerating common wrappers.

    Anthropic models tend to return JSON sometimes wrapped in ```json fences
    or with leading/trailing prose. We strip the common variants before
    delegating to Pydantic. On unparseable input, raises ValueError so the
    LLM client's retry loop can decide whether to back off.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence (with or without language tag).
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    # Some models prepend prose before the object; pluck the first {...} block.
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"no JSON object found in response: {text[:200]!r}")
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in response: {e}") from e
    return schema.model_validate(data)
