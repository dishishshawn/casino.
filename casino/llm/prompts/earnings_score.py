"""Earnings transcript scoring prompt — the v1 strategy LLM call (task 10).

Wraps `LLMClient.call_structured` with the cached system prompt, the
anonymization helper, and a transcript truncation policy that prioritizes
the parts of the call that carry the strongest signal: management's
prepared guidance and analyst Q&A defensiveness (PRD §5.3).

Anonymization (replace ticker and company name with `<COMPANY>`) is
required in backtest mode per PRD §6.2 rule 2 and CLAUDE.md. Live mode
keeps entities for narrative quality. The mode is set on the LLMClient,
not per call, so it can never be forgotten.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from casino.llm.client import LLMClient
from casino.llm.schemas import EarningsTranscriptScore

# Stable system prompt — large + reused across thousands of transcript calls.
# Goes into the cache_control block on every call (PRD §6.1 rule 1).
EARNINGS_SCORE_SYSTEM_PROMPT = (
    "You are an expert equities analyst evaluating earnings call transcripts. "
    "Score each transcript on three independent axes, each on an integer scale "
    "from -2 (most negative) to +2 (most positive):\n"
    "\n"
    "  beat_quality: clean structural beat (+2) to noisy/non-recurring miss (-2). "
    "Penalize one-time gains, FX tailwinds, share-buyback EPS optics, and unusual "
    "items. Reward operational over financial drivers.\n"
    "  guidance_tone: explicit raised guidance (+2) to lowered/withdrawn (-2). "
    "Be skeptical of vague language ('cautiously optimistic', 'navigating') and "
    "missing forward statements; default these toward 0.\n"
    "  qa_defensiveness: confident, specific responses (+2) to evasive, hedged, "
    "deflecting answers (-2). Watch for repeated 'we're working on that', topic "
    "changes, or refusals to quantify.\n"
    "\n"
    "Also produce:\n"
    "  confidence: float in [0, 1], your overall confidence in this scoring. "
    "Lower confidence on truncated, atypical, or low-information transcripts.\n"
    "  reasoning: <= 500 characters justifying the three scores.\n"
    "\n"
    "Output STRICT JSON only matching this exact schema:\n"
    '{"beat_quality": int, "guidance_tone": int, "qa_defensiveness": int, '
    '"confidence": float, "reasoning": "string"}\n'
    "\n"
    "The company name has been redacted to <COMPANY> in backtest contexts. "
    "Do not speculate about the identity. Score on the content only."
)


# Conservative cap: a Sonnet call with a 60k-token system prompt would be
# expensive. Cap user payload at ~100k chars (~25k tokens) to leave room.
_MAX_USER_CHARS: int = 100_000


@dataclass(frozen=True)
class TranscriptParts:
    """Pre-split transcript fed to the prompt.

    `prepared_remarks` is management's scripted opening; `qa_session` is the
    analyst Q&A. Truncation prefers Q&A — that is where defensiveness
    signals live.
    """

    prepared_remarks: str
    qa_session: str


_COMPANY_TOKEN_RE = re.compile(
    r"\b[A-Z][a-zA-Z]+(?:\s+(?:Inc|Corporation|Corp|Co|LLC|plc|Ltd|Limited))?\b"
)


def anonymize(text: str, *, ticker: str, company_aliases: tuple[str, ...] = ()) -> str:
    """Replace ticker and known company names with `<COMPANY>`.

    The provided `ticker` and any `company_aliases` (e.g. "Apple", "Apple Inc.",
    "AAPL Corp") are matched word-bounded, case-insensitively, and replaced.
    We do NOT do open-ended NER here — that would risk overzealously redacting
    industry terms — relying instead on a curated alias list passed by the
    signal layer that already knows the company name.
    """
    out = text
    targets = (ticker, *company_aliases)
    for t in targets:
        if not t:
            continue
        out = re.sub(rf"\b{re.escape(t)}\b", "<COMPANY>", out, flags=re.IGNORECASE)
    return out


def truncate_for_prompt(parts: TranscriptParts, *, max_chars: int = _MAX_USER_CHARS) -> str:
    """Build the truncated user-prompt body.

    Strategy from task 10: prefer the first 60% of prepared remarks (where
    management lays out the quarter) and the last 80% of Q&A (where analysts
    push hardest on weaknesses). Falls back to a straight char cap if either
    section alone fits.
    """
    if len(parts.prepared_remarks) + len(parts.qa_session) + 32 <= max_chars:
        if parts.qa_session:
            return f"{parts.prepared_remarks}\n\n--- Q&A ---\n\n{parts.qa_session}"
        return parts.prepared_remarks

    qa_budget = int(max_chars * 0.55)
    pr_budget = max_chars - qa_budget - 32
    pr = parts.prepared_remarks
    qa = parts.qa_session
    pr_keep = pr[: int(len(pr) * 0.6)] if len(pr) > pr_budget else pr
    if len(pr_keep) > pr_budget:
        pr_keep = pr_keep[:pr_budget]
    qa_keep = qa[-int(len(qa) * 0.8) :] if len(qa) > qa_budget else qa
    if len(qa_keep) > qa_budget:
        qa_keep = qa_keep[-qa_budget:]
    if qa_keep:
        return f"{pr_keep}\n\n--- Q&A (truncated) ---\n\n{qa_keep}"
    return pr_keep


def build_user_prompt(parts: TranscriptParts, *, max_chars: int = _MAX_USER_CHARS) -> str:
    """Render the user message that wraps the (already anonymized) transcript."""
    body = truncate_for_prompt(parts, max_chars=max_chars)
    return (
        "Score this earnings call transcript for <COMPANY>. "
        "Respond with strict JSON matching the schema above.\n\n"
        f"TRANSCRIPT:\n{body}"
    )


def score_transcript(
    *,
    client: LLMClient,
    ticker: str,
    parts: TranscriptParts,
    company_aliases: tuple[str, ...] = (),
    model: str | None = None,
    temperature: float = 0.3,
    max_chars: int = _MAX_USER_CHARS,
) -> EarningsTranscriptScore:
    """Score one transcript. Returns an `EarningsTranscriptScore`.

    Always anonymizes — even in live mode. PRD §6.2 rule 2 (Glasserman & Lin
    2023) found anonymized prompts can outperform; the entity name is mostly
    a distraction for transcript-level scoring.

    `temperature=0.3` is intentional. PRD §6.2 rule 5 wants temperature > 0
    so callers can run the same transcript several times to bound LLM
    stochasticity (Monte Carlo).
    """
    anon_pr = anonymize(parts.prepared_remarks, ticker=ticker, company_aliases=company_aliases)
    anon_qa = anonymize(parts.qa_session, ticker=ticker, company_aliases=company_aliases)
    anon_parts = TranscriptParts(prepared_remarks=anon_pr, qa_session=anon_qa)
    user_prompt = build_user_prompt(anon_parts, max_chars=max_chars)

    transcript_hash = hashlib.sha256(
        (parts.prepared_remarks + "\n" + parts.qa_session).encode("utf-8")
    ).hexdigest()

    used_model = model or "claude-sonnet-4-5"
    resp = client.call_structured(
        system=EARNINGS_SCORE_SYSTEM_PROMPT,
        user=user_prompt,
        schema=EarningsTranscriptScore,
        model=used_model,
        temperature=temperature,
        # Banned entities pass through to the backtest guard so leakage on the
        # specific ticker we're scoring is rejected even if the call site forgot
        # to scrub it.
        extra_banned_entities=(ticker, *company_aliases),
        prompt_hash_input=transcript_hash,
    )
    parsed = resp.parsed
    if not isinstance(parsed, EarningsTranscriptScore):  # pragma: no cover — type narrowing
        raise TypeError(f"unexpected parsed type {type(parsed)!r}")
    # Stamp the transcript hash so audit rows can correlate Monte-Carlo reruns.
    return parsed.model_copy(update={"transcript_hash": transcript_hash})
