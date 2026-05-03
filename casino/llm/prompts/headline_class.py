"""Headline classifier prompt — Haiku-based news triage (task 11).

High-volume sentiment + materiality classifier called from
`casino/jobs/news_intraday.py` (Phase 3 wiring; not in scope here). Uses
Haiku 4.5 because volume is large and the call is short — Sonnet would
cost 3x for the same accuracy on this task per CLAUDE.md and PRD §6.1.

PRD §6.1 rule 5 mandates Batch API for non-latency-sensitive jobs (50%
saving). The end-of-day digest path goes through `classify_headlines_batch`;
the intraday path uses single calls.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import cast

from casino.llm.client import LLMClient
from casino.llm.schemas import HeadlineClassification

# Stable system prompt — cached on every call (PRD §6.1 rule 1).
HEADLINE_CLASS_SYSTEM_PROMPT = (
    "You classify financial news headlines for trading relevance and sentiment. "
    "For each headline, decide:\n"
    "\n"
    "  sentiment: float in [-1, 1]. -1 = clearly bearish for <COMPANY>, "
    "+1 = clearly bullish, 0 = neutral or unclear.\n"
    "  is_material: true only if the headline is likely to move the stock more "
    "than ~1% on the day. Material events: earnings, M&A, regulatory action, "
    "guidance changes, key personnel changes, large product launches, lawsuits "
    "with damages > 1% of market cap, credit-rating changes. Non-material: "
    "blog mentions, hiring posts, sponsorships, opinion pieces.\n"
    "  relevance: one of 'high' / 'medium' / 'low'. high = direct corporate "
    "event; medium = sector/peer news; low = passing mention or noise.\n"
    "  rationale: <= 200 characters.\n"
    "\n"
    "Output strict JSON only matching:\n"
    '{"sentiment": float, "is_material": bool, "relevance": "high"|"medium"|"low", '
    '"rationale": "string"}\n'
    "\n"
    "The company name has been redacted to <COMPANY>. Score on the headline content only."
)


def anonymize_headline(
    headline: str,
    *,
    ticker: str,
    company_aliases: tuple[str, ...] = (),
) -> str:
    """Replace ticker and company aliases with `<COMPANY>` in the headline."""
    out = headline
    for t in (ticker, *company_aliases):
        if not t:
            continue
        out = re.sub(rf"\b{re.escape(t)}\b", "<COMPANY>", out, flags=re.IGNORECASE)
    return out


def build_user_prompt(headline_anon: str) -> str:
    return f"Headline: {headline_anon}\nClassify per the schema."


def classify_headline(
    *,
    client: LLMClient,
    ticker: str,
    headline: str,
    company_aliases: tuple[str, ...] = (),
    model: str | None = None,
    temperature: float = 0.0,
) -> HeadlineClassification:
    """Classify one headline. Returns `HeadlineClassification`.

    Defaults to Haiku 4.5 — overrideable via `model` for ablations.
    Temperature defaults to 0.0 because the headline classifier is meant
    to be deterministic per (model, prompt) pair (no Monte Carlo).
    """
    used_model = model or "claude-haiku-4-5"
    anon = anonymize_headline(headline, ticker=ticker, company_aliases=company_aliases)
    resp = client.call_structured(
        system=HEADLINE_CLASS_SYSTEM_PROMPT,
        user=build_user_prompt(anon),
        schema=HeadlineClassification,
        model=used_model,
        temperature=temperature,
        extra_banned_entities=(ticker, *company_aliases),
    )
    parsed = resp.parsed
    if not isinstance(parsed, HeadlineClassification):  # pragma: no cover
        raise TypeError(f"unexpected parsed type {type(parsed)!r}")
    return parsed


def classify_headlines_batch(
    *,
    client: LLMClient,
    items: Sequence[dict[str, str | tuple[str, ...]]],
    model: str | None = None,
    temperature: float = 0.0,
) -> list[HeadlineClassification]:
    """Batch path — for the end-of-day digest cron.

    Each item is `{"ticker": str, "headline": str, "company_aliases": tuple}`.
    Routes through `LLMClient.call_structured_batch`, which today loops with
    full guards + audit rows and is the swap-in point for Anthropic's
    Messages Batches API once the daily volume warrants it.
    """
    used_model = model or "claude-haiku-4-5"
    payload: list[dict[str, object]] = []
    for it in items:
        ticker = cast(str, it["ticker"])
        headline = cast(str, it["headline"])
        aliases = cast("tuple[str, ...]", it.get("company_aliases", ()))
        anon = anonymize_headline(headline, ticker=ticker, company_aliases=aliases)
        payload.append(
            {
                "system": HEADLINE_CLASS_SYSTEM_PROMPT,
                "user": build_user_prompt(anon),
                "extra_banned_entities": (ticker, *aliases),
            }
        )
    responses = client.call_structured_batch(
        items=payload,
        schema=HeadlineClassification,
        model=used_model,
        temperature=temperature,
    )
    out: list[HeadlineClassification] = []
    for r in responses:
        if not isinstance(r.parsed, HeadlineClassification):  # pragma: no cover
            raise TypeError(f"unexpected parsed type {type(r.parsed)!r}")
        out.append(r.parsed)
    return out
