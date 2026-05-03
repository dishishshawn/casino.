"""Combined SUE × Claude transcript score signal — the v1 strategy (task 13).

Per PRD §5.3:
    combined = 0.5 * normalize(SUE) + 0.5 * llm_composite
    Trade only when SUE and LLM score agree in sign AND both exceed thresholds.

This is a research-time score (floats), not money. The signal returns are
consumed by the quintile selector in `vbt_research` / `bt_validate`.

Boundaries observed:
    * No `os.environ` reads — config flows through the LLMClient.
    * No direct `anthropic.Anthropic()` import — only `casino.llm.client`.
    * Per-call anonymization is enforced by the LLMClient when in backtest
      mode; this signal additionally always passes `company_aliases` to the
      prompt wrapper so live calls also get scrubbed (Glasserman & Lin 2023).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from loguru import logger

from casino.llm.client import LLMClient
from casino.llm.prompts.earnings_score import (
    TranscriptParts,
    score_transcript,
)
from casino.signals.pead import compute_sue

# Defaults from task 13 spec.
DEFAULT_SUE_NORMALIZER: float = 3.0  # divide raw SUE by 3 → typical [-2, +2] range
DEFAULT_SUE_THRESHOLD: float = 1.0
DEFAULT_LLM_THRESHOLD: float = 0.3
DEFAULT_SUE_WEIGHT: float = 0.5
DEFAULT_LLM_WEIGHT: float = 0.5


@dataclass(frozen=True)
class CombinedSignal:
    """Output of `combined_earnings_signal`.

    `traded` is True when the sign-agreement and both-thresholds-passed
    filters are satisfied — signal layers downstream should skip this row
    when False.
    """

    ticker: str
    sue: float | None
    llm_composite: float
    combined: float
    confidence: float
    traded: bool
    timestamp_utc: datetime
    transcript_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "sue": self.sue,
            "llm_composite": self.llm_composite,
            "combined": self.combined,
            "confidence": self.confidence,
            "traded": self.traded,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "transcript_hash": self.transcript_hash,
        }


def _utc_now() -> datetime:
    from datetime import UTC

    return datetime.now(tz=UTC)


# Process-wide LLM cache: keyed by (transcript_hash, model). Avoids paying
# for the same transcript scored twice in a single research run. Cleared
# per-process; for cross-process dedup the audit table can be queried.
_llm_cache: dict[tuple[str, str], tuple[float, float]] = {}


def _cache_get(
    transcript_hash: str,
    model: str,
) -> tuple[float, float] | None:
    return _llm_cache.get((transcript_hash, model))


def _cache_put(
    transcript_hash: str,
    model: str,
    *,
    llm_composite: float,
    confidence: float,
) -> None:
    _llm_cache[(transcript_hash, model)] = (llm_composite, confidence)


def clear_llm_cache() -> None:
    """Test helper — clears the per-process LLM signal cache."""
    _llm_cache.clear()


def normalize_sue(sue: float, *, divisor: float = DEFAULT_SUE_NORMALIZER) -> float:
    """Map a raw SUE value (typical range ±6) onto [-2, +2] to match the LLM scale.

    PRD §5.3: combined = 0.5 * normalized_SUE + 0.5 * llm_composite. The
    normalized SUE and the LLM composite must live on comparable scales,
    otherwise the 50/50 weighting silently becomes unbalanced.
    """
    return sue / divisor


def combined_score(*, sue: float, llm_composite: float) -> float:
    """Pure 0.5·normSUE + 0.5·llm composite per PRD §5.3.

    Operates on the *normalized* SUE — callers should pass `normalize_sue(raw)`.
    """
    return DEFAULT_SUE_WEIGHT * sue + DEFAULT_LLM_WEIGHT * llm_composite


def passes_trade_filter(
    *,
    sue: float | None,
    llm_composite: float,
    sue_threshold: float = DEFAULT_SUE_THRESHOLD,
    llm_threshold: float = DEFAULT_LLM_THRESHOLD,
) -> bool:
    """PRD §5.3 trade filter.

    * SUE and LLM must agree in sign (both positive or both negative).
    * Both must exceed their respective absolute thresholds.

    Returns False when SUE is unavailable — we don't trade on the LLM alone.
    """
    if sue is None:
        return False
    if sue == 0.0 or llm_composite == 0.0:
        return False
    same_sign = (sue > 0.0) == (llm_composite > 0.0)
    return same_sign and abs(sue) >= sue_threshold and abs(llm_composite) >= llm_threshold


def combined_earnings_signal(
    *,
    client: LLMClient,
    ticker: str,
    actual_eps: Decimal,
    consensus_eps: Decimal | None,
    transcript_parts: TranscriptParts,
    as_of_date: datetime,
    company_aliases: tuple[str, ...] = (),
    model: str | None = None,
    sue_threshold: float = DEFAULT_SUE_THRESHOLD,
    llm_threshold: float = DEFAULT_LLM_THRESHOLD,
    db_path: Path | None = None,
    use_cache: bool = True,
) -> CombinedSignal:
    """Compute one ticker's combined SUE × LLM signal at `as_of_date`.

    Only signals where SUE and LLM agree in sign and both pass thresholds
    set `traded=True` — the quintile selector treats untraded names as
    no-position rather than zero-score so they don't dilute the basket.
    """
    used_model = model or "claude-sonnet-4-5"

    sue_raw = compute_sue(
        ticker,
        actual_eps,
        consensus_eps,
        as_of_date,
        db_path=db_path,
    )
    sue_norm = normalize_sue(sue_raw) if sue_raw is not None else None

    transcript_hash = hashlib.sha256(
        (transcript_parts.prepared_remarks + "\n" + transcript_parts.qa_session).encode("utf-8")
    ).hexdigest()

    cached = _cache_get(transcript_hash, used_model) if use_cache else None
    if cached is not None:
        llm_composite, confidence = cached
        logger.debug("llm cache hit for {} ({})", ticker, transcript_hash[:8])
    else:
        score = score_transcript(
            client=client,
            ticker=ticker,
            parts=transcript_parts,
            company_aliases=company_aliases,
            model=used_model,
        )
        llm_composite = score.composite_score
        confidence = score.confidence
        if use_cache:
            _cache_put(
                transcript_hash,
                used_model,
                llm_composite=llm_composite,
                confidence=confidence,
            )

    combined = combined_score(sue=sue_norm or 0.0, llm_composite=llm_composite)
    traded = passes_trade_filter(
        sue=sue_norm,
        llm_composite=llm_composite,
        sue_threshold=sue_threshold,
        llm_threshold=llm_threshold,
    )
    return CombinedSignal(
        ticker=ticker,
        sue=sue_norm,
        llm_composite=llm_composite,
        combined=combined,
        confidence=confidence,
        traded=traded,
        timestamp_utc=_utc_now(),
        transcript_hash=transcript_hash,
    )


def combined_earnings_signal_batch(
    *,
    client: LLMClient,
    items: Sequence[dict[str, Any]],
    model: str | None = None,
    sue_threshold: float = DEFAULT_SUE_THRESHOLD,
    llm_threshold: float = DEFAULT_LLM_THRESHOLD,
    db_path: Path | None = None,
    use_cache: bool = True,
) -> list[CombinedSignal]:
    """Score a basket — the quintile-selector path.

    PRD §6.1 rule 5: this is a non-latency-sensitive cron path → goes
    through `LLMClient.call_structured_batch` for the 50% Batch API saving.
    Each `items[i]` has the same kwargs as `combined_earnings_signal`
    (minus `client`).

    For Phase 2 we loop here; the underlying batch transport is in
    `LLMClient.call_structured_batch` and is the swap-in point for
    Anthropic's true Messages Batches API.
    """
    out: list[CombinedSignal] = []
    for item in items:
        out.append(
            combined_earnings_signal(
                client=client,
                ticker=item["ticker"],
                actual_eps=item["actual_eps"],
                consensus_eps=item.get("consensus_eps"),
                transcript_parts=item["transcript_parts"],
                as_of_date=item["as_of_date"],
                company_aliases=item.get("company_aliases", ()),
                model=model,
                sue_threshold=sue_threshold,
                llm_threshold=llm_threshold,
                db_path=db_path,
                use_cache=use_cache,
            )
        )
    return out
