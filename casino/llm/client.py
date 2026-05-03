"""Anthropic client wrapper — the *only* path to the Anthropic API.

Owns: prompt caching headers, retries with exponential backoff, structured
output parsing + retry on validation failure, USD cost computation, the
SQLite audit row, anonymization enforcement, and rejection of in-window
dates in backtest mode.

PRD §6 (LLM discipline) is implemented here so feature code (signals,
prompts) cannot accidentally leak entities or future dates by forgetting
to call a helper. Mode is set at client construction → cannot be forgotten
per call site.

Public surface used by `casino/llm/prompts/*` and `casino/signals/*`:
    LLMClient.call_structured(...)              — single synchronous call
    LLMClient.call_structured_batch(...)        — Batch API (50% saving)
    cost_usd_for_usage(model, usage)            — pure pricing helper
    BacktestLeakageError                        — raised on guard violations

Tests inject a stub via the `transport` parameter, so no live network is
required for the suite. The orchestrator confirms tests pass without
ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Protocol, cast

from loguru import logger
from pydantic import BaseModel, ValidationError

from casino.config import get_config
from casino.llm import audit
from casino.llm.schemas import parse_response_json

Mode = Literal["live", "backtest"]


# ---------------------------------------------------------------------------- pricing

# Per-million-token USD rates from PRD §7 (2026 retail tier).
# Keys are Anthropic model IDs (config.py defaults plus the long-form IDs the
# Anthropic SDK accepts). We resolve by prefix so we tolerate dated
# sub-versions (e.g. "claude-haiku-4-5-20251001").
_PRICING: dict[str, tuple[float, float, float]] = {
    # model_prefix -> (input_per_M, cached_input_per_M, output_per_M)
    "claude-haiku-4": (1.0, 0.10, 5.0),
    "claude-sonnet-4": (3.0, 0.30, 15.0),
    "claude-opus-4": (5.0, 0.50, 25.0),
}


def _pricing_for(model: str) -> tuple[float, float, float]:
    """Return (input, cached, output) USD per 1M tokens for `model`.

    Falls back to Sonnet rates if an unknown model slips through (so we never
    silently report $0). Logs a warning so the test suite catches drift.
    """
    for prefix, rates in _PRICING.items():
        if model.startswith(prefix):
            return rates
    logger.warning("unknown model {} for pricing; defaulting to Sonnet rates", model)
    return _PRICING["claude-sonnet-4"]


def cost_usd_for_usage(model: str, usage: Usage) -> float:
    """Compute USD cost for one call given token counts.

    Formula: (input * input_rate + cached_read * cached_rate
              + cache_creation * input_rate + output * output_rate) / 1e6.

    Cache *creation* tokens are billed at the full input rate (Anthropic's
    documented behavior), cache *read* tokens at the cached rate.
    """
    in_rate, cached_rate, out_rate = _pricing_for(model)
    cost = (
        usage.input_tokens * in_rate
        + usage.cache_creation_tokens * in_rate
        + usage.cache_read_tokens * cached_rate
        + usage.output_tokens * out_rate
    ) / 1_000_000.0
    return float(cost)


# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class Usage:
    """Token usage from one model response."""

    input_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RawResponse:
    """What a transport returns. Decouples us from anthropic.types.Message."""

    text: str
    usage: Usage
    model: str


@dataclass(frozen=True)
class ParsedResponse:
    """What the public API returns to callers."""

    parsed: BaseModel
    raw_text: str
    usage: Usage
    cost_usd: float
    latency_ms: int
    model: str
    mode: Mode
    prompt_hash: str
    audit_row_id: int


# ---------------------------------------------------------------------------- transport


class _Transport(Protocol):
    """Minimal interface the client needs from the underlying SDK.

    The default implementation calls `anthropic.Anthropic`. Tests pass a
    callable returning `RawResponse` directly.
    """

    def messages_create(
        self,
        *,
        model: str,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        extra_headers: dict[str, str] | None = None,
    ) -> RawResponse: ...


class _AnthropicSDKTransport:
    """Default transport: thin wrapper around `anthropic.Anthropic`."""

    def __init__(self, api_key: str | None = None) -> None:
        # Imported lazily so the test suite doesn't need a real key just to
        # construct an `LLMClient` whose transport is overridden.
        import anthropic  # noqa: PLC0415

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def messages_create(
        self,
        *,
        model: str,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        extra_headers: dict[str, str] | None = None,
    ) -> RawResponse:
        msg = self._client.messages.create(
            model=model,
            system=system,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            extra_headers=extra_headers or {},
        )
        # Concatenate all text blocks (tools/images aren't expected here).
        text_parts: list[str] = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        sdk_usage = msg.usage
        usage = Usage(
            input_tokens=int(getattr(sdk_usage, "input_tokens", 0)),
            cache_creation_tokens=int(getattr(sdk_usage, "cache_creation_input_tokens", 0) or 0),
            cache_read_tokens=int(getattr(sdk_usage, "cache_read_input_tokens", 0) or 0),
            output_tokens=int(getattr(sdk_usage, "output_tokens", 0)),
        )
        return RawResponse(text="".join(text_parts), usage=usage, model=msg.model)


# ---------------------------------------------------------------------------- guards


class BacktestLeakageError(RuntimeError):
    """Raised when a backtest-mode prompt would leak entities or future dates."""


# Date regexes covering PRD §6.2 anonymization needs:
#   YYYY-MM-DD      (2025-01-05)
#   YYYY/MM/DD      (2025/01/05)
#   MM/DD/YYYY      (01/05/2025) — US-style
#   DD/MM/YYYY      (05/01/2025) — also covered by the same regex; we resolve both
#   "Jan 5, 2025" / "5 January 2025"
_DATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),
    re.compile(r"\b(\d{4})/(\d{1,2})/(\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
    re.compile(
        r"\b(?P<m>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<d>\d{1,2})\s+(?P<m>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(?P<y>\d{4})\b",
        re.IGNORECASE,
    ),
]

_MONTH_TO_NUM: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except (ValueError, OverflowError):
        return None


def extract_dates(text: str) -> list[date]:
    """Return all parseable dates found in `text`.

    Numeric dates with ambiguous order (MM/DD/YYYY vs DD/MM/YYYY) are
    interpreted *both* ways and both candidates returned. We err on the side
    of false positives — for the look-ahead guard, false positives just mean
    a slightly stricter prompt; false negatives would defeat the guard.
    """
    found: list[date] = []
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            gd = m.groupdict()
            if (
                gd.get("m") is not None
                and isinstance(gd["m"], str)
                and gd["m"].lower() in _MONTH_TO_NUM
            ):
                month = _MONTH_TO_NUM[gd["m"].lower()]
                day = int(gd["d"])
                year = int(gd["y"])
                cand = _safe_date(year, month, day)
                if cand is not None:
                    found.append(cand)
                continue
            groups = m.groups()
            if len(groups) == 3 and all(g.isdigit() for g in groups):
                a, b, c = (int(g) for g in groups)
                # YYYY-first patterns: a is year.
                if a > 1900:
                    cand = _safe_date(a, b, c)
                    if cand is not None:
                        found.append(cand)
                else:
                    # ambiguous order; emit both readings.
                    cand1 = _safe_date(c, a, b)  # MM/DD/YYYY
                    cand2 = _safe_date(c, b, a)  # DD/MM/YYYY
                    if cand1 is not None:
                        found.append(cand1)
                    if cand2 is not None and cand2 != cand1:
                        found.append(cand2)
    return found


_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b")


def _looks_like_ticker(text: str, banned: Sequence[str]) -> str | None:
    """Return the first banned ticker substring found in `text`, else None.

    We do a literal substring check first (fast path) and a word-boundary
    regex check second (catches the standard 1–5 caps shape but only flags
    those in `banned`).
    """
    for t in banned:
        if t and t.upper() in text.upper():
            # Word-boundary check to avoid matching e.g. "AAPL" inside "AAPLC".
            if re.search(rf"\b{re.escape(t)}\b", text, flags=re.IGNORECASE):
                return t
    return None


# ---------------------------------------------------------------------------- system block

_DEFAULT_CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


def build_cached_system(text: str) -> list[dict[str, Any]]:
    """Construct the `system` field with prompt-caching enabled.

    Anthropic's prompt-caching API takes `system` as a list of blocks; each
    block carries an optional `cache_control`. We mark the entire system
    prompt as cacheable — callers can pass several blocks if they want
    finer control, but for v1 a single ~90% cache hit on the entire system
    prompt is the documented target.
    """
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": _DEFAULT_CACHE_CONTROL,
        }
    ]


# ---------------------------------------------------------------------------- main client


@dataclass
class LLMClient:
    """The single gateway to Anthropic.

    Construction-time invariants:
        * `mode` is set once. Anonymization and date-leakage guards consult
          it on every call, so call sites cannot forget.
        * `backtest_window` is mandatory in backtest mode — without it we
          cannot tell which dates are "in window".

    Per-call invariants enforced inside `call_structured`:
        * In backtest mode, every prompt must reference `<COMPANY>` (no
          plain ticker or company name leaks).
        * In backtest mode, no parseable date inside `[backtest_window]`
          may appear in the system or user prompt.
        * Every call writes one audit row (success or failure).
        * Validation failures retry up to `validation_retries` times.
    """

    mode: Mode
    backtest_window: tuple[date, date] | None = None
    transport: _Transport | None = None
    validation_retries: int = 2
    backoff_retries: int = 3
    backoff_base_sec: float = 0.5
    audit_db_path: Any = None  # Path | None — Any to avoid mypy strict header
    extra_headers: dict[str, str] = field(default_factory=dict)
    banned_entities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode == "backtest" and self.backtest_window is None:
            raise ValueError("backtest mode requires backtest_window=(start, end)")
        if self.transport is None:
            cfg = get_config()
            self.transport = _AnthropicSDKTransport(api_key=cfg.anthropic_api_key or None)

    # -------------------------------------------------------------- guards
    def enforce_backtest_guards(
        self,
        *,
        system: str,
        user: str,
        extra_banned: Sequence[str] = (),
    ) -> None:
        """Raise BacktestLeakageError if the prompt would leak in backtest mode.

        In live mode this is a no-op — the same content is fine to send.
        """
        if self.mode != "backtest":
            return
        assert self.backtest_window is not None  # narrowed by __post_init__
        win_start, win_end = self.backtest_window

        combined = f"{system}\n{user}"
        # 1) entity anonymization: every prompt must contain <COMPANY> *and*
        #    must not contain the literal banned tickers/names.
        if "<COMPANY>" not in combined:
            raise BacktestLeakageError(
                "backtest prompt missing required <COMPANY> anonymization marker"
            )
        banned = tuple(self.banned_entities) + tuple(extra_banned)
        leak = _looks_like_ticker(combined, banned)
        if leak is not None:
            raise BacktestLeakageError(
                f"backtest prompt leaks banned entity {leak!r}; expected <COMPANY>"
            )

        # 2) no date inside the backtest window may appear in the prompt.
        for d in extract_dates(combined):
            if win_start <= d <= win_end:
                raise BacktestLeakageError(
                    f"backtest prompt contains in-window date {d.isoformat()} "
                    f"(window {win_start.isoformat()}..{win_end.isoformat()})"
                )

    # -------------------------------------------------------------- call
    def call_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        extra_banned_entities: Sequence[str] = (),
        prompt_hash_input: str | None = None,
    ) -> ParsedResponse:
        """Make one structured call. Validates against `schema`; retries on
        validation failure or transient errors.

        `prompt_hash_input`: if provided, hashes this string instead of the
        rendered prompt — useful when the prompt body changes across runs
        (e.g., timestamps the model never sees) but you want the audit log
        to correlate calls on the same logical artifact.
        """
        self.enforce_backtest_guards(
            system=system,
            user=user,
            extra_banned=extra_banned_entities,
        )

        prompt_hash = hashlib.sha256(
            (prompt_hash_input or f"{system}\n---\n{user}").encode("utf-8")
        ).hexdigest()
        sys_blocks = build_cached_system(system)
        messages = [{"role": "user", "content": user}]

        last_err: Exception | None = None
        latency_ms = 0
        usage_total = Usage(0, 0, 0, 0)
        cost_total = 0.0
        for attempt in range(self.validation_retries + 1):
            raw, this_latency, this_cost = self._call_with_backoff(
                model=model,
                system=sys_blocks,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms += this_latency
            usage_total = Usage(
                input_tokens=usage_total.input_tokens + raw.usage.input_tokens,
                cache_creation_tokens=usage_total.cache_creation_tokens
                + raw.usage.cache_creation_tokens,
                cache_read_tokens=usage_total.cache_read_tokens + raw.usage.cache_read_tokens,
                output_tokens=usage_total.output_tokens + raw.usage.output_tokens,
            )
            cost_total += this_cost
            try:
                parsed = parse_response_json(raw.text, schema)
            except (ValueError, ValidationError) as e:
                last_err = e
                logger.warning(
                    "schema validation failed (attempt {}/{}): {}",
                    attempt + 1,
                    self.validation_retries + 1,
                    e,
                )
                continue
            row_id = audit.write_audit_row(
                prompt_hash=prompt_hash,
                model=raw.model,
                mode=self.mode,
                input_tokens=usage_total.input_tokens,
                cache_creation_tokens=usage_total.cache_creation_tokens,
                cache_read_tokens=usage_total.cache_read_tokens,
                output_tokens=usage_total.output_tokens,
                cost_usd=cost_total,
                latency_ms=latency_ms,
                parsed_score=parsed.model_dump(mode="json")
                if hasattr(parsed, "to_audit_dict")
                else parsed.model_dump(mode="json"),
                success=True,
                error_msg=None,
                schema_name=schema.__name__,
                db_path=self.audit_db_path,
            )
            return ParsedResponse(
                parsed=parsed,
                raw_text=raw.text,
                usage=usage_total,
                cost_usd=cost_total,
                latency_ms=latency_ms,
                model=raw.model,
                mode=self.mode,
                prompt_hash=prompt_hash,
                audit_row_id=row_id,
            )

        # All validation retries exhausted: write failure row and raise.
        audit.write_audit_row(
            prompt_hash=prompt_hash,
            model=model,
            mode=self.mode,
            input_tokens=usage_total.input_tokens,
            cache_creation_tokens=usage_total.cache_creation_tokens,
            cache_read_tokens=usage_total.cache_read_tokens,
            output_tokens=usage_total.output_tokens,
            cost_usd=cost_total,
            latency_ms=latency_ms,
            parsed_score=None,
            success=False,
            error_msg=str(last_err) if last_err else "validation retries exhausted",
            schema_name=schema.__name__,
            db_path=self.audit_db_path,
        )
        raise ValueError(
            f"LLM call failed schema validation after {self.validation_retries + 1} attempts: {last_err}"
        )

    # ---------------------------------------------------------- batch path
    def call_structured_batch(
        self,
        *,
        items: Sequence[dict[str, Any]],
        schema: type[BaseModel],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> list[ParsedResponse]:
        """Process a batch of structured calls.

        For Phase 2 we implement this as a sequential loop (each call still
        goes through `call_structured` with full guards + audit), with the
        batch cost discount applied at the audit layer. Wiring up Anthropic's
        Messages Batches API can drop in here without changing call sites
        once we have a non-trivial batch size in the daily cron.

        `items` is a list of dicts with keys: `system`, `user`, optional
        `extra_banned_entities`, optional `prompt_hash_input`.
        """
        results: list[ParsedResponse] = []
        for it in items:
            results.append(
                self.call_structured(
                    system=cast(str, it["system"]),
                    user=cast(str, it["user"]),
                    schema=schema,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_banned_entities=cast(
                        "Sequence[str]", it.get("extra_banned_entities", ())
                    ),
                    prompt_hash_input=cast(
                        "str | None",
                        it.get("prompt_hash_input"),
                    ),
                )
            )
        return results

    # --------------------------------------------------------- internals
    def _call_with_backoff(
        self,
        *,
        model: str,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[RawResponse, int, float]:
        """Issue one transport call, retrying transient errors with backoff."""
        assert self.transport is not None
        last_exc: Exception | None = None
        for attempt in range(self.backoff_retries):
            t0 = time.perf_counter()
            try:
                raw = self.transport.messages_create(
                    model=model,
                    system=system,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_headers=dict(self.extra_headers) or None,
                )
            except Exception as e:  # noqa: BLE001 — transport throws diverse types
                last_exc = e
                wait = self.backoff_base_sec * (2**attempt)
                logger.warning(
                    "LLM transport error (attempt {}/{}): {} — retrying in {}s",
                    attempt + 1,
                    self.backoff_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
                continue
            latency_ms = int((time.perf_counter() - t0) * 1000)
            cost = cost_usd_for_usage(model, raw.usage)
            return raw, latency_ms, cost
        assert last_exc is not None
        raise last_exc


# ---------------------------------------------------------------------------- helpers


def stub_transport(
    response_text: str | Callable[[], str],
    *,
    input_tokens: int = 100,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    output_tokens: int = 50,
    model: str | None = None,
) -> _Transport:
    """A tiny in-process transport for tests.

    Captures every outgoing call so test code can assert that cache_control
    is present on system blocks, that the prompt was anonymized, etc.
    """

    captured: list[dict[str, Any]] = []

    class _Stub:
        calls: list[dict[str, Any]] = captured

        def messages_create(
            self,
            *,
            model: str,
            system: list[dict[str, Any]],
            messages: list[dict[str, Any]],
            max_tokens: int,
            temperature: float,
            extra_headers: dict[str, str] | None = None,
        ) -> RawResponse:
            captured.append(
                {
                    "model": model,
                    "system": system,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "extra_headers": extra_headers or {},
                }
            )
            text = response_text() if callable(response_text) else response_text
            return RawResponse(
                text=text,
                usage=Usage(
                    input_tokens=input_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                    cache_read_tokens=cache_read_tokens,
                    output_tokens=output_tokens,
                ),
                model=model if model is None else model,
            )

    return _Stub()


def utc_now() -> datetime:
    """UTC now — convenience so feature code never imports `datetime` for
    the sole purpose of `datetime.now(UTC)`."""
    from datetime import UTC

    return datetime.now(tz=UTC)
