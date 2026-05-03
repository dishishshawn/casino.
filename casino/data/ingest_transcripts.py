"""FMP earnings transcript ingestion → DuckDB `transcripts` table.

Wraps Financial Modeling Prep (FMP) v3 earning_call_transcript endpoints
with rate limiting, retries, and idempotent upserts via `casino.data.store`.

CLI (PRD §11 Phase 2 deliverable 5):
    uv run python -m casino.data.ingest_transcripts --ticker AAPL --quarter 2024-Q4
    uv run python -m casino.data.ingest_transcripts --ticker AAPL --recent 4

Free FMP tier: 250 calls/day. We rate-limit to 1 req/sec by default and
retry transient errors with exponential backoff. The transcripts table
already exists in `store.py`; this module only owns fetching, cleaning
and upserts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from casino.config import get_config
from casino.data import store

_BASE_URL = "https://financialmodelingprep.com/api/v3"
_MAX_RETRIES = 3
_BACKOFF_BASE_SEC = 1.0
_DEFAULT_RATE_LIMIT_SEC = 1.0  # 1 req/sec → 86k/day max, well under daily cap

# FMP includes the apikey in the query string. httpx's raise_for_status()
# embeds the full URL (with params) into the exception message, which leaks
# the key into logs / stack traces. We scrub before raising.
_APIKEY_RE = re.compile(r"(?i)(apikey=)[^&\s'\"]+")


def _scrub_apikey(text: str) -> str:
    return _APIKEY_RE.sub(r"\1***REDACTED***", text)


@dataclass
class TranscriptSections:
    """Two-part split of an earnings call transcript.

    The Q&A is where guidance defensiveness lives (PRD §5.3); we split here
    so prompts can prioritize Q&A over prepared remarks when the transcript
    has to be truncated.
    """

    prepared_remarks: str
    qa_session: str

    @property
    def full_text(self) -> str:
        if self.qa_session:
            return f"{self.prepared_remarks}\n\n--- Q&A ---\n\n{self.qa_session}"
        return self.prepared_remarks


# regex catches the common Q&A delimiters seen in FMP transcripts.
_QA_SPLIT_RE = re.compile(
    r"(?im)^[ \t]*(?:question[\- ]and[\- ]answer|q\s*[&\-]\s*a|"
    r"questions\s+and\s+answers|operator(?:[:\s].*?)?question)\b",
)
# strip nav/ads commonly seen in scraped transcripts (best-effort fallback).
_AD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)advertisement\s*", re.MULTILINE),
    re.compile(r"(?i)sponsored\s+content", re.MULTILINE),
)


def clean_transcript_text(raw: str) -> str:
    """Normalize whitespace and remove common ad/navigation artifacts."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    for pat in _AD_PATTERNS:
        text = pat.sub("", text)
    # Collapse 3+ blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line, then collapse leading blank lines.
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    return text


def split_into_sections(text: str) -> TranscriptSections:
    """Split a transcript into prepared remarks + Q&A.

    If no Q&A delimiter is detected, the entire text is returned as
    `prepared_remarks` and `qa_session` is empty. This is intentionally
    conservative — losing all text to a bad split would silently degrade
    the LLM signal.
    """
    cleaned = clean_transcript_text(text)
    m = _QA_SPLIT_RE.search(cleaned)
    if m is None:
        return TranscriptSections(prepared_remarks=cleaned, qa_session="")
    return TranscriptSections(
        prepared_remarks=cleaned[: m.start()].rstrip(),
        qa_session=cleaned[m.start() :].strip(),
    )


def _utc_now_stamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_event_date(s: str) -> datetime:
    """Parse FMP's date string into tz-aware UTC datetime."""
    if "T" in s:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
    else:
        d = date.fromisoformat(s)
        dt = datetime(d.year, d.month, d.day)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class FMPClient:
    """Minimal FMP transcript client. Auth via FMP_API_KEY in config."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = _BASE_URL,
        raw_dir: Path | None = None,
        client: httpx.Client | None = None,
        rate_limit_sec: float = _DEFAULT_RATE_LIMIT_SEC,
    ) -> None:
        cfg = get_config()
        self.api_key = api_key if api_key is not None else (cfg.fmp_api_key or "")
        self.base_url = base_url.rstrip("/")
        self.raw_dir = raw_dir or (cfg.data_dir / "raw" / "fmp")
        self._client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None
        self._rate_limit_sec = rate_limit_sec
        self._last_call_ts: float = 0.0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> FMPClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        # Rate-limit politely.
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._rate_limit_sec:
            time.sleep(self._rate_limit_sec - elapsed)
        url = f"{self.base_url}{path}"
        full_params: dict[str, Any] = dict(params or {})
        if self.api_key:
            full_params["apikey"] = self.api_key
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.get(url, params=full_params)
            except httpx.RequestError as e:
                last_exc = e
                wait = _BACKOFF_BASE_SEC * (2**attempt)
                logger.warning(
                    "fmp request error {} (attempt {}/{}): retry in {}s",
                    e,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue
            self._last_call_ts = time.monotonic()
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                wait = _BACKOFF_BASE_SEC * (2**attempt)
                logger.warning(
                    "fmp {} (attempt {}/{}): retry in {}s",
                    resp.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return []
            if resp.status_code in (401, 403):
                # Plan-restricted endpoint or expired/invalid key. Log the
                # scrubbed path (never the apikey) and skip — callers expect
                # an empty list instead of a crash so basket loops can finish.
                logger.warning(
                    "fmp {} forbidden for {} — endpoint not available on the "
                    "current FMP plan (or invalid key); skipping. Upgrade at "
                    "https://site.financialmodelingprep.com/developer/docs",
                    resp.status_code,
                    path,
                )
                return []
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # Redact apikey from the URL httpx puts into the exception.
                clean_msg = _scrub_apikey(str(exc))
                raise httpx.HTTPStatusError(
                    clean_msg,
                    request=exc.request,
                    response=exc.response,
                ) from None
            return resp.json()
        if last_exc:
            raise last_exc
        raise RuntimeError(
            f"fmp request to {_scrub_apikey(url)} failed after {_MAX_RETRIES} attempts",
        )

    def _archive(self, ticker: str, payload: Any) -> Path:
        directory = self.raw_dir / "transcripts"
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / f"{ticker}_{_utc_now_stamp()}.json"
        out.write_text(json.dumps(payload, default=str), encoding="utf-8")
        return out

    # ---------------------------------------------------------------- fetch
    def fetch_transcript(
        self,
        ticker: str,
        *,
        quarter: int | None = None,
        year: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch one or all recent transcripts for a ticker.

        Returns a list of normalized rows ready for `store.upsert_transcripts`.
        """
        path = f"/earning_call_transcript/{ticker.upper()}"
        params: dict[str, Any] = {}
        if quarter is not None:
            params["quarter"] = quarter
        if year is not None:
            params["year"] = year
        payload = self._request(path, params=params)
        self._archive(ticker, payload)
        rows: list[dict[str, Any]] = []
        # FMP returns a list of {"symbol", "quarter", "year", "date", "content"}.
        for entry in payload if isinstance(payload, list) else []:
            ev_date_raw = entry.get("date") or entry.get("publishedDate")
            content = entry.get("content") or entry.get("transcript")
            if not ev_date_raw or not content:
                continue
            ev_date = _parse_event_date(str(ev_date_raw))
            sections = split_into_sections(str(content))
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "event_date": ev_date,
                    "transcript_text": sections.full_text,
                    "source": "FMP",
                    "_quarter": entry.get("quarter"),
                    "_year": entry.get("year"),
                }
            )
        return rows


# ---------------------------------------------------------------------------- storage helper
def upsert_transcripts(rows: list[dict[str, Any]], *, db_path: Path | None = None) -> int:
    """Idempotently upsert transcript rows. Returns count attempted.

    Lives here (not in `store.py`) because the transcripts table schema is
    Phase-1 owned; this module is the only Phase-2 producer for it.
    """
    if not rows:
        return 0
    sql = """
        INSERT OR REPLACE INTO transcripts (ticker, event_date, transcript_text, source)
        VALUES (?, ?, ?, ?)
    """
    with store.get_duckdb_conn(db_path) as conn:
        conn.executemany(
            sql,
            [
                (
                    r["ticker"],
                    r["event_date"],
                    r.get("transcript_text"),
                    r.get("source", "FMP"),
                )
                for r in rows
            ],
        )
    return len(rows)


# ============================================================================ CLI
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="casino.data.ingest_transcripts",
        description="Ingest earnings transcripts from FMP into DuckDB.",
    )
    p.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    p.add_argument("--quarter", type=str, default=None, help="Quarter spec like 2024-Q4")
    p.add_argument("--recent", type=int, default=None, help="Fetch N most recent transcripts")
    return p


def _parse_quarter_spec(spec: str) -> tuple[int, int]:
    """Parse '2024-Q4' → (year=2024, quarter=4)."""
    m = re.fullmatch(r"(\d{4})-Q([1-4])", spec.strip(), flags=re.IGNORECASE)
    if not m:
        raise SystemExit(f"invalid --quarter spec {spec!r}; expected 'YYYY-QN'")
    return int(m.group(1)), int(m.group(2))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = get_config()
    if not cfg.fmp_api_key:
        logger.error("FMP_API_KEY not set; refusing to make live requests")
        return 2
    store.create_schema()
    counts: dict[str, int] = {}
    with FMPClient() as client:
        if args.quarter is not None:
            year, q = _parse_quarter_spec(args.quarter)
            rows = client.fetch_transcript(args.ticker, quarter=q, year=year)
            counts[f"{args.ticker}-{args.quarter}"] = upsert_transcripts(rows)
        elif args.recent is not None:
            rows = client.fetch_transcript(args.ticker)
            # FMP returns most-recent first; truncate.
            rows = rows[: args.recent]
            counts[f"{args.ticker}-recent{args.recent}"] = upsert_transcripts(rows)
        else:
            rows = client.fetch_transcript(args.ticker)
            counts[args.ticker] = upsert_transcripts(rows)
    logger.info("transcript ingest summary: {}", counts)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
