"""SEC EDGAR HTTP client.

Fair-access compliance:
- Rate-limited to 10 requests/second.
- User-Agent header includes a real contact email (configurable via SEC_USER_AGENT).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterable
from datetime import date

import httpx
from loguru import logger

from casino.config import get_config

_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


class _RateLimiter:
    """Sliding-window rate limiter: at most `max_calls` per `period` seconds."""

    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self.period:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                wait = self.period - (now - self._calls[0])
                if wait > 0:
                    time.sleep(wait)
                    now = time.monotonic()
                    while self._calls and now - self._calls[0] > self.period:
                        self._calls.popleft()
            self._calls.append(time.monotonic())


class EdgarClient:
    """Minimal EDGAR client with CIK lookup, filing search, and document fetch."""

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        client: httpx.Client | None = None,
        rate_limit: int = 10,
    ) -> None:
        cfg = get_config()
        self.user_agent = user_agent or cfg.sec_user_agent
        self._client = client or httpx.Client(timeout=30.0, headers={"User-Agent": self.user_agent})
        self._owns_client = client is None
        self._limiter = _RateLimiter(max_calls=rate_limit, period=1.0)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _get(self, url: str) -> httpx.Response:
        self._limiter.acquire()
        resp = self._client.get(url, headers={"User-Agent": self.user_agent})
        resp.raise_for_status()
        return resp

    # --------------------------------------------------------------- CIK lookup
    def get_cik_from_ticker(self, ticker: str) -> str:
        """Resolve ticker → 10-digit zero-padded CIK string."""
        payload = self._get(_TICKER_CIK_URL).json()
        # company_tickers.json is keyed by integer string indices.
        target = ticker.upper()
        for entry in payload.values() if isinstance(payload, dict) else payload:
            if str(entry.get("ticker", "")).upper() == target:
                return str(entry["cik_str"]).zfill(10)
        raise ValueError(f"CIK not found for ticker {ticker!r}")

    # ----------------------------------------------------------- submissions
    def search_filings(
        self,
        cik: str,
        *,
        form_types: Iterable[str],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, object]]:
        """Return filing metadata rows for the requested form types."""
        cik10 = cik.zfill(10)
        url = _SUBMISSIONS_URL.format(cik10=cik10)
        payload = self._get(url).json()
        recent = payload.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        accs = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        wanted = {f.upper() for f in form_types}
        out: list[dict[str, object]] = []
        for i, form in enumerate(forms):
            if form.upper() not in wanted:
                continue
            try:
                fdate = date.fromisoformat(dates[i])
            except (ValueError, IndexError):
                continue
            if start_date and fdate < start_date:
                continue
            if end_date and fdate > end_date:
                continue
            accession = accs[i] if i < len(accs) else ""
            primary = primary_docs[i] if i < len(primary_docs) else ""
            acc_nodash = accession.replace("-", "")
            url_doc = f"{_ARCHIVES_BASE}/{int(cik)}/{acc_nodash}/{primary}"
            out.append(
                {
                    "form_type": form,
                    "filing_date": fdate,
                    "accession_number": accession,
                    "primary_document": primary,
                    "url": url_doc,
                    "cik": cik10,
                }
            )
        return out

    def fetch_document(self, url: str) -> bytes:
        """Fetch a single filing document. Returns raw bytes."""
        resp = self._get(url)
        return resp.content

    def store_raw(self, content: bytes, *, ticker: str, form_type: str, accession: str) -> str:
        """Persist raw document under data/raw/edgar/{ticker}/{form}/{accession}.html.

        Returns the relative path string written.
        """
        cfg = get_config()
        out_dir = cfg.data_dir / "raw" / "edgar" / ticker.upper() / form_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{accession}.html"
        out_path.write_bytes(content)
        logger.debug("edgar raw written {}", out_path)
        return str(out_path)
