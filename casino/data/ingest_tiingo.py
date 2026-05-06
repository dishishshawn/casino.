"""Tiingo ingestion: OHLCV, fundamentals, news → DuckDB.

Wraps the Tiingo REST v2 API (api.tiingo.com) with rate limiting, retries with
exponential backoff, raw-JSON archival, and idempotent upserts via
`casino.data.store`.

CLI:
    uv run python -m casino.data.ingest_tiingo --ticker AAPL --days 30
    uv run python -m casino.data.ingest_tiingo --ticker AAPL --start 2024-01-01 --end 2024-12-31
    uv run python -m casino.data.ingest_tiingo --ticker AAPL --fundamentals
    uv run python -m casino.data.ingest_tiingo --ticker AAPL --news
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from casino.config import get_config
from casino.data import store

_BASE_URL = "https://api.tiingo.com"
_MAX_RETRIES = 3
_BACKOFF_BASE_SEC = 1.0


def _utc_now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _parse_iso_to_utc(s: str) -> datetime:
    """Parse a Tiingo timestamp (date or ISO with Z) into a tz-aware UTC datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class TiingoClient:
    """Thin Tiingo client. Auth via TIINGO_API_KEY in config."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = _BASE_URL,
        raw_dir: Path | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        cfg = get_config()
        self.api_key = api_key if api_key is not None else cfg.tiingo_api_key
        self.base_url = base_url.rstrip("/")
        self.raw_dir = raw_dir or (cfg.data_dir / "raw" / "tiingo")
        self._client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TiingoClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ http
    def _request(self, path: str, params: Mapping[str, str | int] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.get(url, params=dict(params or {}), headers=headers)
            except httpx.RequestError as e:
                last_exc = e
                wait = _BACKOFF_BASE_SEC * (2**attempt)
                logger.warning(
                    "tiingo request error {} (attempt {}/{}): retry in {}s",
                    e,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue
            if resp.status_code == 429:
                wait = _BACKOFF_BASE_SEC * (2**attempt)
                logger.warning(
                    "tiingo 429 rate-limited (attempt {}/{}); sleep {}s",
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                wait = _BACKOFF_BASE_SEC * (2**attempt)
                logger.warning(
                    "tiingo 5xx={} (attempt {}/{}); sleep {}s",
                    resp.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        if last_exc:
            raise last_exc
        raise RuntimeError(f"tiingo request to {url} failed after {_MAX_RETRIES} attempts")

    def _archive_raw(self, data_type: str, ticker: str, payload: Any) -> Path:
        directory = self.raw_dir / data_type
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / f"{ticker}_{_utc_now_stamp()}.json"
        out.write_text(json.dumps(payload, default=str), encoding="utf-8")
        return out

    # ----------------------------------------------------------- fetch ohlcv
    def fetch_ohlcv(
        self,
        ticker: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        days: int | None = None,
    ) -> list[dict[str, object]]:
        """Fetch daily bars. Returns list of normalized rows ready for store.upsert_ohlcv."""
        if days is not None:
            end = end_date or date.today()
            start = end - timedelta(days=days)
        else:
            start = start_date or (date.today() - timedelta(days=30))
            end = end_date or date.today()
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "format": "json",
        }
        payload = self._request(f"/tiingo/daily/{ticker.lower()}/prices", params=params)
        self._archive_raw("ohlcv", ticker, payload)
        rows: list[dict[str, object]] = []
        for raw in payload:
            ts = _parse_iso_to_utc(str(raw["date"]))
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "ts": ts,
                    # Prices: float64 is acceptable for OHLCV per CLAUDE.md.
                    "open": float(Decimal(str(raw.get("open", 0.0)))),
                    "high": float(Decimal(str(raw.get("high", 0.0)))),
                    "low": float(Decimal(str(raw.get("low", 0.0)))),
                    "close": float(Decimal(str(raw.get("close", 0.0)))),
                    "volume": int(raw.get("volume", 0)),
                    "adj_close": float(Decimal(str(raw.get("adjClose", raw.get("close", 0.0))))),
                }
            )
        return rows

    # ----------------------------------------------------- fetch fundamentals
    def fetch_fundamentals(self, ticker: str) -> list[dict[str, object]]:
        """Fetch quarterly fundamentals. Returns rows for fundamentals + earnings tables."""
        payload = self._request(f"/tiingo/fundamentals/{ticker.lower()}/statements")
        self._archive_raw("fundamentals", ticker, payload)

        rows: list[dict[str, object]] = []
        # Tiingo fundamentals shape varies; we accept a list of {"date": ..., "statementData": {...}}
        for entry in payload if isinstance(payload, list) else []:
            ed = entry.get("date") or entry.get("quarter") or entry.get("fiscalDate")
            if ed is None:
                continue
            report_date = _parse_iso_to_utc(str(ed))
            stmt = entry.get("statementData") or {}
            # Flatten one level — record numeric leaves only.
            for category, fields in stmt.items() if isinstance(stmt, dict) else []:
                if not isinstance(fields, list):
                    continue
                for field in fields:
                    name = field.get("dataCode")
                    val = field.get("value")
                    if name is None or val is None:
                        continue
                    try:
                        numeric = float(val)
                    except (TypeError, ValueError):
                        continue
                    rows.append(
                        {
                            "ticker": ticker.upper(),
                            "report_date": report_date,
                            "metric_name": f"{category}.{name}",
                            "value": numeric,
                        }
                    )
        return rows

    # -------------------------------------------------------------- fetch news
    def fetch_news(self, ticker: str, *, limit: int = 100) -> list[dict[str, object]]:
        """Fetch news. Returns rows for store.upsert_news."""
        params: dict[str, str | int] = {"tickers": ticker.lower(), "limit": limit}
        payload = self._request("/tiingo/news", params=params)
        self._archive_raw("news", ticker, payload)
        rows: list[dict[str, object]] = []
        for n in payload:
            rows.append(
                {
                    "id": str(n.get("id") or n.get("url")),
                    "ticker": ticker.upper(),
                    "headline": n.get("title", ""),
                    "published_at": _parse_iso_to_utc(str(n["publishedDate"])),
                    "source": n.get("source", "tiingo"),
                    "url": n.get("url", ""),
                }
            )
        return rows


# ============================================================================ CLI
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="casino.data.ingest_tiingo",
        description="Ingest OHLCV / fundamentals / news from Tiingo into DuckDB.",
    )
    p.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    p.add_argument("--days", type=int, default=None, help="Last N days of OHLCV")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    p.add_argument("--fundamentals", action="store_true", help="Also fetch fundamentals")
    p.add_argument("--news", action="store_true", help="Also fetch news")
    p.add_argument("--no-ohlcv", action="store_true", help="Skip OHLCV ingestion")
    return p


def _parse_date(s: str | None) -> date | None:
    return None if s is None else date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = get_config()
    if not cfg.tiingo_api_key:
        logger.error("TIINGO_API_KEY not set; refusing to make live requests")
        return 2

    store.create_schema()
    counts: dict[str, int] = {}
    with TiingoClient() as client:
        if not args.no_ohlcv:
            ohlcv = client.fetch_ohlcv(
                args.ticker,
                start_date=_parse_date(args.start),
                end_date=_parse_date(args.end),
                days=args.days,
            )
            counts["ohlcv"] = store.upsert_ohlcv(ohlcv)
        if args.fundamentals:
            fundamentals = client.fetch_fundamentals(args.ticker)
            counts["fundamentals"] = store.upsert_fundamentals(fundamentals)
        if args.news:
            news = client.fetch_news(args.ticker)
            counts["news"] = store.upsert_news(news)
    logger.info("ingest summary {}: {}", args.ticker, counts)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
