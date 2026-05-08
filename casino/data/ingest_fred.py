"""FRED yield-curve ingestion (key-less CSV endpoint) → DuckDB.

The St. Louis Fed publishes daily Treasury yields via fredgraph.csv at:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10

No API key is required for the CSV path; we use it directly via httpx so we
do not pull in `pandas_datareader` for a single endpoint.

Default series for the carry signal (cross-asset Branch C step 2):

    DGS10  - 10-year Treasury constant maturity yield (%)
    DGS5   - 5-year Treasury constant maturity yield (%)
    DGS2   - 2-year Treasury constant maturity yield (%)
    DGS3MO - 3-month Treasury constant maturity yield (%)
    DTB3   - 3-month Treasury bill, secondary-market rate (%)

Values stored are *decimal-of-percent* (e.g. 4.25 for 4.25%) to match the
upstream CSV. The `casino.signals.carry` module divides by 100 at use time.

CLI:

    uv run python -m casino.data.ingest_fred
    uv run python -m casino.data.ingest_fred --series DGS10,DTB3 --start 2010-01-01

Idempotent: re-runs upsert by (series_id, ts) primary key.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from casino.data import store

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_DEFAULT_SERIES: tuple[str, ...] = ("DGS10", "DGS5", "DGS2", "DGS3MO", "DTB3")
_DEFAULT_START = "2005-01-01"
_RATE_LIMIT_SEC = 0.4


def _coerce_float(s: str) -> float | None:
    """Parse a FRED CSV value cell. '.' and '' both mean missing."""
    s = s.strip()
    if not s or s == ".":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_csv(series_id: str, text: str) -> list[dict[str, object]]:
    """Parse a FRED graph CSV body into store-shaped rows.

    Expected header: DATE,<series_id>. The value column header sometimes
    differs in case from the series_id; we resolve by position to be safe.
    """
    rows: list[dict[str, object]] = []
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        return rows
    if len(header) < 2:
        logger.warning("fred CSV for {} had unexpected header: {}", series_id, header)
        return rows
    for raw in reader:
        if len(raw) < 2:
            continue
        ds, vs = raw[0].strip(), raw[1].strip()
        if not ds:
            continue
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            continue
        v = _coerce_float(vs)
        if v is None:
            # Skip missing observations; the table is sparse on weekends/holidays
            # and we do not want to write NULL rows that would break forward-fill
            # in load_fred_panel.
            continue
        rows.append({"series_id": series_id, "ts": d, "value": v})
    return rows


def fetch_series(
    series_id: str,
    *,
    start: str | date | None = _DEFAULT_START,
    end: str | date | None = None,
    client: httpx.Client | None = None,
) -> list[dict[str, object]]:
    """Fetch one FRED series and return store-shaped rows. Empty on error."""
    owns_client = client is None
    c = client or httpx.Client(timeout=30.0)
    try:
        params: dict[str, str] = {"id": series_id}
        if start is not None:
            params["cosd"] = str(start)
        if end is not None:
            params["coed"] = str(end)
        try:
            resp = c.get(_FRED_CSV_URL, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("fred fetch failed for {}: {}", series_id, exc)
            return []
        text = resp.text
        rows = _parse_csv(series_id, text)
        return rows
    finally:
        if owns_client:
            c.close()


def ingest_series(
    series: list[str],
    *,
    start: str | date | None = _DEFAULT_START,
    end: str | date | None = None,
    db_path: Path | None = None,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
    client: httpx.Client | None = None,
) -> dict[str, int]:
    """Fetch each series and upsert into DuckDB. Returns {series_id: row_count}."""
    store.create_schema(db_path=db_path)
    counts: dict[str, int] = {}
    owns_client = client is None
    c = client or httpx.Client(timeout=30.0)
    try:
        for i, sid in enumerate(series):
            if i > 0 and rate_limit_sec > 0:
                time.sleep(rate_limit_sec)
            rows = fetch_series(sid, start=start, end=end, client=c)
            n = store.upsert_fred_yields(rows, db_path=db_path) if rows else 0
            counts[sid] = n
            logger.info("fred {}: {} rows", sid, n)
    finally:
        if owns_client:
            c.close()
    total = sum(counts.values())
    logger.info("fred ingest summary: {} series, {} rows total", len(series), total)
    return counts


# ============================================================================ CLI
def _parse_csv_arg(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="casino.data.ingest_fred",
        description="Ingest FRED yield series via the key-less CSV endpoint.",
    )
    p.add_argument(
        "--series",
        type=str,
        default=",".join(_DEFAULT_SERIES),
        help=f"Comma-separated FRED series IDs (default: {','.join(_DEFAULT_SERIES)})",
    )
    p.add_argument("--start", type=str, default=_DEFAULT_START)
    p.add_argument("--end", type=str, default=None)
    p.add_argument(
        "--rate-limit-sec",
        type=float,
        default=_RATE_LIMIT_SEC,
        help="Seconds to sleep between series fetches (default 0.4)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    series = _parse_csv_arg(args.series)
    if not series:
        logger.error("no series resolved from --series argument")
        return 2
    counts = ingest_series(
        series,
        start=args.start,
        end=args.end,
        rate_limit_sec=args.rate_limit_sec,
    )
    if not any(counts.values()):
        logger.error("no FRED rows ingested; check network or series IDs")
        return 1
    return 0


def __getattr__(name: str) -> Any:  # pragma: no cover - convenience
    raise AttributeError(name)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
