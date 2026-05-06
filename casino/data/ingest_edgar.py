"""SEC EDGAR ingestion CLI: fetch filings, parse, store in DuckDB.

CLI:
    uv run python -m casino.data.ingest_edgar --ticker AAPL --form 8-K --days 7
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from loguru import logger

from casino.config import get_config
from casino.data import store
from casino.data.edgar_client import EdgarClient
from casino.data.edgar_parser import parse_filing_document


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="casino.data.ingest_edgar",
        description="Ingest SEC EDGAR filings (8-K, 10-K, 10-Q, Form 4) into DuckDB.",
    )
    p.add_argument("--ticker", required=True, help="Ticker symbol")
    p.add_argument(
        "--form",
        action="append",
        required=True,
        help="Form type to fetch (repeatable). e.g. 8-K 10-K 10-Q '4'",
    )
    p.add_argument("--days", type=int, default=7, help="Look-back window in days")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    return p


def _to_utc_midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def ingest_for_ticker(
    ticker: str,
    *,
    forms: list[str],
    start_date: date,
    end_date: date,
    client: EdgarClient | None = None,
    db_path: Path | None = None,
) -> int:
    """Fetch filings for one ticker and upsert to the filings table.

    Returns the number of rows upserted.
    """
    owns_client = client is None
    if client is None:
        client = EdgarClient()
    try:
        cik = client.get_cik_from_ticker(ticker)
        meta = client.search_filings(
            cik,
            form_types=forms,
            start_date=start_date,
            end_date=end_date,
        )
        if not meta:
            logger.info(
                "no filings for {} forms={} window={}..{}", ticker, forms, start_date, end_date
            )
            return 0

        rows: list[dict[str, object]] = []
        for entry in meta:
            url = str(entry["url"])
            try:
                raw = client.fetch_document(url)
            except Exception as e:  # noqa: BLE001 — log and continue
                logger.warning("edgar fetch failed for {}: {}", url, e)
                continue
            raw_path = client.store_raw(
                raw,
                ticker=ticker,
                form_type=str(entry["form_type"]),
                accession=str(entry["accession_number"]),
            )
            parsed = parse_filing_document(raw, str(entry["form_type"]))
            filing_date = entry["filing_date"]
            assert isinstance(filing_date, date)
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "cik": entry.get("cik"),
                    "form_type": entry["form_type"],
                    "filing_date": _to_utc_midnight(filing_date),
                    "accession_number": entry["accession_number"],
                    "url": url,
                    "full_text": parsed["text"],
                    "has_item_202": parsed["has_item_202"],
                    "raw_path": raw_path,
                }
            )

        store.upsert_filings(rows, db_path=db_path)
        return len(rows)
    finally:
        if owns_client:
            client.close()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = get_config()
    if not cfg.sec_user_agent or "@" not in cfg.sec_user_agent:
        logger.error("SEC_USER_AGENT must include a real contact email")
        return 2

    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=args.days)
    store.create_schema()
    n = ingest_for_ticker(args.ticker, forms=args.form, start_date=start, end_date=end)
    logger.info("ingested {} filings for {}", n, args.ticker)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
