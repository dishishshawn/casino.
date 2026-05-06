"""Earnings-call transcript ingestion from free Hugging Face datasets.

Used for the Phase 1 Stage 1 text-PEAD kill-test (memory: fmp_decision_2026-05-05).
Pulls transcripts for S&P 500 names from one of:

    1. kurry/sp500_earnings_transcripts        — primary, MIT, ~33k rows, 2005-2025
    2. glopardo/sp500-earnings-transcripts     — fallback, ECB WP 3093, 2014-2024

Both are one-time snapshots — sufficient to *backtest* a text-PEAD signal but NOT
to live-trade with. Empirical kill threshold: Q4-Q0 20-day forward-return spread <
+1.0% gross → permanently kill the EDLLLS strategy on US large caps.

CLI:
    uv run python -m casino.data.ingest_transcripts_hf
    uv run python -m casino.data.ingest_transcripts_hf --dataset glopardo/sp500-earnings-transcripts
    uv run python -m casino.data.ingest_transcripts_hf --tickers AAPL,MSFT --limit 100
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from casino.data import store

_DEFAULT_DATASET = "kurry/sp500_earnings_transcripts"
_FALLBACK_DATASET = "glopardo/sp500-earnings-transcripts"

# Column-name heuristics: HF dataset schemas vary between authors. Try several
# common variants before giving up.
_TICKER_COLS = ("ticker", "symbol", "Symbol", "company_ticker", "stock")
_DATE_COLS = (
    "date",
    "call_date",
    "event_date",
    "report_date",
    "Date",
    "earnings_date",
    "transcript_date",
)
_TEXT_COLS = (
    "transcript",
    "text",
    "transcript_text",
    "content",
    "full_transcript",
    "body",
)


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _to_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if bool(pd.isna(ts)):
        return None
    py: datetime = ts.to_pydatetime()
    return py if py.tzinfo else py.replace(tzinfo=UTC)


def _normalize(df: pd.DataFrame, source: str) -> list[dict[str, object]]:
    ticker_col = _pick_col(df, _TICKER_COLS)
    date_col = _pick_col(df, _DATE_COLS)
    text_col = _pick_col(df, _TEXT_COLS)
    if not (ticker_col and date_col and text_col):
        raise RuntimeError(
            f"could not locate required columns in dataset; "
            f"have={list(df.columns)} need ticker={_TICKER_COLS}, date={_DATE_COLS}, text={_TEXT_COLS}"
        )
    logger.info("transcripts schema: ticker={}, date={}, text={}", ticker_col, date_col, text_col)

    rows: list[dict[str, object]] = []
    for _, r in df.iterrows():
        ticker = str(r[ticker_col]).strip().upper()
        if not ticker or ticker in ("NAN", "NONE"):
            continue
        event_date = _to_utc(r[date_col])
        if event_date is None:
            continue
        text = r[text_col]
        if text is None or (isinstance(text, float) and pd.isna(text)):
            continue
        text_str = str(text).strip()
        if not text_str:
            continue
        rows.append(
            {
                "ticker": ticker,
                "event_date": event_date,
                "transcript_text": text_str,
                "source": source,
            }
        )
    return rows


def fetch_dataset(dataset: str = _DEFAULT_DATASET) -> pd.DataFrame:
    """Download the HF dataset and return a single concatenated pandas DataFrame.

    Imports `datasets` lazily so the rest of the project doesn't pay the cost
    until this ingester actually runs.
    """
    from datasets import load_dataset

    logger.info("loading HF dataset {} (this may take a few minutes on first run)", dataset)
    ds = load_dataset(dataset)
    frames = [pd.DataFrame(ds[split]) for split in ds.keys()]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info("loaded {} rows from {}", len(df), dataset)
    return df


def ingest(
    dataset: str = _DEFAULT_DATASET,
    *,
    tickers: list[str] | None = None,
    limit: int | None = None,
    db_path: Path | None = None,
) -> int:
    """Pull transcripts and upsert to DuckDB. Returns rows written."""
    store.create_schema(db_path=db_path)
    df = fetch_dataset(dataset)
    if df.empty:
        logger.warning("dataset {} returned no rows", dataset)
        return 0

    rows = _normalize(df, source=dataset)
    if tickers:
        wanted = {t.upper() for t in tickers}
        rows = [r for r in rows if r["ticker"] in wanted]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        logger.warning("no transcript rows after filtering")
        return 0

    n = store.upsert_transcripts(rows, db_path=db_path)
    logger.info("upserted {} transcripts ({} unique tickers)", n, len({r["ticker"] for r in rows}))
    return n


# ============================================================================ CLI
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="casino.data.ingest_transcripts_hf",
        description="Ingest earnings-call transcripts from a free Hugging Face dataset.",
    )
    p.add_argument(
        "--dataset",
        default=_DEFAULT_DATASET,
        help=f"HF dataset id (default {_DEFAULT_DATASET}; fallback {_FALLBACK_DATASET})",
    )
    p.add_argument("--tickers", default=None, help="Comma-separated tickers to filter (optional)")
    p.add_argument("--limit", type=int, default=None, help="Cap rows for smoke-test runs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    )
    n = ingest(args.dataset, tickers=tickers, limit=args.limit)
    return 0 if n > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
