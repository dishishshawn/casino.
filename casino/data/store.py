"""DuckDB data store — the *only* module that opens DuckDB.

Other modules import helpers from here. This concentrates point-in-time
correctness, connection lifetime, and Parquet archival in one place per
PRD §4.1 and CLAUDE.md.

Schemas:
- ohlcv             — daily bars (prices float64; PRD §10 allows float for OHLCV)
- fundamentals      — wide-format quarterly metrics
- earnings          — actual & consensus EPS per ticker × report date
- news              — news headlines
- filings           — SEC EDGAR filing metadata + extracted text
- transcripts       — earnings-call transcripts
- index_constituents — point-in-time index membership (survivorship-bias defense)

# Phase 2/3 note: when orders/fills/positions tables land (tasks 20-22), money
# columns will be VARCHAR (string-encoded Decimal) per CLAUDE.md "money is Decimal"
# rule. Those tables are NOT created here in Phase 1.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from loguru import logger

from casino.config import get_config

if TYPE_CHECKING:
    import pandas as pd

# Public schema definitions. Idempotent (CREATE TABLE IF NOT EXISTS).
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS ohlcv (
        ticker      VARCHAR     NOT NULL,
        ts          TIMESTAMPTZ NOT NULL,
        open        DOUBLE,
        high        DOUBLE,
        low         DOUBLE,
        close       DOUBLE,
        volume      BIGINT,
        adj_close   DOUBLE,
        PRIMARY KEY (ticker, ts)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fundamentals (
        ticker        VARCHAR     NOT NULL,
        report_date   TIMESTAMPTZ NOT NULL,
        metric_name   VARCHAR     NOT NULL,
        value         DOUBLE,
        PRIMARY KEY (ticker, report_date, metric_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS earnings (
        ticker         VARCHAR     NOT NULL,
        report_date    TIMESTAMPTZ NOT NULL,
        period_end     TIMESTAMPTZ,
        actual_eps     DOUBLE,
        consensus_eps  DOUBLE,
        revenue        DOUBLE,
        source         VARCHAR,
        PRIMARY KEY (ticker, report_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS news (
        id            VARCHAR     PRIMARY KEY,
        ticker        VARCHAR,
        headline      VARCHAR,
        published_at  TIMESTAMPTZ NOT NULL,
        source        VARCHAR,
        url           VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS filings (
        ticker            VARCHAR     NOT NULL,
        cik               VARCHAR,
        form_type         VARCHAR     NOT NULL,
        filing_date       TIMESTAMPTZ NOT NULL,
        accession_number  VARCHAR     NOT NULL,
        url               VARCHAR,
        full_text         VARCHAR,
        has_item_202      BOOLEAN     DEFAULT FALSE,
        raw_path          VARCHAR,
        PRIMARY KEY (ticker, accession_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transcripts (
        ticker          VARCHAR     NOT NULL,
        event_date      TIMESTAMPTZ NOT NULL,
        transcript_text VARCHAR,
        source          VARCHAR,
        PRIMARY KEY (ticker, event_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS finbert_scores (
        ticker        VARCHAR     NOT NULL,
        event_date    TIMESTAMPTZ NOT NULL,
        score_pos     DOUBLE,
        score_neu     DOUBLE,
        score_neg     DOUBLE,
        score_net     DOUBLE,
        n_chunks      INTEGER,
        model_name    VARCHAR,
        PRIMARY KEY (ticker, event_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS index_constituents (
        index_name VARCHAR     NOT NULL,
        ticker     VARCHAR     NOT NULL,
        added_on   TIMESTAMPTZ NOT NULL,
        removed_on TIMESTAMPTZ,
        PRIMARY KEY (index_name, ticker, added_on)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fred_yields (
        series_id VARCHAR     NOT NULL,
        ts        TIMESTAMPTZ NOT NULL,
        value     DOUBLE,
        PRIMARY KEY (series_id, ts)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dividends (
        ticker VARCHAR     NOT NULL,
        ts     TIMESTAMPTZ NOT NULL,
        amount DOUBLE      NOT NULL,
        PRIMARY KEY (ticker, ts)
    )
    """,
)


def _ensure_data_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_duckdb_conn(
    db_path: Path | None = None,
    *,
    read_only: bool = False,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a DuckDB connection with sane PRAGMAs.

    Auto-commits on clean exit, raises on exception (DuckDB doesn't support
    explicit rollback for non-transactional statements; users wanting txns
    should use BEGIN/COMMIT manually).
    """
    cfg = get_config()
    target = db_path if db_path is not None else cfg.duckdb_path
    _ensure_data_dir(target)
    conn = duckdb.connect(database=str(target), read_only=read_only)
    try:
        if not read_only:
            # DuckDB defaults threads to the number of physical cores; we just
            # cap memory so a runaway query can't exhaust the VPS.
            conn.execute("PRAGMA memory_limit='4GB'")
        yield conn
    finally:
        conn.close()


def create_schema(db_path: Path | None = None) -> None:
    """Create (or idempotently re-create) all Phase 1 tables."""
    with get_duckdb_conn(db_path) as conn:
        for stmt in _SCHEMA_STATEMENTS:
            conn.execute(stmt)
    logger.debug("create_schema completed for {}", db_path or get_config().duckdb_path)


def point_in_time_constituents(
    index_name: str,
    as_of_date: datetime,
    *,
    db_path: Path | None = None,
) -> list[str]:
    """Return tickers that were members of `index_name` on `as_of_date`.

    Survivorship-bias defense: queries the historical membership table, NOT a
    static "current S&P 500" list. A ticker is a member if it was added on or
    before `as_of_date` and either has no removal date or was removed strictly
    after `as_of_date`.

    Returns an empty list if the constituents table is empty or has no entries
    for the requested index — callers should treat empty as "no PIT data
    available" (warn and fall back).
    """
    sql = """
        SELECT DISTINCT ticker
        FROM index_constituents
        WHERE index_name = ?
          AND added_on <= ?
          AND (removed_on IS NULL OR removed_on > ?)
        ORDER BY ticker
    """
    with get_duckdb_conn(db_path, read_only=False) as conn:
        rows = conn.execute(sql, [index_name, as_of_date, as_of_date]).fetchall()
    return [r[0] for r in rows]


def archive_to_parquet(
    table_name: str,
    output_dir: Path,
    *,
    db_path: Path | None = None,
    compression: str = "zstd",
) -> Path:
    """Export a table to a compressed Parquet file in `output_dir`.

    Returns the path to the written file. Uses DuckDB native COPY TO.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    out_path = output_dir / f"{table_name}_{ts}.parquet"
    sql = f"COPY {table_name} TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION '{compression}')"
    with get_duckdb_conn(db_path) as conn:
        conn.execute(sql)
    logger.info("archived {} -> {}", table_name, out_path)
    return out_path


def upsert_ohlcv(rows: list[dict[str, object]], *, db_path: Path | None = None) -> int:
    """Idempotently upsert OHLCV rows. Returns count attempted."""
    if not rows:
        return 0
    sql = """
        INSERT OR REPLACE INTO ohlcv (ticker, ts, open, high, low, close, volume, adj_close)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_duckdb_conn(db_path) as conn:
        conn.executemany(
            sql,
            [
                (
                    r["ticker"],
                    r["ts"],
                    r.get("open"),
                    r.get("high"),
                    r.get("low"),
                    r.get("close"),
                    r.get("volume"),
                    r.get("adj_close"),
                )
                for r in rows
            ],
        )
    return len(rows)


def upsert_news(rows: list[dict[str, object]], *, db_path: Path | None = None) -> int:
    """Idempotently upsert news rows. Returns count attempted."""
    if not rows:
        return 0
    sql = """
        INSERT OR REPLACE INTO news (id, ticker, headline, published_at, source, url)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with get_duckdb_conn(db_path) as conn:
        conn.executemany(
            sql,
            [
                (
                    r["id"],
                    r.get("ticker"),
                    r.get("headline"),
                    r["published_at"],
                    r.get("source"),
                    r.get("url"),
                )
                for r in rows
            ],
        )
    return len(rows)


def upsert_fundamentals(
    rows: list[dict[str, object]],
    *,
    db_path: Path | None = None,
) -> int:
    """Idempotently upsert fundamentals rows."""
    if not rows:
        return 0
    sql = """
        INSERT OR REPLACE INTO fundamentals (ticker, report_date, metric_name, value)
        VALUES (?, ?, ?, ?)
    """
    with get_duckdb_conn(db_path) as conn:
        conn.executemany(
            sql,
            [(r["ticker"], r["report_date"], r["metric_name"], r.get("value")) for r in rows],
        )
    return len(rows)


def upsert_earnings(rows: list[dict[str, object]], *, db_path: Path | None = None) -> int:
    """Idempotently upsert earnings rows."""
    if not rows:
        return 0
    sql = """
        INSERT OR REPLACE INTO earnings
            (ticker, report_date, period_end, actual_eps, consensus_eps, revenue, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_duckdb_conn(db_path) as conn:
        conn.executemany(
            sql,
            [
                (
                    r["ticker"],
                    r["report_date"],
                    r.get("period_end"),
                    r.get("actual_eps"),
                    r.get("consensus_eps"),
                    r.get("revenue"),
                    r.get("source"),
                )
                for r in rows
            ],
        )
    return len(rows)


def upsert_filings(rows: list[dict[str, object]], *, db_path: Path | None = None) -> int:
    """Idempotently upsert SEC EDGAR filing rows."""
    if not rows:
        return 0
    sql = """
        INSERT OR REPLACE INTO filings
            (ticker, cik, form_type, filing_date, accession_number, url, full_text, has_item_202, raw_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_duckdb_conn(db_path) as conn:
        conn.executemany(
            sql,
            [
                (
                    r["ticker"],
                    r.get("cik"),
                    r["form_type"],
                    r["filing_date"],
                    r["accession_number"],
                    r.get("url"),
                    r.get("full_text"),
                    bool(r.get("has_item_202", False)),
                    r.get("raw_path"),
                )
                for r in rows
            ],
        )
    return len(rows)


def upsert_transcripts(rows: list[dict[str, object]], *, db_path: Path | None = None) -> int:
    """Idempotently upsert earnings-transcript rows on (ticker, event_date)."""
    if not rows:
        return 0
    sql = """
        INSERT OR REPLACE INTO transcripts (ticker, event_date, transcript_text, source)
        VALUES (?, ?, ?, ?)
    """
    with get_duckdb_conn(db_path) as conn:
        conn.executemany(
            sql,
            [
                (r["ticker"], r["event_date"], r.get("transcript_text"), r.get("source"))
                for r in rows
            ],
        )
    return len(rows)


def upsert_finbert_scores(
    rows: list[dict[str, object]],
    *,
    db_path: Path | None = None,
) -> int:
    """Idempotently upsert FinBERT score rows on (ticker, event_date)."""
    if not rows:
        return 0
    sql = """
        INSERT OR REPLACE INTO finbert_scores
            (ticker, event_date, score_pos, score_neu, score_neg, score_net, n_chunks, model_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_duckdb_conn(db_path) as conn:
        conn.executemany(
            sql,
            [
                (
                    r["ticker"],
                    r["event_date"],
                    r.get("score_pos"),
                    r.get("score_neu"),
                    r.get("score_neg"),
                    r.get("score_net"),
                    r.get("n_chunks"),
                    r.get("model_name"),
                )
                for r in rows
            ],
        )
    return len(rows)


def upsert_fred_yields(rows: list[dict[str, object]], *, db_path: Path | None = None) -> int:
    """Idempotently upsert FRED yield rows on (series_id, ts).

    Each row: {"series_id": str, "ts": datetime, "value": float | None}.
    """
    if not rows:
        return 0
    sql = """
        INSERT OR REPLACE INTO fred_yields (series_id, ts, value)
        VALUES (?, ?, ?)
    """
    with get_duckdb_conn(db_path) as conn:
        conn.executemany(
            sql,
            [(r["series_id"], r["ts"], r.get("value")) for r in rows],
        )
    return len(rows)


def load_fred_panel(
    *,
    series_ids: list[str],
    start: datetime,
    end: datetime,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Return a wide panel (date × series_id) of FRED yield values.

    Values are returned as decimals (e.g. 4.25 for 4.25%, matching the FRED CSV
    convention — callers wanting percent-of-1 should divide by 100).
    """
    import pandas as pd  # local to avoid hard dep at module import time

    if not series_ids:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(series_ids))
    sql = f"""
        SELECT series_id, ts, value
        FROM fred_yields
        WHERE ts BETWEEN ? AND ?
          AND series_id IN ({placeholders})
    """
    params: list[object] = [start, end, *series_ids]
    with get_duckdb_conn(db_path, read_only=True) as conn:
        df = conn.execute(sql, params).df()
    if df.empty:
        return pd.DataFrame()
    panel: pd.DataFrame = df.pivot(index="ts", columns="series_id", values="value").sort_index()
    return panel


def upsert_dividends(rows: list[dict[str, object]], *, db_path: Path | None = None) -> int:
    """Idempotently upsert ticker-dividend rows on (ticker, ts)."""
    if not rows:
        return 0
    sql = """
        INSERT OR REPLACE INTO dividends (ticker, ts, amount)
        VALUES (?, ?, ?)
    """
    with get_duckdb_conn(db_path) as conn:
        conn.executemany(
            sql,
            [(r["ticker"], r["ts"], r["amount"]) for r in rows],
        )
    return len(rows)


def load_dividends_panel(
    *,
    tickers: list[str],
    start: datetime,
    end: datetime,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Return long-format dividends DataFrame (ticker, ts, amount) for the universe."""
    import pandas as pd  # local

    if not tickers:
        return pd.DataFrame(columns=["ticker", "ts", "amount"])
    placeholders = ",".join("?" * len(tickers))
    sql = f"""
        SELECT ticker, ts, amount
        FROM dividends
        WHERE ts BETWEEN ? AND ?
          AND ticker IN ({placeholders})
        ORDER BY ticker, ts
    """
    params: list[object] = [start, end, *tickers]
    with get_duckdb_conn(db_path, read_only=True) as conn:
        df: pd.DataFrame = conn.execute(sql, params).df()
    return df


def upsert_index_constituent(
    *,
    index_name: str,
    ticker: str,
    added_on: datetime,
    removed_on: datetime | None = None,
    db_path: Path | None = None,
) -> None:
    """Insert or update a single index_constituents row."""
    with get_duckdb_conn(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO index_constituents (index_name, ticker, added_on, removed_on)
            VALUES (?, ?, ?, ?)
            """,
            [index_name, ticker, added_on, removed_on],
        )
