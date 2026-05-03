"""SQLite audit log for LLM calls.

Every call through `casino.llm.client.LLMClient` writes one row here. The
dashboard (Phase 3) reads from this table for the per-call ledger.

Schema lives in `state.sqlite` (PRD §3): "SQLite for orders and run state".
The LLM audit ledger is run state — keeping it out of DuckDB also avoids
contention with research queries.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from casino.config import get_config

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc         TEXT    NOT NULL,
    prompt_hash           TEXT    NOT NULL,
    model                 TEXT    NOT NULL,
    mode                  TEXT    NOT NULL,
    input_tokens          INTEGER NOT NULL,
    cache_creation_tokens INTEGER NOT NULL,
    cache_read_tokens     INTEGER NOT NULL,
    output_tokens         INTEGER NOT NULL,
    cost_usd              REAL    NOT NULL,
    latency_ms            INTEGER NOT NULL,
    parsed_score_json     TEXT,
    success               INTEGER NOT NULL,
    error_msg             TEXT,
    schema_name           TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_ts    ON llm_calls(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_llm_calls_hash  ON llm_calls(prompt_hash);
CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls(model);
"""


def _resolve_db_path(db_path: Path | None) -> Path:
    if db_path is not None:
        return db_path
    return get_config().state_sqlite_path


@contextmanager
def get_audit_conn(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection to the LLM audit DB. Creates parent dir."""
    target = _resolve_db_path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    finally:
        conn.close()


def init_audit_schema(db_path: Path | None = None) -> None:
    """Create the `llm_calls` table idempotently."""
    with get_audit_conn(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.debug("llm audit schema initialized at {}", _resolve_db_path(db_path))


def write_audit_row(
    *,
    prompt_hash: str,
    model: str,
    mode: str,
    input_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
    parsed_score: dict[str, Any] | None,
    success: bool,
    error_msg: str | None,
    schema_name: str | None,
    db_path: Path | None = None,
) -> int:
    """Insert one audit row. Returns the new row id."""
    init_audit_schema(db_path)
    sql = """
        INSERT INTO llm_calls (
            timestamp_utc, prompt_hash, model, mode,
            input_tokens, cache_creation_tokens, cache_read_tokens, output_tokens,
            cost_usd, latency_ms, parsed_score_json, success, error_msg, schema_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    ts = datetime.now(tz=UTC).isoformat()
    parsed_json = json.dumps(parsed_score) if parsed_score is not None else None
    with get_audit_conn(db_path) as conn:
        cur = conn.execute(
            sql,
            (
                ts,
                prompt_hash,
                model,
                mode,
                input_tokens,
                cache_creation_tokens,
                cache_read_tokens,
                output_tokens,
                cost_usd,
                latency_ms,
                parsed_json,
                1 if success else 0,
                error_msg,
                schema_name,
            ),
        )
        rowid = int(cur.lastrowid or 0)
    return rowid


def fetch_recent_calls(
    *,
    limit: int = 50,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the most-recent `limit` audit rows as dicts (newest first)."""
    init_audit_schema(db_path)
    sql = """
        SELECT id, timestamp_utc, prompt_hash, model, mode,
               input_tokens, cache_creation_tokens, cache_read_tokens, output_tokens,
               cost_usd, latency_ms, parsed_score_json, success, error_msg, schema_name
        FROM llm_calls
        ORDER BY id DESC
        LIMIT ?
    """
    with get_audit_conn(db_path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    cols = [
        "id",
        "timestamp_utc",
        "prompt_hash",
        "model",
        "mode",
        "input_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
        "output_tokens",
        "cost_usd",
        "latency_ms",
        "parsed_score_json",
        "success",
        "error_msg",
        "schema_name",
    ]
    return [dict(zip(cols, r, strict=True)) for r in rows]
