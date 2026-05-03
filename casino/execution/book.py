"""SQLite order/fill/position book + run-state flags.

This is the *internal* book — what `casino` thinks it holds. The broker is
queried separately and the two are reconciled in `casino.execution.reconcile`
(CLAUDE.md §4.1: "reconcile is the source of truth for what we actually
hold"). The book lives in `state.sqlite` alongside the LLM audit log
(PRD §3 — SQLite is for orders + run state, DuckDB is for market data).

PRD §10 / CLAUDE.md hard rule "money is Decimal":
* All monetary columns are `VARCHAR` storing canonical Decimal strings
  (`str(Decimal(...))`).
* Quantities are integers (whole-share strategy in v1).
* All timestamps stored as ISO-8601 UTC strings.

The schema is intentionally minimal — orders, fills, positions, and a
`run_state` key/value table for flags like `trading_disabled` (the kill
switch's effect). Daily P&L lives in its own table written by the EOD
reconcile job.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from loguru import logger

from casino.config import get_config

OrderSide = Literal["buy", "sell"]
PositionSide = Literal["long", "short"]


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS orders (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    broker_order_id     TEXT    NOT NULL UNIQUE,
    client_order_id     TEXT,
    symbol              TEXT    NOT NULL,
    side                TEXT    NOT NULL,
    qty                 INTEGER NOT NULL,
    stop_price          TEXT    NOT NULL,
    limit_price         TEXT,
    submitted_at_utc    TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    notional_estimate   TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    broker_order_id TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    qty             INTEGER NOT NULL,
    price           TEXT    NOT NULL,
    filled_at_utc   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);

CREATE TABLE IF NOT EXISTS positions (
    symbol           TEXT PRIMARY KEY,
    side             TEXT NOT NULL,
    qty              INTEGER NOT NULL,
    avg_entry_price  TEXT NOT NULL,
    opened_at_utc    TEXT NOT NULL,
    last_update_utc  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_state (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    updated_at_utc  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    date              TEXT PRIMARY KEY,  -- YYYY-MM-DD UTC
    equity_open       TEXT NOT NULL,
    equity_close      TEXT NOT NULL,
    realized_pl       TEXT NOT NULL,
    unrealized_pl     TEXT NOT NULL,
    n_positions       INTEGER NOT NULL,
    n_orders          INTEGER NOT NULL,
    notes             TEXT
);
"""


# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class StoredOrder:
    """One row in the orders table — what we sent to the broker."""

    broker_order_id: str
    client_order_id: str | None
    symbol: str
    side: OrderSide
    qty: int
    stop_price: Decimal
    limit_price: Decimal | None
    submitted_at_utc: datetime
    status: str
    notional_estimate: Decimal | None


@dataclass(frozen=True)
class StoredPosition:
    """One row in the positions table — internal book view."""

    symbol: str
    side: PositionSide
    qty: int
    avg_entry_price: Decimal
    opened_at_utc: datetime
    last_update_utc: datetime


@dataclass(frozen=True)
class DailyPnLRow:
    """One row in the daily_pnl table."""

    date: str  # YYYY-MM-DD
    equity_open: Decimal
    equity_close: Decimal
    realized_pl: Decimal
    unrealized_pl: Decimal
    n_positions: int
    n_orders: int
    notes: str | None


# ---------------------------------------------------------------------------- connection


def _resolve_db_path(db_path: Path | None) -> Path:
    return db_path if db_path is not None else get_config().state_sqlite_path


@contextmanager
def get_book_conn(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection to the book DB. Creates parent dir."""
    target = _resolve_db_path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    finally:
        conn.close()


def init_schema(db_path: Path | None = None) -> None:
    """Idempotently create all execution tables."""
    with get_book_conn(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.debug("execution book schema initialized at {}", _resolve_db_path(db_path))


# ---------------------------------------------------------------------------- helpers


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _decimal_str(v: Decimal | None) -> str | None:
    return str(v) if v is not None else None


def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _decimal_or_none(v: object) -> Decimal | None:
    if v is None:
        return None
    s = str(v)
    if s == "" or s.lower() == "none":
        return None
    return Decimal(s)


# ---------------------------------------------------------------------------- orders


def insert_order(
    *,
    broker_order_id: str,
    client_order_id: str | None,
    symbol: str,
    side: OrderSide,
    qty: int,
    stop_price: Decimal,
    limit_price: Decimal | None,
    submitted_at_utc: datetime | None,
    status: str,
    notional_estimate: Decimal | None,
    db_path: Path | None = None,
) -> int:
    """Insert one order row. Returns the new row id."""
    init_schema(db_path)
    ts = submitted_at_utc.isoformat() if submitted_at_utc is not None else _utc_now_iso()
    sql = """
        INSERT INTO orders (
            broker_order_id, client_order_id, symbol, side, qty,
            stop_price, limit_price, submitted_at_utc, status, notional_estimate
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_book_conn(db_path) as conn:
        cur = conn.execute(
            sql,
            (
                broker_order_id,
                client_order_id,
                symbol.upper(),
                side,
                int(qty),
                str(stop_price),
                _decimal_str(limit_price),
                ts,
                status,
                _decimal_str(notional_estimate),
            ),
        )
        return int(cur.lastrowid or 0)


def update_order_status(
    *,
    broker_order_id: str,
    status: str,
    db_path: Path | None = None,
) -> None:
    init_schema(db_path)
    with get_book_conn(db_path) as conn:
        conn.execute(
            "UPDATE orders SET status = ? WHERE broker_order_id = ?",
            (status, broker_order_id),
        )


def fetch_open_orders(db_path: Path | None = None) -> list[StoredOrder]:
    """Return orders whose status indicates they're still working."""
    init_schema(db_path)
    open_statuses = (
        "new",
        "accepted",
        "pending_new",
        "partially_filled",
        "held",
        "calculated",
        "replaced",
    )
    placeholders = ",".join(["?"] * len(open_statuses))
    sql = f"""
        SELECT broker_order_id, client_order_id, symbol, side, qty, stop_price,
               limit_price, submitted_at_utc, status, notional_estimate
        FROM orders
        WHERE status IN ({placeholders})
        ORDER BY id ASC
    """
    with get_book_conn(db_path) as conn:
        rows = conn.execute(sql, open_statuses).fetchall()
    return [
        StoredOrder(
            broker_order_id=str(r[0]),
            client_order_id=str(r[1]) if r[1] is not None else None,
            symbol=str(r[2]),
            side="sell" if str(r[3]) == "sell" else "buy",
            qty=int(r[4]),
            stop_price=Decimal(str(r[5])),
            limit_price=_decimal_or_none(r[6]),
            submitted_at_utc=_parse_iso(str(r[7])),
            status=str(r[8]),
            notional_estimate=_decimal_or_none(r[9]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------- fills


def insert_fill(
    *,
    broker_order_id: str,
    symbol: str,
    side: OrderSide,
    qty: int,
    price: Decimal,
    filled_at_utc: datetime | None = None,
    db_path: Path | None = None,
) -> int:
    init_schema(db_path)
    ts = filled_at_utc.isoformat() if filled_at_utc is not None else _utc_now_iso()
    sql = """
        INSERT INTO fills (broker_order_id, symbol, side, qty, price, filled_at_utc)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with get_book_conn(db_path) as conn:
        cur = conn.execute(
            sql,
            (broker_order_id, symbol.upper(), side, int(qty), str(price), ts),
        )
        return int(cur.lastrowid or 0)


# ---------------------------------------------------------------------------- positions


def upsert_position(
    *,
    symbol: str,
    side: PositionSide,
    qty: int,
    avg_entry_price: Decimal,
    opened_at_utc: datetime | None = None,
    db_path: Path | None = None,
) -> None:
    init_schema(db_path)
    now = _utc_now_iso()
    opened = opened_at_utc.isoformat() if opened_at_utc is not None else now
    sql = """
        INSERT INTO positions (symbol, side, qty, avg_entry_price, opened_at_utc, last_update_utc)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            side            = excluded.side,
            qty             = excluded.qty,
            avg_entry_price = excluded.avg_entry_price,
            last_update_utc = excluded.last_update_utc
    """
    with get_book_conn(db_path) as conn:
        conn.execute(
            sql,
            (symbol.upper(), side, int(qty), str(avg_entry_price), opened, now),
        )


def delete_position(symbol: str, *, db_path: Path | None = None) -> None:
    init_schema(db_path)
    with get_book_conn(db_path) as conn:
        conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol.upper(),))


def fetch_positions(db_path: Path | None = None) -> list[StoredPosition]:
    init_schema(db_path)
    sql = """
        SELECT symbol, side, qty, avg_entry_price, opened_at_utc, last_update_utc
        FROM positions
        ORDER BY symbol ASC
    """
    with get_book_conn(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [
        StoredPosition(
            symbol=str(r[0]),
            side="short" if str(r[1]) == "short" else "long",
            qty=int(r[2]),
            avg_entry_price=Decimal(str(r[3])),
            opened_at_utc=_parse_iso(str(r[4])),
            last_update_utc=_parse_iso(str(r[5])),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------- run_state


TRADING_DISABLED_KEY = "trading_disabled"


def set_state(key: str, value: str, *, db_path: Path | None = None) -> None:
    init_schema(db_path)
    sql = """
        INSERT INTO run_state (key, value, updated_at_utc)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                       updated_at_utc = excluded.updated_at_utc
    """
    with get_book_conn(db_path) as conn:
        conn.execute(sql, (key, value, _utc_now_iso()))


def get_state(key: str, *, db_path: Path | None = None) -> str | None:
    init_schema(db_path)
    with get_book_conn(db_path) as conn:
        row = conn.execute("SELECT value FROM run_state WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def is_trading_disabled(db_path: Path | None = None) -> bool:
    """Return True iff the kill switch has flipped the disable flag."""
    val = get_state(TRADING_DISABLED_KEY, db_path=db_path)
    return val == "1"


def set_trading_disabled(
    disabled: bool,
    *,
    reason: str | None = None,
    db_path: Path | None = None,
) -> None:
    set_state(TRADING_DISABLED_KEY, "1" if disabled else "0", db_path=db_path)
    if reason is not None:
        set_state(f"{TRADING_DISABLED_KEY}_reason", reason, db_path=db_path)


# ---------------------------------------------------------------------------- daily_pnl


def upsert_daily_pnl(row: DailyPnLRow, *, db_path: Path | None = None) -> None:
    init_schema(db_path)
    sql = """
        INSERT INTO daily_pnl (date, equity_open, equity_close, realized_pl,
                               unrealized_pl, n_positions, n_orders, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            equity_open   = excluded.equity_open,
            equity_close  = excluded.equity_close,
            realized_pl   = excluded.realized_pl,
            unrealized_pl = excluded.unrealized_pl,
            n_positions   = excluded.n_positions,
            n_orders      = excluded.n_orders,
            notes         = excluded.notes
    """
    with get_book_conn(db_path) as conn:
        conn.execute(
            sql,
            (
                row.date,
                str(row.equity_open),
                str(row.equity_close),
                str(row.realized_pl),
                str(row.unrealized_pl),
                int(row.n_positions),
                int(row.n_orders),
                row.notes,
            ),
        )


def fetch_daily_pnl(*, limit: int = 365, db_path: Path | None = None) -> list[DailyPnLRow]:
    init_schema(db_path)
    sql = """
        SELECT date, equity_open, equity_close, realized_pl, unrealized_pl,
               n_positions, n_orders, notes
        FROM daily_pnl
        ORDER BY date DESC
        LIMIT ?
    """
    with get_book_conn(db_path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [
        DailyPnLRow(
            date=str(r[0]),
            equity_open=Decimal(str(r[1])),
            equity_close=Decimal(str(r[2])),
            realized_pl=Decimal(str(r[3])),
            unrealized_pl=Decimal(str(r[4])),
            n_positions=int(r[5]),
            n_orders=int(r[6]),
            notes=str(r[7]) if r[7] is not None else None,
        )
        for r in rows
    ]
