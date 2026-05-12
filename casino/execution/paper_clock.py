"""SQLite tables for the 30-day TSMOM paper-trade clock.

Branch C amendment 2026-05-07 (PRD §6.3 amendment): the original 3-month
paper window was reduced to a binding **30-day cap**. After day 30 the
runner emits a binary COMMIT-or-KILL verdict. To support that verdict we
persist three things alongside the existing `daily_pnl` / `orders` /
`positions` tables in ``state.sqlite``:

* ``paper_clock`` — one row per paper-trade run with start NAV, start
  date (UTC), the active config snapshot, and resolution status. Idempotent
  ``ensure_started`` lets the runner be invoked safely on every rebal day
  without duplicating the start record.
* ``kill_event`` — one row per kill criterion that ever fires. Stores the
  triggering criterion, value, threshold, NAV at kill, and timestamp so
  the day-30 verdict and the dashboard can show exactly why the run died.
* ``rebal_event`` — one row per completed monthly rebalance. The day-30
  verdict checks ``rebals_completed >= 1`` as a COMMIT precondition.

Schema decisions:

* All money values stored as canonical Decimal strings (PRD §10 / CLAUDE.md
  "money is Decimal") — same convention as ``casino/execution/book.py``.
* All timestamps stored as ISO-8601 UTC strings.
* The schema reuses ``casino.execution.book.get_book_conn`` so
  ``state.sqlite`` stays the single SQLite file for all execution state.
* ``paper_clock`` is intentionally a single-row table (PK ``run_id``);
  the runner uses a fixed ``run_id="DiCaprio"`` so re-running the runner
  on day 7 doesn't reset the clock.

Imported by ``casino.execution.tsmom_runner`` (writes start row + rebal
events) and ``casino.execution.tsmom_clock_check`` (reads everything; writes
kill events; writes the verdict row).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from loguru import logger

from casino.execution import book

# The runner uses this constant `run_id` so re-invocations on subsequent
# rebal days append rebal_event rows instead of resetting the clock.
DEFAULT_RUN_ID = "DiCaprio"

# 30-day cap (Branch C amendment 2026-05-07).
PAPER_CAP_DAYS = 30

# Kill-criterion canonical names. Tests + the verdict script assert on
# these literal strings.
KILL_CRITERIA: tuple[str, ...] = (
    "drawdown",
    "single_day",
    "cap_violation",
    "reconcile_drift",
    "ks_test",
)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paper_clock (
    run_id              TEXT    PRIMARY KEY,
    strategy            TEXT    NOT NULL,
    started_at_utc      TEXT    NOT NULL,
    start_nav           TEXT    NOT NULL,
    cap_days            INTEGER NOT NULL,
    config_json         TEXT,
    verdict             TEXT,                 -- NULL | 'COMMIT' | 'KILL'
    verdict_at_utc      TEXT,
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS kill_event (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL,
    fired_at_utc        TEXT    NOT NULL,
    criterion           TEXT    NOT NULL,
    value               TEXT    NOT NULL,
    threshold           TEXT    NOT NULL,
    nav_at_kill         TEXT    NOT NULL,
    detail              TEXT
);
CREATE INDEX IF NOT EXISTS idx_kill_event_run ON kill_event(run_id);

CREATE TABLE IF NOT EXISTS rebal_event (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL,
    rebal_at_utc        TEXT    NOT NULL,
    n_orders_submitted  INTEGER NOT NULL,
    nav_at_rebal        TEXT    NOT NULL,
    target_weights_json TEXT    NOT NULL,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_rebal_event_run ON rebal_event(run_id);
"""


# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class PaperClockRow:
    """One row of the paper_clock table."""

    run_id: str
    strategy: str
    started_at_utc: datetime
    start_nav: Decimal
    cap_days: int
    config_json: str | None
    verdict: str | None  # None | "COMMIT" | "KILL"
    verdict_at_utc: datetime | None
    notes: str | None


@dataclass(frozen=True)
class KillEventRow:
    """One row of the kill_event table."""

    id: int
    run_id: str
    fired_at_utc: datetime
    criterion: str
    value: Decimal
    threshold: Decimal
    nav_at_kill: Decimal
    detail: str | None


@dataclass(frozen=True)
class RebalEventRow:
    """One row of the rebal_event table."""

    id: int
    run_id: str
    rebal_at_utc: datetime
    n_orders_submitted: int
    nav_at_rebal: Decimal
    target_weights_json: str
    notes: str | None


# ---------------------------------------------------------------------------- helpers


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def init_schema(db_path: Path | None = None) -> None:
    """Idempotently create the paper_clock / kill_event / rebal_event tables."""
    book.init_schema(db_path)  # ensure base schema (run_state etc.) exists too
    with book.get_book_conn(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)


# ---------------------------------------------------------------------------- paper_clock


def ensure_started(
    *,
    run_id: str = DEFAULT_RUN_ID,
    strategy: str = "tsmom_long_only",
    start_nav: Decimal,
    cap_days: int = PAPER_CAP_DAYS,
    config_json: str | None = None,
    db_path: Path | None = None,
) -> PaperClockRow:
    """Insert the start row if it doesn't exist; return the row.

    Idempotent — calling on day 7 returns the day-0 row unchanged. The
    runner calls this at the top of every monthly rebal so the clock is
    *guaranteed* to have started, and we never lose the start NAV even
    if the dashboard is the first thing the operator opens.
    """
    init_schema(db_path)
    with book.get_book_conn(db_path) as conn:
        row = conn.execute(
            "SELECT run_id FROM paper_clock WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO paper_clock (
                    run_id, strategy, started_at_utc, start_nav, cap_days,
                    config_json, verdict, verdict_at_utc, notes
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    run_id,
                    strategy,
                    _utc_now_iso(),
                    str(start_nav),
                    int(cap_days),
                    config_json,
                ),
            )
            logger.warning(
                "paper_clock: STARTED run_id={} strategy={} start_nav=${} cap={}d",
                run_id,
                strategy,
                start_nav,
                cap_days,
            )
    fetched = fetch_paper_clock(run_id=run_id, db_path=db_path)
    if fetched is None:
        raise RuntimeError("paper_clock: ensure_started failed to persist row")
    return fetched


def fetch_paper_clock(
    *,
    run_id: str = DEFAULT_RUN_ID,
    db_path: Path | None = None,
) -> PaperClockRow | None:
    """Return the paper_clock row for ``run_id`` or None."""
    init_schema(db_path)
    with book.get_book_conn(db_path) as conn:
        r = conn.execute(
            """
            SELECT run_id, strategy, started_at_utc, start_nav, cap_days,
                   config_json, verdict, verdict_at_utc, notes
            FROM paper_clock
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    if r is None:
        return None
    return PaperClockRow(
        run_id=str(r[0]),
        strategy=str(r[1]),
        started_at_utc=_parse_iso(str(r[2])),
        start_nav=Decimal(str(r[3])),
        cap_days=int(r[4]),
        config_json=str(r[5]) if r[5] is not None else None,
        verdict=str(r[6]) if r[6] is not None else None,
        verdict_at_utc=_parse_iso(str(r[7])) if r[7] is not None else None,
        notes=str(r[8]) if r[8] is not None else None,
    )


def set_verdict(
    *,
    verdict: str,
    run_id: str = DEFAULT_RUN_ID,
    notes: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Persist the day-30 COMMIT/KILL verdict.

    Idempotent: re-running with the same verdict updates the timestamp;
    the dashboard always reads the latest.
    """
    if verdict not in ("COMMIT", "KILL"):
        raise ValueError(f"verdict must be 'COMMIT' or 'KILL', got {verdict!r}")
    init_schema(db_path)
    with book.get_book_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE paper_clock
               SET verdict = ?, verdict_at_utc = ?, notes = COALESCE(?, notes)
             WHERE run_id = ?
            """,
            (verdict, _utc_now_iso(), notes, run_id),
        )
    logger.warning("paper_clock: verdict={} run_id={}", verdict, run_id)


def days_elapsed(
    *,
    now: datetime | None = None,
    run_id: str = DEFAULT_RUN_ID,
    db_path: Path | None = None,
) -> int | None:
    """Return integer days elapsed since the clock started, or None if not started."""
    row = fetch_paper_clock(run_id=run_id, db_path=db_path)
    if row is None:
        return None
    cur = now if now is not None else _utc_now()
    return (cur - row.started_at_utc).days


# ---------------------------------------------------------------------------- kill_event


def insert_kill_event(
    *,
    criterion: str,
    value: Decimal,
    threshold: Decimal,
    nav_at_kill: Decimal,
    detail: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    db_path: Path | None = None,
) -> int:
    """Persist one kill-criterion firing. Returns the new row id."""
    if criterion not in KILL_CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; must be one of {KILL_CRITERIA}")
    init_schema(db_path)
    with book.get_book_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO kill_event (
                run_id, fired_at_utc, criterion, value, threshold, nav_at_kill, detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                _utc_now_iso(),
                criterion,
                str(value),
                str(threshold),
                str(nav_at_kill),
                detail,
            ),
        )
        return int(cur.lastrowid or 0)


def fetch_kill_events(
    *,
    run_id: str = DEFAULT_RUN_ID,
    db_path: Path | None = None,
) -> list[KillEventRow]:
    init_schema(db_path)
    with book.get_book_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, run_id, fired_at_utc, criterion, value, threshold,
                   nav_at_kill, detail
            FROM kill_event
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
    return [
        KillEventRow(
            id=int(r[0]),
            run_id=str(r[1]),
            fired_at_utc=_parse_iso(str(r[2])),
            criterion=str(r[3]),
            value=Decimal(str(r[4])),
            threshold=Decimal(str(r[5])),
            nav_at_kill=Decimal(str(r[6])),
            detail=str(r[7]) if r[7] is not None else None,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------- rebal_event


def insert_rebal_event(
    *,
    n_orders_submitted: int,
    nav_at_rebal: Decimal,
    target_weights_json: str,
    notes: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    db_path: Path | None = None,
) -> int:
    init_schema(db_path)
    with book.get_book_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO rebal_event (
                run_id, rebal_at_utc, n_orders_submitted, nav_at_rebal,
                target_weights_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                _utc_now_iso(),
                int(n_orders_submitted),
                str(nav_at_rebal),
                target_weights_json,
                notes,
            ),
        )
        return int(cur.lastrowid or 0)


def fetch_rebal_events(
    *,
    run_id: str = DEFAULT_RUN_ID,
    db_path: Path | None = None,
) -> list[RebalEventRow]:
    init_schema(db_path)
    with book.get_book_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, run_id, rebal_at_utc, n_orders_submitted,
                   nav_at_rebal, target_weights_json, notes
            FROM rebal_event
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
    return [
        RebalEventRow(
            id=int(r[0]),
            run_id=str(r[1]),
            rebal_at_utc=_parse_iso(str(r[2])),
            n_orders_submitted=int(r[3]),
            nav_at_rebal=Decimal(str(r[4])),
            target_weights_json=str(r[5]),
            notes=str(r[6]) if r[6] is not None else None,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------- helpers for the verdict


def is_last_business_day_of_month(d: date) -> bool:
    """Return True iff `d` is the last Mon-Fri date of its calendar month.

    Defined with weekday rules only (we do NOT consult an exchange holiday
    calendar; market-closure handling is the runner's job via the broker
    clock). For paper trading this is good enough — the runner additionally
    verifies the market is actually open before submitting.
    """
    nxt = d + timedelta(days=1)
    if nxt.month != d.month:  # last calendar day
        return d.weekday() < 5
    # else, walk forward looking for any later weekday in same month
    cur = d + timedelta(days=1)
    while cur.month == d.month:
        if cur.weekday() < 5:
            return False
        cur = cur + timedelta(days=1)
    return d.weekday() < 5
