"""Tests for casino.execution.paper_clock — schema + CRUD round-trips.

Covers:
* Schema is idempotently re-creatable (calling ``init_schema`` twice is a
  no-op).
* ``ensure_started`` is idempotent — re-invocation with the same run_id
  preserves the original start row and timestamp.
* PK on ``paper_clock.run_id`` enforced.
* ``insert_kill_event`` rejects unknown criteria; round-trips known ones.
* ``insert_rebal_event`` round-trips with ordered fetch.
* ``set_verdict`` rejects bad values, persists good ones.
* ``days_elapsed`` arithmetic.
* ``is_last_business_day_of_month`` weekday rules.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from casino.config import get_config
from casino.execution import book, paper_clock


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated SQLite per test."""
    p = tmp_path / "state.sqlite"
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(p))
    get_config.cache_clear()
    book.init_schema(p)
    paper_clock.init_schema(p)
    return p


# ---------------------------------------------------------------------------- schema


def test_init_schema_is_idempotent(state: Path) -> None:
    """Re-running init_schema must not raise or wipe data."""
    paper_clock.init_schema(state)
    paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    paper_clock.init_schema(state)  # second call must not drop the row
    assert paper_clock.fetch_paper_clock(db_path=state) is not None


def test_paper_clock_pk_enforced(state: Path) -> None:
    """run_id is the PK; a duplicate insert at the SQL level raises IntegrityError."""
    import sqlite3

    paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    with book.get_book_conn(state) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO paper_clock (run_id, strategy, started_at_utc, "
                "start_nav, cap_days) VALUES (?, ?, ?, ?, ?)",
                (
                    paper_clock.DEFAULT_RUN_ID,
                    "tsmom_long_only",
                    datetime.now(tz=UTC).isoformat(),
                    "100000",
                    30,
                ),
            )


# ---------------------------------------------------------------------------- ensure_started


def test_ensure_started_creates_row(state: Path) -> None:
    row = paper_clock.ensure_started(
        start_nav=Decimal("100000.50"),
        config_json='{"foo": 1}',
        db_path=state,
    )
    assert row.run_id == paper_clock.DEFAULT_RUN_ID
    assert row.start_nav == Decimal("100000.50")
    assert row.cap_days == paper_clock.PAPER_CAP_DAYS
    assert row.config_json == '{"foo": 1}'
    assert row.verdict is None


def test_ensure_started_is_idempotent(state: Path) -> None:
    """Calling on day 7 must not reset the day-0 row."""
    first = paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    # Second call with a *different* start_nav must NOT clobber.
    second = paper_clock.ensure_started(start_nav=Decimal("999999"), db_path=state)
    assert first.started_at_utc == second.started_at_utc
    assert second.start_nav == Decimal("100000")


# ---------------------------------------------------------------------------- kill_event


def test_insert_kill_event_rejects_unknown_criterion(state: Path) -> None:
    paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    with pytest.raises(ValueError):
        paper_clock.insert_kill_event(
            criterion="not_a_real_one",
            value=Decimal("0.5"),
            threshold=Decimal("0.1"),
            nav_at_kill=Decimal("100000"),
            db_path=state,
        )


def test_insert_and_fetch_kill_events(state: Path) -> None:
    paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    for crit in ("drawdown", "single_day", "ks_test"):
        paper_clock.insert_kill_event(
            criterion=crit,
            value=Decimal("0.123"),
            threshold=Decimal("0.10"),
            nav_at_kill=Decimal("90000"),
            detail=f"detail-{crit}",
            db_path=state,
        )
    rows = paper_clock.fetch_kill_events(db_path=state)
    assert len(rows) == 3
    assert [r.criterion for r in rows] == ["drawdown", "single_day", "ks_test"]
    assert all(r.run_id == paper_clock.DEFAULT_RUN_ID for r in rows)
    assert all(r.value == Decimal("0.123") for r in rows)


def test_kill_criteria_constant_matches_amendment() -> None:
    """The five criteria are the canonical ones documented in the PRD amendment."""
    assert set(paper_clock.KILL_CRITERIA) == {
        "drawdown",
        "single_day",
        "cap_violation",
        "reconcile_drift",
        "ks_test",
    }


# ---------------------------------------------------------------------------- rebal_event


def test_insert_and_fetch_rebal_events_ordered(state: Path) -> None:
    paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    paper_clock.insert_rebal_event(
        n_orders_submitted=3,
        nav_at_rebal=Decimal("100100"),
        target_weights_json='[{"symbol":"SPY","weight":0.5}]',
        notes="rebal-1",
        db_path=state,
    )
    paper_clock.insert_rebal_event(
        n_orders_submitted=4,
        nav_at_rebal=Decimal("100200"),
        target_weights_json="[]",
        notes="rebal-2",
        db_path=state,
    )
    rows = paper_clock.fetch_rebal_events(db_path=state)
    assert len(rows) == 2
    assert rows[0].notes == "rebal-1"
    assert rows[1].notes == "rebal-2"
    assert rows[0].nav_at_rebal == Decimal("100100")


# ---------------------------------------------------------------------------- verdict


def test_set_verdict_rejects_bad_value(state: Path) -> None:
    paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    with pytest.raises(ValueError):
        paper_clock.set_verdict(verdict="MAYBE", db_path=state)


def test_set_verdict_persists_commit(state: Path) -> None:
    paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    paper_clock.set_verdict(verdict="COMMIT", notes="all gates passed", db_path=state)
    row = paper_clock.fetch_paper_clock(db_path=state)
    assert row is not None
    assert row.verdict == "COMMIT"
    assert row.verdict_at_utc is not None
    assert row.notes == "all gates passed"


def test_set_verdict_persists_kill(state: Path) -> None:
    paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    paper_clock.set_verdict(verdict="KILL", db_path=state)
    row = paper_clock.fetch_paper_clock(db_path=state)
    assert row is not None
    assert row.verdict == "KILL"


# ---------------------------------------------------------------------------- days_elapsed


def test_days_elapsed_none_when_not_started(state: Path) -> None:
    assert paper_clock.days_elapsed(db_path=state) is None


def test_days_elapsed_arithmetic(state: Path) -> None:
    paper_clock.ensure_started(start_nav=Decimal("100000"), db_path=state)
    row = paper_clock.fetch_paper_clock(db_path=state)
    assert row is not None
    fake_now = row.started_at_utc + timedelta(days=14, hours=3)
    assert paper_clock.days_elapsed(now=fake_now, db_path=state) == 14


# ---------------------------------------------------------------------------- last business day


def test_is_last_business_day_friday_at_month_end() -> None:
    # 2026-05-29 is a Friday; May 30/31 are Sat/Sun → Friday IS last bday.
    assert paper_clock.is_last_business_day_of_month(date(2026, 5, 29)) is True


def test_is_not_last_business_day_midmonth() -> None:
    assert paper_clock.is_last_business_day_of_month(date(2026, 5, 15)) is False


def test_weekend_is_never_last_business_day() -> None:
    # 2026-05-30 (Sat) and 2026-05-31 (Sun) are not business days at all.
    assert paper_clock.is_last_business_day_of_month(date(2026, 5, 30)) is False
    assert paper_clock.is_last_business_day_of_month(date(2026, 5, 31)) is False


def test_last_calendar_day_weekday_is_last_bday() -> None:
    # 2026-04-30 is a Thursday and the last calendar day → True.
    assert paper_clock.is_last_business_day_of_month(date(2026, 4, 30)) is True
