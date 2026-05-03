"""Tests for casino.data.store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from casino.config import get_config
from casino.data import store


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    get_config.cache_clear()
    p = tmp_path / "casino.duckdb"
    store.create_schema(db_path=p)
    return p


def test_schema_idempotent(db_path: Path) -> None:
    # second call must not raise
    store.create_schema(db_path=db_path)


def test_ohlcv_upsert_round_trip(db_path: Path) -> None:
    ts = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
    n = store.upsert_ohlcv(
        [
            {
                "ticker": "AAPL",
                "ts": ts,
                "open": 180.0,
                "high": 181.0,
                "low": 179.0,
                "close": 180.5,
                "volume": 1_000_000,
                "adj_close": 180.5,
            }
        ],
        db_path=db_path,
    )
    assert n == 1
    with store.get_duckdb_conn(db_path) as conn:
        row = conn.execute("SELECT ticker, close, volume FROM ohlcv WHERE ticker='AAPL'").fetchone()
    assert row is not None
    assert row[0] == "AAPL"
    assert row[1] == pytest.approx(180.5)
    assert row[2] == 1_000_000


def test_ohlcv_upsert_replaces_on_duplicate(db_path: Path) -> None:
    ts = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
    base = {
        "ticker": "AAPL",
        "ts": ts,
        "open": 100.0,
        "high": 100.0,
        "low": 100.0,
        "close": 100.0,
        "volume": 1,
        "adj_close": 100.0,
    }
    store.upsert_ohlcv([base], db_path=db_path)
    updated = {**base, "close": 200.0}
    store.upsert_ohlcv([updated], db_path=db_path)
    with store.get_duckdb_conn(db_path) as conn:
        rows = conn.execute("SELECT close FROM ohlcv WHERE ticker='AAPL'").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(200.0)


def test_pit_constituents_membership_window(db_path: Path) -> None:
    added = datetime(2020, 1, 1, tzinfo=UTC)
    removed = datetime(2022, 6, 30, tzinfo=UTC)
    store.upsert_index_constituent(
        index_name="SP500",
        ticker="ABC",
        added_on=added,
        removed_on=removed,
        db_path=db_path,
    )
    store.upsert_index_constituent(
        index_name="SP500",
        ticker="XYZ",
        added_on=added,
        removed_on=None,
        db_path=db_path,
    )

    # Inside membership window: both present
    inside = store.point_in_time_constituents(
        "SP500", datetime(2021, 6, 1, tzinfo=UTC), db_path=db_path
    )
    assert set(inside) == {"ABC", "XYZ"}

    # After ABC's removal: only XYZ
    after = store.point_in_time_constituents(
        "SP500", datetime(2023, 1, 1, tzinfo=UTC), db_path=db_path
    )
    assert after == ["XYZ"]

    # Before any membership: empty
    before = store.point_in_time_constituents(
        "SP500", datetime(2010, 1, 1, tzinfo=UTC), db_path=db_path
    )
    assert before == []


def test_pit_constituents_unknown_index(db_path: Path) -> None:
    out = store.point_in_time_constituents("FAKE_INDEX", datetime.now(tz=UTC), db_path=db_path)
    assert out == []


def test_archive_to_parquet_round_trip(db_path: Path, tmp_path: Path) -> None:
    ts = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
    store.upsert_ohlcv(
        [
            {
                "ticker": "AAPL",
                "ts": ts,
                "open": 180.0,
                "high": 181.0,
                "low": 179.0,
                "close": 180.5,
                "volume": 1_000_000,
                "adj_close": 180.5,
            },
            {
                "ticker": "AAPL",
                "ts": ts + timedelta(days=1),
                "open": 181.0,
                "high": 182.0,
                "low": 180.0,
                "close": 181.5,
                "volume": 900_000,
                "adj_close": 181.5,
            },
        ],
        db_path=db_path,
    )
    out_dir = tmp_path / "parquet_out"
    written = store.archive_to_parquet("ohlcv", out_dir, db_path=db_path)
    assert written.exists()
    # round-trip via duckdb to verify data
    with store.get_duckdb_conn(db_path) as conn:
        n = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{written.as_posix()}')").fetchone()
    assert n is not None
    assert n[0] == 2


def test_news_upsert(db_path: Path) -> None:
    pub = datetime(2024, 5, 1, tzinfo=UTC)
    n = store.upsert_news(
        [
            {
                "id": "n1",
                "ticker": "AAPL",
                "headline": "Apple beats EPS",
                "published_at": pub,
                "source": "Tiingo",
                "url": "https://example.com/n1",
            }
        ],
        db_path=db_path,
    )
    assert n == 1
    with store.get_duckdb_conn(db_path) as conn:
        row = conn.execute("SELECT headline FROM news WHERE id='n1'").fetchone()
    assert row is not None
    assert row[0] == "Apple beats EPS"


def test_earnings_upsert_and_query(db_path: Path) -> None:
    rd = datetime(2024, 5, 1, tzinfo=UTC)
    store.upsert_earnings(
        [
            {
                "ticker": "AAPL",
                "report_date": rd,
                "period_end": rd,
                "actual_eps": 2.18,
                "consensus_eps": 2.10,
                "revenue": 100.0,
                "source": "tiingo",
            }
        ],
        db_path=db_path,
    )
    with store.get_duckdb_conn(db_path) as conn:
        row = conn.execute(
            "SELECT actual_eps, consensus_eps FROM earnings WHERE ticker='AAPL'"
        ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(2.18)
    assert row[1] == pytest.approx(2.10)
