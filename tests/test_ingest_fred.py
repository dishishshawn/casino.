"""Tests for casino.data.ingest_fred. HTTP fully mocked — no live FRED hits."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from casino.config import get_config
from casino.data import ingest_fred, store


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    get_config.cache_clear()
    monkeypatch.setenv("CASINO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CASINO_DUCKDB_PATH", str(tmp_path / "casino.duckdb"))
    get_config.cache_clear()
    p = tmp_path / "casino.duckdb"
    store.create_schema(db_path=p)
    return p


# A minimal FRED CSV body. Real bodies use DOS line endings; httpx.Response
# normalizes either way.
_SAMPLE_CSV = (
    "DATE,DGS10\n"
    "2024-01-02,3.95\n"
    "2024-01-03,3.92\n"
    "2024-01-04,.\n"  # missing observation row — must be skipped
    "2024-01-05,3.99\n"
)


def _fake_handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=_SAMPLE_CSV)


def test_parse_csv_skips_missing_rows() -> None:
    rows = ingest_fred._parse_csv("DGS10", _SAMPLE_CSV)
    # 3 well-formed rows; the "." row is skipped.
    assert len(rows) == 3
    assert all(r["series_id"] == "DGS10" for r in rows)
    assert {r["value"] for r in rows} == {3.95, 3.92, 3.99}
    for r in rows:
        ts = r["ts"]
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None


def test_fetch_series_uses_provided_client() -> None:
    transport = httpx.MockTransport(_fake_handler)
    client = httpx.Client(transport=transport)
    rows = ingest_fred.fetch_series("DGS10", start="2024-01-01", end="2024-01-31", client=client)
    client.close()
    assert len(rows) == 3
    assert rows[0]["series_id"] == "DGS10"


def test_fetch_series_swallows_http_errors() -> None:
    def boom(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream unavailable")

    client = httpx.Client(transport=httpx.MockTransport(boom))
    rows = ingest_fred.fetch_series("DGS10", client=client)
    client.close()
    assert rows == []


def test_ingest_round_trip(env: Path) -> None:
    """3 sample CSV rows round-trip into the DB and out via load_fred_panel."""
    transport = httpx.MockTransport(_fake_handler)
    client = httpx.Client(transport=transport)
    counts = ingest_fred.ingest_series(
        ["DGS10"],
        start="2024-01-01",
        end="2024-01-31",
        db_path=env,
        rate_limit_sec=0.0,
        client=client,
    )
    client.close()
    assert counts == {"DGS10": 3}

    panel = store.load_fred_panel(
        series_ids=["DGS10"],
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 31, tzinfo=UTC),
        db_path=env,
    )
    assert not panel.empty
    assert "DGS10" in panel.columns
    # Exactly the 3 written rows; the missing-data row was filtered.
    assert len(panel) == 3
    # Values match upstream (percent units).
    vals = panel["DGS10"].tolist()
    assert sorted(vals) == [3.92, 3.95, 3.99]


def test_ingest_idempotent(env: Path) -> None:
    """Re-running upserts cleanly via PK on (series_id, ts)."""
    transport = httpx.MockTransport(_fake_handler)
    client = httpx.Client(transport=transport)
    ingest_fred.ingest_series(
        ["DGS10"], start="2024-01-01", db_path=env, rate_limit_sec=0.0, client=client
    )
    ingest_fred.ingest_series(
        ["DGS10"], start="2024-01-01", db_path=env, rate_limit_sec=0.0, client=client
    )
    client.close()
    with store.get_duckdb_conn(env, read_only=True) as conn:
        n = conn.execute("SELECT count(*) FROM fred_yields").fetchone()
    assert n is not None and n[0] == 3  # unique (series_id, ts) keys


def test_parse_csv_handles_empty_body() -> None:
    assert ingest_fred._parse_csv("DGS10", "") == []
    # Header but no data rows.
    assert ingest_fred._parse_csv("DGS10", "DATE,DGS10\n") == []
