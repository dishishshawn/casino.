"""Tests for casino.data.ingest_yfinance. yfinance fully mocked — no network."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from casino.config import get_config
from casino.data import store
from casino.data.ingest_yfinance import (
    _normalize_ohlcv_row,
    _normalize_row,
    fetch_earnings,
    fetch_ohlcv,
    ingest_ohlcv,
    ingest_tickers,
)


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


# -------- minimal pandas-like fakes (we don't need a real DataFrame) --------
class _FakeRow(dict[str, Any]):
    """Dict-like row that also supports attribute fallback used by the normalizer."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class _FakeTimestamp:
    """Mimics pandas.Timestamp.to_pydatetime()."""

    def __init__(self, dt: datetime) -> None:
        self._dt = dt

    def to_pydatetime(self) -> datetime:
        return self._dt


class _FakeDF:
    def __init__(self, items: list[tuple[Any, _FakeRow]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def iterrows(self):  # noqa: ANN201 — match pandas signature loosely
        return iter(self._items)


class _FakeTicker:
    def __init__(
        self,
        df: _FakeDF | None = None,
        *,
        history_df: _FakeDF | None = None,
    ) -> None:
        self._df = df
        self._history_df = history_df

    @property
    def earnings_dates(self) -> _FakeDF | None:
        return self._df

    def history(self, **_kwargs: Any) -> _FakeDF | None:
        return self._history_df


# ----------------------------------------------------------- normalizer unit
def test_normalize_row_extracts_actual_and_estimate() -> None:
    ts = _FakeTimestamp(datetime(2024, 10, 31, 20, 30, tzinfo=UTC))
    row = _FakeRow({"Reported EPS": 1.64, "EPS Estimate": 1.60, "Surprise(%)": 2.5})
    out = _normalize_row("aapl", ts, row)
    assert out is not None
    assert out["ticker"] == "AAPL"
    assert out["actual_eps"] == 1.64
    assert out["consensus_eps"] == 1.60
    assert out["revenue"] is None
    assert out["source"] == "yfinance"
    assert isinstance(out["report_date"], datetime)
    assert out["report_date"].tzinfo is not None


def test_normalize_row_handles_naive_datetime_as_utc() -> None:
    naive = datetime(2024, 1, 31, 12, 0)  # no tzinfo
    row = _FakeRow({"Reported EPS": float("nan"), "EPS Estimate": 0.5})
    out = _normalize_row("MSFT", naive, row)
    assert out is not None
    assert out["actual_eps"] is None  # NaN → None
    assert out["consensus_eps"] == 0.5
    assert out["report_date"].tzinfo is not None


def test_normalize_row_returns_none_for_unparseable_timestamp() -> None:
    out = _normalize_row("X", "not-a-date", _FakeRow({}))
    assert out is None


# ----------------------------------------------------------- fetch behavior
def test_fetch_earnings_returns_rows_for_populated_df() -> None:
    df = _FakeDF(
        [
            (
                _FakeTimestamp(datetime(2024, 7, 31, 20, 30, tzinfo=UTC)),
                _FakeRow({"Reported EPS": 1.40, "EPS Estimate": 1.38}),
            ),
            (
                _FakeTimestamp(datetime(2024, 10, 31, 20, 30, tzinfo=UTC)),
                _FakeRow({"Reported EPS": float("nan"), "EPS Estimate": 1.60}),
            ),
        ]
    )

    def factory(_t: str) -> _FakeTicker:
        return _FakeTicker(df)

    rows = fetch_earnings("AAPL", factory=factory)
    assert len(rows) == 2
    assert rows[0]["actual_eps"] == 1.40
    assert rows[1]["actual_eps"] is None  # future / unreported
    assert rows[1]["consensus_eps"] == 1.60


def test_fetch_earnings_empty_df_returns_empty() -> None:
    def factory(_t: str) -> _FakeTicker:
        return _FakeTicker(_FakeDF([]))

    assert fetch_earnings("ZZZZ", factory=factory) == []


def test_fetch_earnings_none_df_returns_empty() -> None:
    def factory(_t: str) -> _FakeTicker:
        return _FakeTicker(None)

    assert fetch_earnings("ZZZZ", factory=factory) == []


def test_fetch_earnings_swallows_construction_errors() -> None:
    def factory(_t: str) -> _FakeTicker:
        raise RuntimeError("yfinance is unhappy")

    assert fetch_earnings("BAD", factory=factory) == []


def test_fetch_earnings_swallows_attribute_errors() -> None:
    class _Boom:
        @property
        def earnings_dates(self) -> Any:
            raise RuntimeError("404 from Yahoo")

    def factory(_t: str) -> _Boom:
        return _Boom()

    assert fetch_earnings("AAPL", factory=factory) == []


# ----------------------------------------------------------- end-to-end ingest
def test_ingest_tickers_round_trip(env: Path) -> None:
    """fetch → upsert_earnings → DuckDB; verify row counts and read-back."""
    df = _FakeDF(
        [
            (
                _FakeTimestamp(datetime(2024, 7, 31, 20, 30, tzinfo=UTC)),
                _FakeRow({"Reported EPS": 1.40, "EPS Estimate": 1.38}),
            ),
        ]
    )

    def factory(_t: str) -> _FakeTicker:
        return _FakeTicker(df)

    counts = ingest_tickers(
        ["AAPL", "MSFT"],
        factory=factory,
        db_path=env,
        rate_limit_sec=0.0,
    )
    assert counts == {"AAPL": 1, "MSFT": 1}

    with store.get_duckdb_conn(env) as conn:
        out = conn.execute(
            "SELECT ticker, actual_eps, consensus_eps, source FROM earnings ORDER BY ticker"
        ).fetchall()
    assert out == [
        ("AAPL", 1.40, 1.38, "yfinance"),
        ("MSFT", 1.40, 1.38, "yfinance"),
    ]


def test_ingest_tickers_skips_empty_results(env: Path) -> None:
    def factory(_t: str) -> _FakeTicker:
        return _FakeTicker(_FakeDF([]))

    counts = ingest_tickers(["XXXX"], factory=factory, db_path=env, rate_limit_sec=0.0)
    assert counts == {"XXXX": 0}
    with store.get_duckdb_conn(env) as conn:
        out = conn.execute("SELECT count(*) FROM earnings").fetchone()
    assert out is not None and out[0] == 0


def test_normalize_row_drops_nan_estimate() -> None:
    """NaN consensus must round-trip to None, not be stored as a sentinel."""
    ts = _FakeTimestamp(datetime(2024, 1, 1, tzinfo=UTC))
    row = _FakeRow({"Reported EPS": 0.5, "EPS Estimate": math.nan})
    out = _normalize_row("AAPL", ts, row)
    assert out is not None
    assert out["consensus_eps"] is None


# ============================================================================ ohlcv
def test_normalize_ohlcv_row_extracts_all_fields() -> None:
    ts = _FakeTimestamp(datetime(2024, 6, 3, tzinfo=UTC))
    row = _FakeRow(
        {
            "Open": 150.0,
            "High": 152.5,
            "Low": 149.25,
            "Close": 151.75,
            "Volume": 1_234_567,
            "Adj Close": 151.5,
        }
    )
    out = _normalize_ohlcv_row("aapl", ts, row)
    assert out is not None
    assert out["ticker"] == "AAPL"
    assert out["open"] == 150.0
    assert out["high"] == 152.5
    assert out["low"] == 149.25
    assert out["close"] == 151.75
    assert out["volume"] == 1_234_567
    assert out["adj_close"] == 151.5
    assert isinstance(out["ts"], datetime)
    assert out["ts"].tzinfo is not None


def test_normalize_ohlcv_row_skips_when_close_missing() -> None:
    """A bar without a close cannot anchor adj_close fallback or PEAD math; drop it."""
    ts = _FakeTimestamp(datetime(2024, 6, 3, tzinfo=UTC))
    row = _FakeRow({"Open": 150.0, "High": 152.0, "Low": 149.0})
    assert _normalize_ohlcv_row("AAPL", ts, row) is None


def test_normalize_ohlcv_row_falls_back_for_missing_adj_close() -> None:
    """If yfinance omits Adj Close, fall back to close so downstream joins still work."""
    ts = _FakeTimestamp(datetime(2024, 6, 3, tzinfo=UTC))
    row = _FakeRow({"Open": 150.0, "High": 152.0, "Low": 149.0, "Close": 151.0, "Volume": 100})
    out = _normalize_ohlcv_row("AAPL", ts, row)
    assert out is not None
    assert out["adj_close"] == 151.0


def test_fetch_ohlcv_returns_rows_for_populated_history() -> None:
    history = _FakeDF(
        [
            (
                _FakeTimestamp(datetime(2024, 6, 3, tzinfo=UTC)),
                _FakeRow(
                    {
                        "Open": 100.0, "High": 101.0, "Low": 99.0,
                        "Close": 100.5, "Volume": 1_000_000, "Adj Close": 100.5,
                    }
                ),
            ),
            (
                _FakeTimestamp(datetime(2024, 6, 4, tzinfo=UTC)),
                _FakeRow(
                    {
                        "Open": 100.5, "High": 102.0, "Low": 100.0,
                        "Close": 101.5, "Volume": 900_000, "Adj Close": 101.5,
                    }
                ),
            ),
        ]
    )

    def factory(_t: str) -> _FakeTicker:
        return _FakeTicker(history_df=history)

    rows = fetch_ohlcv("AAPL", factory=factory)
    assert len(rows) == 2
    assert rows[0]["close"] == 100.5
    assert rows[1]["close"] == 101.5


def test_fetch_ohlcv_swallows_history_errors() -> None:
    class _Boom:
        @property
        def earnings_dates(self) -> Any:
            return None

        def history(self, **_kwargs: Any) -> Any:
            raise RuntimeError("Yahoo says no")

    def factory(_t: str) -> _Boom:
        return _Boom()

    assert fetch_ohlcv("AAPL", factory=factory) == []


def test_ingest_ohlcv_round_trip(env: Path) -> None:
    """fetch_ohlcv → store.upsert_ohlcv → DuckDB; verify primary-key dedupe."""
    history = _FakeDF(
        [
            (
                _FakeTimestamp(datetime(2024, 6, 3, tzinfo=UTC)),
                _FakeRow(
                    {
                        "Open": 100.0, "High": 101.0, "Low": 99.0,
                        "Close": 100.5, "Volume": 1_000, "Adj Close": 100.5,
                    }
                ),
            ),
        ]
    )

    def factory(_t: str) -> _FakeTicker:
        return _FakeTicker(history_df=history)

    counts = ingest_ohlcv(
        ["AAPL", "MSFT"],
        factory=factory,
        db_path=env,
        rate_limit_sec=0.0,
    )
    assert counts == {"AAPL": 1, "MSFT": 1}

    with store.get_duckdb_conn(env) as conn:
        out = conn.execute(
            "SELECT ticker, close, volume FROM ohlcv ORDER BY ticker"
        ).fetchall()
    assert out == [("AAPL", 100.5, 1000), ("MSFT", 100.5, 1000)]

    # Re-ingesting the same bar must dedupe via INSERT OR REPLACE on (ticker, ts).
    counts2 = ingest_ohlcv(["AAPL"], factory=factory, db_path=env, rate_limit_sec=0.0)
    assert counts2 == {"AAPL": 1}
    with store.get_duckdb_conn(env) as conn:
        n = conn.execute("SELECT count(*) FROM ohlcv").fetchone()
    assert n is not None and n[0] == 2  # still 2 unique (ticker, ts) keys
