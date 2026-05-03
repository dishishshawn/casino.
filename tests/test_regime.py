"""Tests for casino.signals.regime."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from casino.config import get_config
from casino.data import store
from casino.signals import regime


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    get_config.cache_clear()
    p = tmp_path / "casino.duckdb"
    store.create_schema(db_path=p)
    return p


def _seed_spy(
    db_path: Path,
    *,
    days: int,
    last_close: float,
    base_close: float,
) -> datetime:
    """Insert `days` daily SPY bars, with linear interpolation between
    `base_close` and `last_close`. Returns the date of the last bar.
    """
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(days):
        ts = base + timedelta(days=i)
        if days == 1:
            close = last_close
        else:
            close = base_close + (last_close - base_close) * (i / (days - 1))
        rows.append(
            {
                "ticker": "SPY",
                "ts": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000_000,
                "adj_close": close,
            }
        )
    store.upsert_ohlcv(rows, db_path=db_path)
    return base + timedelta(days=days - 1)


def test_regime_risk_on_when_above_ma(db_path: Path) -> None:
    last = _seed_spy(db_path, days=200, last_close=500.0, base_close=400.0)
    state = regime.evaluate_regime(
        as_of=last,
        window=200,
        db_path=db_path,
    )
    assert state.risk_on is True
    assert state.benchmark_ticker == "SPY"
    assert state.moving_average is not None
    assert state.benchmark_close is not None


def test_regime_risk_off_when_below_ma(db_path: Path) -> None:
    last = _seed_spy(db_path, days=200, last_close=300.0, base_close=400.0)
    state = regime.evaluate_regime(
        as_of=last,
        window=200,
        db_path=db_path,
    )
    assert state.risk_on is False
    assert state.benchmark_close is not None
    assert state.moving_average is not None
    assert state.benchmark_close < state.moving_average


def test_regime_fails_closed_on_insufficient_history(db_path: Path) -> None:
    last = _seed_spy(db_path, days=10, last_close=500.0, base_close=400.0)
    state = regime.evaluate_regime(
        as_of=last,
        window=200,
        db_path=db_path,
    )
    assert state.risk_on is False
    assert "insufficient" in state.reason


def test_is_risk_on_convenience_wrapper(db_path: Path) -> None:
    last = _seed_spy(db_path, days=200, last_close=500.0, base_close=400.0)
    assert regime.is_risk_on(as_of=last, db_path=db_path) is True
