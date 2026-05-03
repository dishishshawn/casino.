"""Tests for casino.signals.pead (SUE math)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from casino.config import get_config
from casino.data import store
from casino.signals import pead


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    get_config.cache_clear()
    p = tmp_path / "casino.duckdb"
    store.create_schema(db_path=p)
    return p


def _seed_history(
    db_path: Path, ticker: str, surprises: list[float], *, base: datetime | None = None
) -> None:
    """Insert N quarters of synthetic earnings prior to `base` (default now)."""
    base = base or datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i, s in enumerate(surprises):
        rd = base - timedelta(days=90 * (i + 1))
        rows.append(
            {
                "ticker": ticker,
                "report_date": rd,
                "period_end": rd,
                "actual_eps": 2.0 + s,
                "consensus_eps": 2.0,
                "revenue": 0.0,
                "source": "test",
            }
        )
    store.upsert_earnings(rows, db_path=db_path)


def test_compute_sue_positive_beat(db_path: Path) -> None:
    # Historical surprises with std ~0.05; current beat = 0.10 → SUE ≈ 2.0
    _seed_history(db_path, "AAPL", [-0.05, 0.05, -0.03, 0.04, 0.0, -0.02, 0.03, -0.01])
    sue = pead.compute_sue(
        "AAPL",
        actual_eps=Decimal("2.10"),
        consensus_eps=Decimal("2.00"),
        as_of_date=datetime(2024, 1, 1, tzinfo=UTC),
        db_path=db_path,
    )
    assert sue is not None
    assert sue > 0
    assert 1.5 < sue < 5.0  # rough range


def test_compute_sue_negative_miss(db_path: Path) -> None:
    _seed_history(db_path, "AAPL", [0.05, -0.05, 0.03, -0.04, 0.0, 0.02, -0.03, 0.01])
    sue = pead.compute_sue(
        "AAPL",
        actual_eps=Decimal("1.85"),
        consensus_eps=Decimal("2.00"),
        as_of_date=datetime(2024, 1, 1, tzinfo=UTC),
        db_path=db_path,
    )
    assert sue is not None
    assert sue < 0


def test_compute_sue_missing_consensus_returns_none(db_path: Path) -> None:
    _seed_history(db_path, "AAPL", [0.05, -0.05, 0.03])
    sue = pead.compute_sue(
        "AAPL",
        actual_eps=Decimal("2.10"),
        consensus_eps=None,
        as_of_date=datetime(2024, 1, 1, tzinfo=UTC),
        db_path=db_path,
    )
    assert sue is None


def test_compute_sue_insufficient_history_uses_industry_fallback(db_path: Path) -> None:
    # Only 1 historical quarter — std is undefined; fall back to industry default
    _seed_history(db_path, "NEW", [0.05])
    sue = pead.compute_sue(
        "NEW",
        actual_eps=Decimal("2.10"),
        consensus_eps=Decimal("2.00"),
        as_of_date=datetime(2024, 1, 1, tzinfo=UTC),
        db_path=db_path,
    )
    assert sue is not None
    assert sue == pytest.approx(0.10 / pead.DEFAULT_INDUSTRY_STD)


def test_get_earnings_surprises_respects_as_of_date(db_path: Path) -> None:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    _seed_history(db_path, "AAPL", [0.01, 0.02, 0.03, 0.04], base=base)
    df = pead.get_earnings_surprises(
        "AAPL",
        lookback_quarters=10,
        as_of_date=base,
        db_path=db_path,
    )
    assert len(df) == 4

    # narrower as_of cuts off earliest quarters
    earlier = base - timedelta(days=180)
    df2 = pead.get_earnings_surprises(
        "AAPL",
        lookback_quarters=10,
        as_of_date=earlier,
        db_path=db_path,
    )
    assert len(df2) < 4


def test_compute_sue_top_decile_per_literature(db_path: Path) -> None:
    """SUE > 3 should land a quarter in the academic top decile (PEAD lit)."""
    _seed_history(db_path, "AAPL", [0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.0])
    # std ~0.013, surprise = 0.05 → SUE ~3.7
    sue = pead.compute_sue(
        "AAPL",
        actual_eps=Decimal("2.05"),
        consensus_eps=Decimal("2.00"),
        as_of_date=datetime(2024, 1, 1, tzinfo=UTC),
        db_path=db_path,
    )
    assert sue is not None and sue > 3.0
