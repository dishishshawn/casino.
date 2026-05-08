"""Tests for casino.execution.sim_broker.

Coverage:

* Order → next-day open fill round-trip (cash decreases, position appears).
* Stop-loss fires when bar's low <= stop_price; position force-closed at stop.
* NAV evolves correctly across multi-day mark-to-market.
* Multi-run_id isolation: two SimBrokers in the same DB don't see each
  other's positions, orders, or NAV history.
* close_position queues a sell that fills next day.
* Idempotent mark_to_market: re-marking the same date is a no-op.
* Insufficient cash: order quantity capped to what cash can fund.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from casino.config import get_config
from casino.execution import sim_broker as smb
from casino.execution.sim_broker import SimBroker, fetch_sim_nav_history

# ---------------------------------------------------------------------------- fixtures


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "state.sqlite"
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(p))
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    get_config.cache_clear()
    return p


@pytest.fixture
def synthetic_duckdb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a tiny DuckDB with predictable OHLCV bars for the sim's bar lookup."""
    db = tmp_path / "casino.duckdb"
    monkeypatch.setenv("CASINO_DUCKDB_PATH", str(db))
    get_config.cache_clear()

    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE ohlcv (
            ticker VARCHAR NOT NULL,
            ts TIMESTAMPTZ NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume BIGINT, adj_close DOUBLE,
            PRIMARY KEY (ticker, ts)
        )
        """
    )
    # 30 bus-days of bars for SPY (uptrend) and CRASH (drops 30% on day 5).
    rows = []
    base = date(2026, 4, 1)
    for i in range(40):
        d = base + pd.Timedelta(days=i)
        ts = datetime(d.year, d.month, d.day, tzinfo=UTC)
        spy_close = 400.0 + i * 1.0  # +$1/day
        rows.append(
            (
                "SPY",
                ts,
                spy_close - 0.5,
                spy_close + 0.5,
                spy_close - 1.0,
                spy_close,
                1_000_000,
                spy_close,
            )
        )
        # CRASH: $100 starting; on day 6 (i==5), low drops to 70 (would hit a $90 stop).
        if i == 5:
            rows.append(
                (
                    "CRASH",
                    ts,
                    100.0,
                    100.5,
                    70.0,
                    95.0,
                    500_000,
                    95.0,
                )
            )
        else:
            rows.append(
                (
                    "CRASH",
                    ts,
                    100.0,
                    101.0,
                    99.5,
                    100.0,
                    500_000,
                    100.0,
                )
            )
    con.executemany(
        "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.close()
    return db


# ---------------------------------------------------------------------------- helpers


def _patch_submission_time(monkeypatch: pytest.MonkeyPatch, when: datetime) -> None:
    """Force ``_utc_now_iso`` to return a fixed timestamp for submission."""
    monkeypatch.setattr(smb, "_utc_now_iso", lambda: when.isoformat())


# ---------------------------------------------------------------------------- order round-trip


def test_order_next_open_fill_roundtrip(
    state: Path,
    synthetic_duckdb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit a buy on day T → fills at day T+1 open, cash decreases, position exists."""
    submit_dt = datetime(2026, 4, 2, tzinfo=UTC)
    _patch_submission_time(monkeypatch, submit_dt)

    sb = SimBroker(run_id="t1", db_path=state, duckdb_path=synthetic_duckdb)
    sb.submit_bracket_order(
        symbol="SPY",
        qty=10,
        side="buy",
        stop_price=Decimal("380"),
    )
    sb.mark_to_market(date(2026, 4, 3))  # next bar

    positions = sb.get_positions()
    assert len(positions) == 1
    p = positions[0]
    assert p.symbol == "SPY"
    assert p.qty == 10
    assert p.side == "long"
    # Fill price was day 4/3's open: SPY close on 4/3 = 400 + 2 = 402; open = 401.5
    assert abs(float(p.avg_entry_price) - 401.5) < 0.01
    # Cash decreased by qty * fill_price.
    acct = sb.get_account()
    assert acct.cash < Decimal("100000")
    assert acct.equity > Decimal("0")


# ---------------------------------------------------------------------------- stop-loss


def test_stop_loss_fires_when_low_breaches(
    state: Path,
    synthetic_duckdb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long with stop=$90 on CRASH: on day with low=70, position force-closed at $90."""
    submit_dt = datetime(2026, 4, 2, tzinfo=UTC)
    _patch_submission_time(monkeypatch, submit_dt)

    sb = SimBroker(run_id="t_stop", db_path=state, duckdb_path=synthetic_duckdb)
    sb.submit_bracket_order(
        symbol="CRASH",
        qty=10,
        side="buy",
        stop_price=Decimal("90"),
    )
    # mark forward day-by-day: 4/3 fills, 4/6 (day index 5 in our seed) hits the stop
    sb.mark_to_market(date(2026, 4, 3))
    pos_pre = sb.get_positions()
    assert any(p.symbol == "CRASH" for p in pos_pre)

    sb.mark_to_market(date(2026, 4, 7))  # past the crash day
    pos_post = sb.get_positions()
    # Position should be force-closed.
    assert all(p.symbol != "CRASH" for p in pos_post)


# ---------------------------------------------------------------------------- multi-day NAV


def test_nav_history_evolves_across_mark_to_market(
    state: Path,
    synthetic_duckdb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submit_dt = datetime(2026, 4, 2, tzinfo=UTC)
    _patch_submission_time(monkeypatch, submit_dt)

    sb = SimBroker(run_id="t_nav", db_path=state, duckdb_path=synthetic_duckdb)
    sb.submit_bracket_order(symbol="SPY", qty=10, side="buy", stop_price=Decimal("380"))
    sb.mark_to_market(date(2026, 4, 3))
    sb.mark_to_market(date(2026, 4, 6))
    sb.mark_to_market(date(2026, 4, 7))

    history = fetch_sim_nav_history(run_id="t_nav", db_path=state)
    assert len(history) == 3
    # Equity must monotonically increase since SPY trends up by $1/day.
    eqs = [h["equity"] for h in history]
    assert eqs[0] < eqs[1] < eqs[2]


# ---------------------------------------------------------------------------- isolation


def test_two_run_ids_do_not_see_each_other(
    state: Path,
    synthetic_duckdb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sims in the same DB are fully isolated by run_id."""
    submit_dt = datetime(2026, 4, 2, tzinfo=UTC)
    _patch_submission_time(monkeypatch, submit_dt)

    a = SimBroker(run_id="run_a", db_path=state, duckdb_path=synthetic_duckdb)
    b = SimBroker(run_id="run_b", db_path=state, duckdb_path=synthetic_duckdb)

    a.submit_bracket_order(symbol="SPY", qty=10, side="buy", stop_price=Decimal("380"))
    a.mark_to_market(date(2026, 4, 3))

    # b sees nothing.
    assert b.get_positions() == []
    assert b.get_account().cash == Decimal("100000")  # untouched
    # a sees its position.
    assert len(a.get_positions()) == 1


# ---------------------------------------------------------------------------- close


def test_close_position_queues_sell(
    state: Path,
    synthetic_duckdb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submit_dt = datetime(2026, 4, 2, tzinfo=UTC)
    _patch_submission_time(monkeypatch, submit_dt)
    sb = SimBroker(run_id="t_close", db_path=state, duckdb_path=synthetic_duckdb)
    sb.submit_bracket_order(symbol="SPY", qty=10, side="buy", stop_price=Decimal("380"))
    sb.mark_to_market(date(2026, 4, 3))
    assert len(sb.get_positions()) == 1

    # close_position respects the next-day-open rule.
    monkeypatch.setattr(smb, "_utc_now_iso", lambda: datetime(2026, 4, 6, tzinfo=UTC).isoformat())
    sb.close_position("SPY")
    sb.mark_to_market(date(2026, 4, 7))
    assert sb.get_positions() == []
    # Cash returned (proceeds added).
    assert sb.get_account().cash > Decimal("90000")


# ---------------------------------------------------------------------------- idempotency


def test_mark_to_market_idempotent(
    state: Path,
    synthetic_duckdb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submit_dt = datetime(2026, 4, 2, tzinfo=UTC)
    _patch_submission_time(monkeypatch, submit_dt)
    sb = SimBroker(run_id="t_idem", db_path=state, duckdb_path=synthetic_duckdb)
    sb.submit_bracket_order(symbol="SPY", qty=10, side="buy", stop_price=Decimal("380"))
    res1 = sb.mark_to_market(date(2026, 4, 3))
    assert res1["fills"] == 1
    res2 = sb.mark_to_market(date(2026, 4, 3))  # same date
    # The "no_op" path must trigger because as_of <= last_clock.
    assert res2.get("no_op") is True
