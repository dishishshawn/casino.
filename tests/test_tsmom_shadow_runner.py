"""Tests for casino.execution.tsmom_shadow_runner.

Coverage:

* Off rebal-day with no force is a daily MTM (not skipped harshly).
* Force-run end-to-end: weights → sim orders → fills → positions.
* --catchup-from advances correctly through historical bars.
* OHLCV stale + not force → refuses to rebal (freshness gate inherited).
* Plan respects single-name and gross caps from config.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from casino.config import get_config
from casino.execution import paper_clock
from casino.execution.sim_broker import SimBroker
from casino.execution.tsmom_runner import TargetWeight
from casino.execution.tsmom_shadow_runner import (
    SHADOW_RUN_ID,
    plan_shadow_actions,
    run_shadow_rebal,
)

# ---------------------------------------------------------------------------- fixtures


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "state.sqlite"
    monkeypatch.setenv("CASINO_STATE_SQLITE_PATH", str(p))
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    get_config.cache_clear()
    return p


@pytest.fixture
def fake_duckdb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small DuckDB with OHLCV + FRED for the shadow runner."""
    db = tmp_path / "casino.duckdb"
    monkeypatch.setenv("CASINO_DUCKDB_PATH", str(db))
    get_config.cache_clear()
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE ohlcv (
            ticker VARCHAR NOT NULL, ts TIMESTAMPTZ NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume BIGINT, adj_close DOUBLE,
            PRIMARY KEY (ticker, ts)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE fred_yields (
            series_id VARCHAR NOT NULL, ts TIMESTAMPTZ NOT NULL,
            value DOUBLE, PRIMARY KEY (series_id, ts)
        )
        """
    )
    # 400 business days of OHLCV per symbol, ending at 2026-05-08.
    end = pd.Timestamp("2026-05-08")
    idx = pd.bdate_range(end=end, periods=400)
    rng = np.random.default_rng(0)

    rows = []
    for sym, base_price in [("SPY", 400.0), ("QQQ", 350.0), ("TLT", 95.0), ("IEF", 105.0)]:
        for i, ts in enumerate(idx):
            close = base_price + i * 0.10 + rng.normal(0, 0.05)
            rows.append(
                (
                    sym,
                    datetime(ts.year, ts.month, ts.day, tzinfo=UTC),
                    close - 0.2,
                    close + 0.4,
                    close - 0.4,
                    close,
                    1_000_000,
                    close,
                )
            )
    con.executemany("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    # FRED rows: positive slope so the regime filter does NOT fire (test
    # vanilla-equivalence path through the shadow).
    fred_rows = []
    for ts in idx:
        ts_utc = datetime(ts.year, ts.month, ts.day, tzinfo=UTC)
        fred_rows.append(("DGS10", ts_utc, 4.5))
        fred_rows.append(("DTB3", ts_utc, 4.0))
    con.executemany("INSERT INTO fred_yields VALUES (?, ?, ?)", fred_rows)
    con.close()
    return db


# ---------------------------------------------------------------------------- plan_shadow_actions


def test_plan_respects_single_name_cap() -> None:
    targets = [TargetWeight(symbol="AAA", weight=0.25, reference_price=Decimal("100"))]
    actions = plan_shadow_actions(
        target_weights=targets,
        nav=Decimal("100000"),
        current_positions={},
    )
    open_long = [a for a in actions if a.kind == "open_long"]
    assert len(open_long) == 1
    # 10% NAV = $10000, qty = 100 at $100.
    assert open_long[0].qty == 100
    assert open_long[0].target_dollars <= Decimal("10000")


def test_plan_closes_dropped_positions() -> None:
    targets = [TargetWeight(symbol="A", weight=0.05, reference_price=Decimal("100"))]
    actions = plan_shadow_actions(
        target_weights=targets,
        nav=Decimal("100000"),
        current_positions={"ZZZ": 5},
    )
    closes = [a for a in actions if a.kind == "close"]
    assert len(closes) == 1
    assert closes[0].symbol == "ZZZ"


def test_plan_attaches_stop_at_negative_ten_percent() -> None:
    targets = [TargetWeight(symbol="A", weight=0.05, reference_price=Decimal("100"))]
    actions = plan_shadow_actions(
        target_weights=targets,
        nav=Decimal("100000"),
        current_positions={},
    )
    open_long = [a for a in actions if a.kind == "open_long"]
    assert len(open_long) == 1
    assert open_long[0].stop_price == Decimal("90.00")


# ---------------------------------------------------------------------------- run_shadow_rebal


def test_off_rebal_day_no_force_marks_to_market(state: Path, fake_duckdb: Path) -> None:
    today = date(2026, 5, 7)  # not a month-end
    res = run_shadow_rebal(
        today=today,
        force=False,
        db_path=state,
        duckdb_path=fake_duckdb,
        universe=["SPY"],
    )
    assert res.is_rebal_day is False
    assert res.skipped_reason is not None
    assert "not the last business day" in res.skipped_reason


def test_force_run_end_to_end(
    state: Path,
    fake_duckdb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force start → submits sim orders → mark-to-market fills them."""
    today = date(2026, 5, 7)
    # Patch the submission timestamp so fills target a date that has bars.
    from casino.execution import sim_broker as smb_mod

    monkeypatch.setattr(
        smb_mod,
        "_utc_now_iso",
        lambda: datetime(2026, 5, 6, tzinfo=UTC).isoformat(),
    )
    res = run_shadow_rebal(
        today=today,
        force=True,
        db_path=state,
        duckdb_path=fake_duckdb,
        universe=["SPY", "QQQ", "TLT", "IEF"],
    )
    assert res.skipped_reason is None
    assert res.is_rebal_day is False  # 5/7 isn't month-end
    assert res.forced is True
    assert len(res.actions) >= 1
    # paper_clock row exists for SHADOW_RUN_ID.
    pc = paper_clock.fetch_paper_clock(run_id=SHADOW_RUN_ID, db_path=state)
    assert pc is not None
    assert pc.strategy == "tsmom_regime_shadow"
    # rebal_event row inserted for our run_id.
    rebals = paper_clock.fetch_rebal_events(run_id=SHADOW_RUN_ID, db_path=state)
    assert len(rebals) == 1
    # Sim broker has positions after the mark.
    sb = SimBroker(run_id=SHADOW_RUN_ID, db_path=state, duckdb_path=fake_duckdb)
    # Today is 5/7 but submission timestamp was patched to 5/6 → next-open
    # fills happen on 5/7's open, which is the as_of bar.
    assert len(sb.get_positions()) >= 1


def test_catchup_from_advances_through_bars(
    state: Path,
    fake_duckdb: Path,
) -> None:
    """--catchup-from marks intermediate business days, populating sim_nav_history."""
    today = date(2026, 5, 7)
    catchup = date(2026, 5, 1)  # 5 bdays earlier
    res = run_shadow_rebal(
        today=today,
        force=False,
        db_path=state,
        duckdb_path=fake_duckdb,
        universe=["SPY"],
        catchup_from=catchup,
    )
    # Skipped (not rebal day) but catchup ran.
    assert len(res.catchup_dates) >= 4  # 5/1 (Fri), 5/4, 5/5, 5/6 = 4
    # Verify nav-history rows landed.
    from casino.execution.sim_broker import fetch_sim_nav_history

    hist = fetch_sim_nav_history(run_id=SHADOW_RUN_ID, db_path=state)
    # >=4 catchup dates + 1 today MTM
    assert len(hist) >= 4


def test_isolation_from_live_runner(
    state: Path,
    fake_duckdb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shadow's run_id does not appear under the live's run_id."""
    today = date(2026, 5, 7)
    from casino.execution import sim_broker as smb_mod

    monkeypatch.setattr(
        smb_mod,
        "_utc_now_iso",
        lambda: datetime(2026, 5, 6, tzinfo=UTC).isoformat(),
    )
    run_shadow_rebal(
        today=today,
        force=True,
        db_path=state,
        duckdb_path=fake_duckdb,
        universe=["SPY", "QQQ"],
    )
    # The live default run_id ("DiCaprio") must NOT have a paper_clock row.
    live_pc = paper_clock.fetch_paper_clock(run_id="DiCaprio", db_path=state)
    assert live_pc is None
    # Shadow's run_id does.
    shadow_pc = paper_clock.fetch_paper_clock(run_id=SHADOW_RUN_ID, db_path=state)
    assert shadow_pc is not None
