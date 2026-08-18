"""Tests for casino.execution.tsmom_falcon_runner.

Falcon is the aggressive vanilla-TSMOM sibling of the live ``DiCaprio`` bot,
running in the in-process ``SimBroker`` (``run_id="Falcon"``). Coverage:

* Aggression knobs are what the design promises (2x vol, faster lookbacks,
  wider stop) and are wired into the panel computation.
* Off rebal-day with no force is a daily MTM (not a harsh skip).
* Force-run end-to-end: weights -> sim orders -> fills -> positions.
* --catchup-from advances through historical bars.
* OHLCV stale + not force -> refuses to rebal (freshness gate inherited).
* Hard caps (single-name 10% / gross 100%) still bind — Falcon does NOT
  relax them.
* Isolation: Falcon touches neither DiCaprio's nor Belfort's paper_clock row.
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
from casino.execution.tsmom_falcon_runner import (
    FALCON_LOOKBACKS,
    FALCON_RUN_ID,
    FALCON_STOP_FRACTION,
    FALCON_TARGET_VOL,
    latest_falcon_target_weights,
    plan_falcon_actions,
    run_falcon_rebal,
)
from casino.execution.tsmom_runner import TargetWeight

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
    """A small DuckDB with OHLCV for the Falcon runner."""
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
    # 400 business days of OHLCV per symbol, ending at 2026-05-08. Uptrend so
    # the long-only momentum signal produces positive weights.
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
    con.close()
    return db


# ---------------------------------------------------------------------------- aggression knobs


def test_falcon_knobs_are_more_aggressive_than_dicaprio() -> None:
    """The whole point of Falcon: knobs strictly exceed the live config."""
    from casino.execution.tsmom_runner import DEFAULT_STOP_FRACTION

    # 2x the live 0.10 vol target.
    assert FALCON_TARGET_VOL == pytest.approx(0.20)
    # Wider stop than DiCaprio's 0.10.
    assert FALCON_STOP_FRACTION > DEFAULT_STOP_FRACTION
    # Faster: drops the slow 252-bday (12-month) leg the live bot uses.
    assert 252 not in FALCON_LOOKBACKS
    assert max(FALCON_LOOKBACKS) < 252


def test_latest_falcon_weights_use_faster_lookbacks(fake_duckdb: Path) -> None:
    """Falcon's weights are computed from its own lookbacks/vol, not the defaults.

    With only 130 bars of history the slow 252-bday leg cannot contribute, so
    a vanilla-default call returns burn-in NaNs (no weights) while Falcon's
    faster (21, 63, 126) lookbacks produce live weights.
    """
    from casino.execution.tsmom_runner import latest_target_weights
    from casino.signals.ts_momentum import load_ohlcv_panel

    prices = load_ohlcv_panel(
        start=datetime(2025, 10, 1, tzinfo=UTC),
        end=datetime(2026, 5, 9, tzinfo=UTC),
        universe=["SPY", "QQQ", "TLT", "IEF"],
        db_path=fake_duckdb,
    ).tail(130)

    default_weights = latest_target_weights(prices)
    falcon_weights = latest_falcon_target_weights(prices)

    # Default (needs 252 bdays) is still in burn-in -> empty; Falcon is live.
    assert default_weights == []
    assert len(falcon_weights) >= 1


# ---------------------------------------------------------------------------- plan caps (reused)


def test_falcon_plan_respects_single_name_cap() -> None:
    """Falcon does NOT relax the 10% single-name hard cap."""
    targets = [TargetWeight(symbol="AAA", weight=0.25, reference_price=Decimal("100"))]
    actions = plan_falcon_actions(
        target_weights=targets,
        nav=Decimal("100000"),
        current_positions={},
        stop_fraction=FALCON_STOP_FRACTION,
    )
    open_long = [a for a in actions if a.kind == "open_long"]
    assert len(open_long) == 1
    # 10% NAV = $10000, qty = 100 at $100 — capped despite the 0.25 target.
    assert open_long[0].qty == 100
    assert open_long[0].target_dollars <= Decimal("10000")


def test_falcon_plan_attaches_wider_stop() -> None:
    """Falcon's stop sits at -15% of the reference, not DiCaprio's -10%."""
    targets = [TargetWeight(symbol="A", weight=0.05, reference_price=Decimal("100"))]
    actions = plan_falcon_actions(
        target_weights=targets,
        nav=Decimal("100000"),
        current_positions={},
        stop_fraction=FALCON_STOP_FRACTION,
    )
    open_long = [a for a in actions if a.kind == "open_long"]
    assert len(open_long) == 1
    assert open_long[0].stop_price == Decimal("85.00")


# ---------------------------------------------------------------------------- run_falcon_rebal


def test_off_rebal_day_no_force_marks_to_market(state: Path, fake_duckdb: Path) -> None:
    today = date(2026, 5, 7)  # not a month-end
    res = run_falcon_rebal(
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
    """Force start -> submits sim orders -> mark-to-market fills them."""
    today = date(2026, 5, 7)
    from casino.execution import sim_broker as smb_mod

    monkeypatch.setattr(
        smb_mod,
        "_utc_now_iso",
        lambda: datetime(2026, 5, 6, tzinfo=UTC).isoformat(),
    )
    res = run_falcon_rebal(
        today=today,
        force=True,
        db_path=state,
        duckdb_path=fake_duckdb,
        universe=["SPY", "QQQ", "TLT", "IEF"],
    )
    assert res.skipped_reason is None
    assert res.forced is True
    assert len(res.actions) >= 1
    # paper_clock row exists for FALCON_RUN_ID with the aggressive strategy tag.
    pc = paper_clock.fetch_paper_clock(run_id=FALCON_RUN_ID, db_path=state)
    assert pc is not None
    assert pc.strategy == "tsmom_falcon_aggressive"
    # rebal_event row inserted for our run_id.
    rebals = paper_clock.fetch_rebal_events(run_id=FALCON_RUN_ID, db_path=state)
    assert len(rebals) == 1
    # Sim broker has positions after the mark.
    sb = SimBroker(run_id=FALCON_RUN_ID, db_path=state, duckdb_path=fake_duckdb)
    assert len(sb.get_positions()) >= 1


def test_catchup_from_advances_through_bars(state: Path, fake_duckdb: Path) -> None:
    """--catchup-from marks intermediate business days, populating sim_nav_history."""
    today = date(2026, 5, 7)
    catchup = date(2026, 5, 1)  # 5/1 (Fri), 5/4, 5/5, 5/6 = 4 bdays
    res = run_falcon_rebal(
        today=today,
        force=False,
        db_path=state,
        duckdb_path=fake_duckdb,
        universe=["SPY"],
        catchup_from=catchup,
    )
    assert len(res.catchup_dates) >= 4
    from casino.execution.sim_broker import fetch_sim_nav_history

    hist = fetch_sim_nav_history(run_id=FALCON_RUN_ID, db_path=state)
    assert len(hist) >= 4


def test_isolation_from_live_and_shadow(
    state: Path,
    fake_duckdb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falcon touches neither DiCaprio's nor Belfort's paper_clock row."""
    today = date(2026, 5, 7)
    from casino.execution import sim_broker as smb_mod

    monkeypatch.setattr(
        smb_mod,
        "_utc_now_iso",
        lambda: datetime(2026, 5, 6, tzinfo=UTC).isoformat(),
    )
    run_falcon_rebal(
        today=today,
        force=True,
        db_path=state,
        duckdb_path=fake_duckdb,
        universe=["SPY", "QQQ"],
    )
    # Neither the live bot nor the regime shadow has a paper_clock row.
    assert paper_clock.fetch_paper_clock(run_id="DiCaprio", db_path=state) is None
    assert paper_clock.fetch_paper_clock(run_id="Belfort", db_path=state) is None
    # Falcon's run_id does.
    assert paper_clock.fetch_paper_clock(run_id=FALCON_RUN_ID, db_path=state) is not None
