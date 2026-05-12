"""Tests pinning the single source of truth for ``TSMOM_UNIVERSE``.

Reasoning: pre-2026-05-11, ``TSMOM_UNIVERSE`` was duplicated as a literal
tuple in ``casino.execution.tsmom_runner`` and ``casino.jobs.heartbeat``
(P1 #7 from ``.taskmaster/docs/structure_review.md``). If the runner's
universe is updated and the heartbeat's freshness gate isn't, the system
trades a ticker the OHLCV-freshness watch isn't monitoring — a yfinance
ingest failure on the new ticker would not trip the daily warning.

These tests guarantee:

1. The canonical definition lives in ``casino.signals.ts_momentum``.
2. The three known consumers reference the same tuple object (identity,
   not just equality). A future refactor that reintroduces a local copy
   fails this test immediately.
3. The universe contains the expected 10-symbol cross-asset ETF set.
   Changing this on purpose is fine — the test breaks loudly so the
   operator knows to also re-run any historical baselines that assumed
   it.
"""

from __future__ import annotations

from casino.execution import tsmom_runner, tsmom_shadow_runner
from casino.jobs import heartbeat
from casino.signals import ts_momentum

EXPECTED_UNIVERSE: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "GLD",
    "DBC",
    "USO",
)


def test_canonical_universe_value() -> None:
    """The canonical tuple is the 10-symbol cross-asset ETF set."""
    assert ts_momentum.TSMOM_UNIVERSE == EXPECTED_UNIVERSE


def test_runner_imports_canonical_universe() -> None:
    """tsmom_runner re-exports the canonical tuple — same object, not a copy."""
    assert tsmom_runner.TSMOM_UNIVERSE is ts_momentum.TSMOM_UNIVERSE


def test_shadow_runner_imports_canonical_universe() -> None:
    """Shadow runner reads the same universe so live and shadow trade the same set."""
    assert tsmom_shadow_runner.TSMOM_UNIVERSE is ts_momentum.TSMOM_UNIVERSE


def test_heartbeat_imports_canonical_universe() -> None:
    """Heartbeat freshness gate watches the same set the runners trade."""
    assert heartbeat.TSMOM_UNIVERSE is ts_momentum.TSMOM_UNIVERSE


def test_universe_is_immutable_tuple() -> None:
    """Tuple not list — accidental mutation would propagate to every consumer."""
    assert isinstance(ts_momentum.TSMOM_UNIVERSE, tuple)
