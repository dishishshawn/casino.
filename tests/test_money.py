"""Tests for casino._money.round_money — pin the rounding contract.

Reasoning: ``round_money`` is consumed by both the live TSMOM runner and
the shadow simulator. Live/shadow parity depends on both sides using
identical rounding. These tests pin the policy (two decimals,
ROUND_HALF_UP, half away from zero) so a future change has to break a
test rather than silently drift.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from casino._money import MONEY_QUANTUM, round_money


def test_round_money_two_decimal_quantum() -> None:
    """MONEY_QUANTUM is Decimal('0.01') — pinned for explicitness."""
    assert MONEY_QUANTUM == Decimal("0.01")


@pytest.mark.parametrize(
    "value,expected",
    [
        # Trivial cases — already at quantum.
        (Decimal("0"), Decimal("0.00")),
        (Decimal("1.00"), Decimal("1.00")),
        (Decimal("100.50"), Decimal("100.50")),
        # Truncation cases.
        (Decimal("0.123"), Decimal("0.12")),
        (Decimal("0.124"), Decimal("0.12")),
        # Half-up rule.
        (Decimal("0.125"), Decimal("0.13")),
        (Decimal("0.135"), Decimal("0.14")),
        (Decimal("1.005"), Decimal("1.01")),
        # Negative — half away from zero.
        (Decimal("-0.125"), Decimal("-0.13")),
        (Decimal("-0.005"), Decimal("-0.01")),
        # Large notionals (NAV-sized).
        (Decimal("100000.005"), Decimal("100000.01")),
        (Decimal("9999.994"), Decimal("9999.99")),
        (Decimal("9999.995"), Decimal("10000.00")),
    ],
)
def test_round_money_cases(value: Decimal, expected: Decimal) -> None:
    """Each case pins a specific rounding outcome under ROUND_HALF_UP."""
    assert round_money(value) == expected


def test_round_money_is_idempotent() -> None:
    """round(round(x)) == round(x) — the function fixes its output."""
    for v in (Decimal("0.125"), Decimal("100.005"), Decimal("-1.234")):
        once = round_money(v)
        assert round_money(once) == once


def test_round_money_preserves_decimal_type() -> None:
    """Return is always Decimal, never float — protects money discipline."""
    result = round_money(Decimal("1.005"))
    assert isinstance(result, Decimal)


def test_runners_share_the_canonical_helper() -> None:
    """Regression: live + shadow runners must use casino._money.round_money.

    The 2026-05-11 structure review (task 46) deduped two private copies
    that could silently diverge. Both runners now import from
    casino._money via an ``as _round_money`` alias for backwards
    compatibility at call sites; this test pins that the alias resolves
    to the same callable on both sides (identity, not just equality).
    """
    from casino import _money
    from casino.execution import tsmom_runner, tsmom_shadow_runner

    assert tsmom_runner._round_money is _money.round_money, (
        "tsmom_runner._round_money diverged from casino._money.round_money"
    )
    assert tsmom_shadow_runner._round_money is _money.round_money, (
        "tsmom_shadow_runner._round_money diverged from casino._money.round_money"
    )
