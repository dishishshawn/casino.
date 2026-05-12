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

from casino._money import MONEY_QUANTUM, floor_shares, round_money


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


# ---------------------------------------------------------------------------- floor_shares


@pytest.mark.parametrize(
    "target_dollars,reference_price,expected",
    [
        # Exact divisions — no rounding needed.
        (Decimal("1000.00"), Decimal("100.00"), 10),
        (Decimal("950.00"), Decimal("100.00"), 9),
        (Decimal("0.00"), Decimal("100.00"), 0),
        # Pre-2026-05-12 the live runner used ROUND_HALF_UP and would have
        # returned 11 here; shadow rounded DOWN and returned 10. Task 47
        # unified both to floor-down (10).
        (Decimal("1050.00"), Decimal("100.00"), 10),
        (Decimal("1051.00"), Decimal("100.00"), 10),
        (Decimal("1099.99"), Decimal("100.00"), 10),
        # Just under one share — must return 0, not 1.
        (Decimal("99.99"), Decimal("100.00"), 0),
        # Realistic TSMOM sizing: ~10% of $100k NAV / ~$430 SPY.
        (Decimal("10000.00"), Decimal("430.00"), 23),
        # Negative target_dollars shouldn't happen in practice (planner
        # filters with `if target_dollars <= 0: continue`) but the helper
        # rounds towards zero, so the integer result follows ROUND_DOWN.
        (Decimal("-50.00"), Decimal("100.00"), 0),
    ],
)
def test_floor_shares_cases(
    target_dollars: Decimal, reference_price: Decimal, expected: int
) -> None:
    """Each case pins floor-down semantics. ROUND_HALF_UP would fail many of these."""
    assert floor_shares(target_dollars, reference_price) == expected


def test_floor_shares_zero_reference_returns_zero() -> None:
    """Defensive: never divide by zero. Planner emits a skip on zero qty."""
    assert floor_shares(Decimal("1000"), Decimal("0")) == 0
    assert floor_shares(Decimal("1000"), Decimal("-1")) == 0


def test_floor_shares_returns_int() -> None:
    """Return type is int, not Decimal — share counts are integers in v1."""
    result = floor_shares(Decimal("1000"), Decimal("100"))
    assert isinstance(result, int)


def test_runners_share_floor_shares() -> None:
    """Regression for task 47: both runners use casino._money.floor_shares.

    Pre-fix the live runner used ROUND_HALF_UP inline and the shadow used
    ROUND_DOWN inline, producing live/shadow divergence on every name
    where ``target_dollars / ref`` had a fractional part >= 0.5. The
    helper now exists as a single canonical callable both runners
    reference; this test asserts the identity to prevent a future
    refactor reintroducing the divergence.
    """
    from casino import _money
    from casino.execution import tsmom_runner, tsmom_shadow_runner

    assert tsmom_runner.floor_shares is _money.floor_shares
    assert tsmom_shadow_runner.floor_shares is _money.floor_shares
