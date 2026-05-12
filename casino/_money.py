"""Money rounding — single source of truth across runners.

Until 2026-05-11 the helper ``_round_money(d) -> Decimal`` was duplicated
inside ``casino.execution.tsmom_runner`` and
``casino.execution.tsmom_shadow_runner``. Both used ``ROUND_HALF_UP`` at
two decimals, but nothing structurally prevented one side from drifting
(e.g., switching to ``ROUND_DOWN`` for conservatism) — which would
silently make live and shadow sizing diverge and contaminate the
Belfort-vs-DiCaprio comparison that is the whole point of running a
shadow.

This module is the canonical home. The semantics are deliberately
pinned by the unit tests in ``tests/test_money.py``:

* Two-decimal quantization (cents precision).
* Banker's rounding is NOT used; we use ``ROUND_HALF_UP`` so 0.125 -> 0.13.
* Negative values round away from zero on a half (so -0.125 -> -0.13).

Pure-function. No internal dependencies beyond ``decimal``.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

# Quantization grid for whole-cent money. Exposed so callers can reuse
# it (e.g., for ``Decimal.quantize`` calls outside this module) without
# re-deriving the string literal.
MONEY_QUANTUM: Decimal = Decimal("0.01")


def round_money(d: Decimal) -> Decimal:
    """Round a Decimal money value to whole cents using ROUND_HALF_UP.

    The runners use this to floor target dollar allocations before
    dividing by reference price for whole-share sizing. The contract
    (two decimals, half-up) is load-bearing for live/shadow parity.
    """
    return d.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def floor_shares(target_dollars: Decimal, reference_price: Decimal) -> int:
    """Convert a dollar target to a whole-share count, ALWAYS rounding down.

    Used by the rebal planners on both the live and shadow runners. The
    contract:

    * ``target_dollars / reference_price`` is computed as Decimal.
    * The result is quantized to an integer with ``ROUND_DOWN`` — never
      half-up. The docstrings in both runners say "whole-share qty floored
      conservatively"; ROUND_DOWN is what makes that comment true and
      prevents the planner from sizing one share over the single-name cap.
    * ``reference_price <= 0`` is treated as "cannot price; return zero
      shares" rather than raising. Callers downstream of the planner
      already handle a 0-qty action by emitting a ``skip`` action with
      a reason.

    Pre-2026-05-12 the live runner used ``ROUND_HALF_UP`` for this step
    while the shadow used ``ROUND_DOWN``. Task 47 unified both to
    ``ROUND_DOWN`` so the Belfort-vs-DiCaprio comparison isn't
    contaminated by a one-share-per-name discrepancy.
    """
    if reference_price <= Decimal("0"):
        return 0
    return int((target_dollars / reference_price).to_integral_value(rounding=ROUND_DOWN))
