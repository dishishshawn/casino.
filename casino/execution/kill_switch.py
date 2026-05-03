"""Kill-switch CLI entry: ``python -m casino.execution.kill_switch``.

CLAUDE.md hard rule 4 / PRD §8: the kill switch must remain a single
command that flattens positions and disables order entry.

Usage:

    uv run python -m casino.execution.kill_switch
    uv run python -m casino.execution.kill_switch --reason "drawdown breach"
    uv run python -m casino.execution.kill_switch --reenable     # operator-only

This module is a thin CLI wrapper over `casino.execution.risk.flatten_and_disable`
so that programmatic callers (alerts pipeline, the dashboard) can import the
function while operators run the command from a shell.
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from casino.execution.risk import flatten_and_disable, re_enable_trading


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.execution.kill_switch",
        description="Flatten all positions and disable trading.",
    )
    parser.add_argument(
        "--reason",
        default="manual",
        help="Why the kill switch was engaged (logged + stored).",
    )
    parser.add_argument(
        "--reenable",
        action="store_true",
        help="Clear the trading_disabled flag instead of engaging the switch. "
        "Use only after manual review per RUNBOOK.",
    )
    args = parser.parse_args(argv)

    if args.reenable:
        re_enable_trading()
        logger.warning("trading flag cleared; system will accept orders again")
        return 0

    result = flatten_and_disable(reason=args.reason)
    logger.warning(
        "kill switch result: cancelled={}, closed={}, flag_set={}",
        result.cancelled_orders,
        result.closed_positions,
        result.flag_set,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
