"""Market regime filter (e.g., SPY > 200-day MA gating).

# Phase 3: implemented in task 14.
"""

from __future__ import annotations

from datetime import datetime


def is_risk_on(*, as_of: datetime) -> bool:
    """Return True if the regime is risk-on. Phase 3."""
    raise NotImplementedError("regime filter is implemented in Phase 3 (task 14)")
