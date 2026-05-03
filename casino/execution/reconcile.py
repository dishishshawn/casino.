"""Broker-vs-internal-book reconciliation.

The source of truth for "what we actually hold." Any module reasoning about
positions reads from reconcile, not from the broker API directly.

# Phase 3: implemented in task 22.
"""

from __future__ import annotations
