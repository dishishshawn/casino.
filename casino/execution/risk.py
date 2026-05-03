"""Position sizing, stops, and the kill switch.

Authoritative for sizing and exposure caps. Brokers and signals are *inputs* to
risk; risk is not an advisory layer downstream of them.

# Phase 3: implemented in task 21.
"""

from __future__ import annotations
