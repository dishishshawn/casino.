"""Test-suite fixtures applied automatically.

Critical guard: clears DISCORD_WEBHOOK_URL for every test so the alert
helpers in `casino.monitoring.alerts` never POST to Discord during pytest.
The runner / clock-check call alerts as a side effect of normal flow; if
the operator's `.env` has a real webhook configured, those calls would
spam Discord on every test run.
"""

from __future__ import annotations

import pytest

from casino.config import get_config


@pytest.fixture(autouse=True)
def _disable_discord_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    get_config.cache_clear()
