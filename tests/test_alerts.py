"""Tests for casino.monitoring.alerts."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx
import pytest

from casino.config import get_config
from casino.monitoring import alerts


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    get_config.cache_clear()


def _capture() -> tuple[list[tuple[str, dict[str, Any]]], alerts.WebhookTransport]:
    captured: list[tuple[str, dict[str, Any]]] = []

    def transport(url: str, payload: dict[str, Any]) -> httpx.Response:
        captured.append((url, payload))
        return httpx.Response(204)

    return captured, transport


def test_fire_sends_payload_when_url_configured() -> None:
    captured, tx = _capture()
    result = alerts.fire(
        title="Hello",
        message="World",
        severity="info",
        transport=tx,
    )
    assert result.sent is True
    assert result.status_code == 204
    assert len(captured) == 1
    url, payload = captured[0]
    assert url == "https://example.invalid/webhook"
    assert payload["embeds"][0]["title"] == "Hello"


def test_fire_no_url_logs_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear at the env layer AND override on the config object so a populated
    # local .env can't satisfy the URL check. Otherwise dotenv re-supplies it.
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
    get_config.cache_clear()
    captured, tx = _capture()
    # Pass an explicit empty webhook_url so the test does not depend on
    # whether .env happens to be populated in the developer's working tree.
    result = alerts.fire(
        title="x",
        message="y",
        severity="warning",
        transport=tx,
        webhook_url="",
    )
    assert result.sent is False
    assert "not configured" in result.reason
    assert captured == []


def test_alert_drawdown_breach_critical_at_threshold() -> None:
    captured, tx = _capture()
    result = alerts.alert_drawdown_breach(
        drawdown_pct=0.10,
        high_water_mark=Decimal("100000"),
        current_equity=Decimal("90000"),
        transport=tx,
    )
    assert result.sent is True
    assert captured[0][1]["embeds"][0]["color"] == 0xE74C3C  # red / critical


def test_alert_llm_spend_warns_when_over() -> None:
    captured, tx = _capture()
    alerts.alert_llm_spend(
        daily_spend_usd=Decimal("6.00"),
        threshold_usd=Decimal("5.00"),
        transport=tx,
    )
    embed = captured[0][1]["embeds"][0]
    assert embed["color"] == 0xF1C40F  # warning / amber


def test_alert_reconciliation_drift_critical() -> None:
    captured, tx = _capture()
    alerts.alert_reconciliation_drift(
        n_drift=2,
        summary="AAA broker_only; BBB qty_mismatch",
        transport=tx,
    )
    embed = captured[0][1]["embeds"][0]
    assert embed["color"] == 0xE74C3C
    # New copy splits the semicolon-separated summary into bullet lines
    # and prepends a plain-English explainer. Verify both halves landed.
    assert "Positions out of sync" in embed["title"]
    assert "AAA broker_only" in embed["description"]
    assert "BBB qty_mismatch" in embed["description"]
    assert "RUNBOOK" in embed["description"]


def test_alert_kill_criterion_plain_english_and_pct() -> None:
    captured, tx = _capture()
    alerts.alert_kill_criterion(
        criterion="reconcile_drift",
        value=Decimal("0.03696150344177440814098986236"),
        threshold=Decimal("0.01"),
        nav=Decimal("100009.46"),
        days_elapsed=None,
        detail="|sum(broker_mv - book_notional)|=$3696.5 nav=$100009.46 drift=3.6962% entries=1",
        transport=tx,
    )
    embed = captured[0][1]["embeds"][0]
    assert embed["color"] == 0xE74C3C
    assert "Trading halted" in embed["title"]
    assert "positions out of sync with broker" in embed["title"]
    # Percentages are rounded for human readability.
    field_map = {f["name"]: f["value"] for f in embed["fields"]}
    assert field_map["Measured"] == "3.70%"
    assert field_map["Safety limit"] == "1.00%"
    assert field_map["NAV"] == "$100,009.46"
    assert field_map["Status"] == "trading disabled"
    # Recovery hint is included.
    assert "trading_disabled" in embed["description"]


def test_alert_unhandled_exception_critical() -> None:
    captured, tx = _capture()
    alerts.alert_unhandled_exception(
        job="earnings_daily",
        exc_type="RuntimeError",
        detail="boom",
        transport=tx,
    )
    embed = captured[0][1]["embeds"][0]
    assert embed["color"] == 0xE74C3C
    assert "earnings_daily" in embed["title"]


def test_alert_order_fill_info() -> None:
    captured, tx = _capture()
    alerts.alert_order_fill(
        symbol="AAA",
        side="buy",
        qty=10,
        price=Decimal("100.50"),
        order_id="ord-1",
        transport=tx,
    )
    embed = captured[0][1]["embeds"][0]
    assert embed["color"] == 0x2ECC71  # info / green
    assert "AAA" in embed["title"]


def test_alert_order_submitted_info() -> None:
    captured, tx = _capture()
    alerts.alert_order_submitted(
        run_id="DiCaprio",
        symbol="SPY",
        side="buy",
        qty=5,
        reference_price=Decimal("737.62"),
        stop_price=Decimal("663.86"),
        order_id="ord-abc",
        transport=tx,
    )
    embed = captured[0][1]["embeds"][0]
    assert embed["color"] == 0x2ECC71  # info / green
    assert "DiCaprio" in embed["title"]
    assert "SPY" in embed["title"]
    assert "ord-abc" in embed["description"]


def test_alert_rebal_summary_info_when_no_drift() -> None:
    captured, tx = _capture()
    alerts.alert_rebal_summary(
        run_id="DiCaprio",
        rebal_date="2026-05-08",
        nav=Decimal("100000"),
        n_orders_submitted=1,
        target_weights=[{"symbol": "SPY", "weight": 0.86}],
        drift_after=0,
        forced=True,
        dry_run=False,
        transport=tx,
    )
    embed = captured[0][1]["embeds"][0]
    assert embed["color"] == 0x2ECC71  # info / green
    assert "DiCaprio" in embed["title"]
    assert "FORCED" in embed["title"]


def test_alert_rebal_summary_warning_when_drift() -> None:
    captured, tx = _capture()
    alerts.alert_rebal_summary(
        run_id="DiCaprio",
        rebal_date="2026-05-08",
        nav=Decimal("100000"),
        n_orders_submitted=1,
        target_weights=[{"symbol": "SPY", "weight": 0.86}],
        drift_after=2,
        forced=False,
        dry_run=False,
        transport=tx,
    )
    embed = captured[0][1]["embeds"][0]
    assert embed["color"] == 0xF1C40F  # warning / amber


def test_alert_handles_transport_exception() -> None:
    def boom(_url: str, _payload: dict[str, Any]) -> httpx.Response:
        raise httpx.ConnectError("nope")

    result = alerts.fire(
        title="t",
        message="m",
        severity="critical",
        transport=boom,
    )
    assert result.sent is False
    assert "transport error" in result.reason
