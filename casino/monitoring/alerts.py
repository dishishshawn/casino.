"""Discord webhook alerts.

Centralized alert dispatcher used by jobs, the kill switch, the EOD
reconciler, and the cost-budget guard. PRD §10 alert rules:

* Order fills (info)
* Drawdown > 10% (critical)
* Daily LLM spend > $5 (warning)
* Broker reconciliation mismatch (critical)
* Unhandled exceptions in any cron job (critical)

Implementation notes:

* The webhook URL comes from `casino.config.discord_webhook_url`; when
  unset, alerts are logged at the requested severity but not sent. This
  lets development run without a Discord URL configured.
* Network is stubbable: tests inject a `transport` callable that captures
  payloads instead of POSTing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

import httpx
from loguru import logger

from casino.config import get_config

Severity = Literal["info", "warning", "critical"]


# Discord embed colors (RGB-encoded ints).
_COLOR_BY_SEVERITY: dict[Severity, int] = {
    "info": 0x2ECC71,  # green
    "warning": 0xF1C40F,  # amber
    "critical": 0xE74C3C,  # red
}


# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class Alert:
    """Structured alert envelope."""

    title: str
    message: str
    severity: Severity
    fields: dict[str, str]
    timestamp_utc: datetime


@dataclass(frozen=True)
class AlertResult:
    """Outcome of one dispatch."""

    sent: bool
    status_code: int | None
    reason: str


WebhookTransport = Callable[[str, dict[str, Any]], httpx.Response]


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _default_transport(url: str, payload: dict[str, Any]) -> httpx.Response:
    """Default httpx-backed POST. Synchronous, 10 s timeout."""
    return httpx.post(url, json=payload, timeout=10.0)


# ---------------------------------------------------------------------------- core dispatch


def _build_payload(alert: Alert) -> dict[str, Any]:
    """Construct a Discord webhook payload from an Alert."""
    fields = [
        {"name": k, "value": v[:1024], "inline": len(v) < 64} for k, v in alert.fields.items()
    ]
    embed: dict[str, Any] = {
        "title": alert.title[:256],
        "description": alert.message[:4000],
        "color": _COLOR_BY_SEVERITY.get(alert.severity, 0x7289DA),
        "timestamp": alert.timestamp_utc.isoformat(),
        "fields": fields,
        "footer": {"text": f"casino · {alert.severity}"},
    }
    return {"embeds": [embed]}


def send_alert(
    alert: Alert,
    *,
    webhook_url: str | None = None,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """Send an alert to Discord. Returns success/failure metadata.

    Logs at a severity-appropriate level regardless of whether the
    webhook was reachable, so the local log is always complete.
    """
    log_fn = {
        "info": logger.info,
        "warning": logger.warning,
        "critical": logger.error,
    }[alert.severity]
    log_fn("ALERT [{}] {}: {}", alert.severity, alert.title, alert.message)

    cfg = get_config()
    url = webhook_url if webhook_url is not None else cfg.discord_webhook_url
    if not url:
        return AlertResult(
            sent=False,
            status_code=None,
            reason="discord_webhook_url not configured; alert logged only",
        )

    payload = _build_payload(alert)
    tx: WebhookTransport = transport if transport is not None else _default_transport
    try:
        resp = tx(url, payload)
    except Exception as e:  # noqa: BLE001 — network errors must not crash callers
        logger.error("alerts: webhook POST failed: {}", e)
        return AlertResult(sent=False, status_code=None, reason=f"transport error: {e}")
    ok = 200 <= resp.status_code < 300
    return AlertResult(
        sent=ok,
        status_code=resp.status_code,
        reason="ok" if ok else f"HTTP {resp.status_code}",
    )


def fire(
    *,
    title: str,
    message: str,
    severity: Severity = "info",
    fields: dict[str, str] | None = None,
    webhook_url: str | None = None,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """Convenience wrapper: build an Alert and dispatch it."""
    return send_alert(
        Alert(
            title=title,
            message=message,
            severity=severity,
            fields=fields or {},
            timestamp_utc=_utc_now(),
        ),
        webhook_url=webhook_url,
        transport=transport,
    )


# ---------------------------------------------------------------------------- typed helpers (PRD §10 rules)


def alert_order_fill(
    *,
    symbol: str,
    side: str,
    qty: int,
    price: Decimal,
    order_id: str,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    return fire(
        title=f"Order filled: {side.upper()} {qty} {symbol}",
        message=f"Filled at {price} (broker order {order_id})",
        severity="info",
        fields={
            "Symbol": symbol,
            "Side": side,
            "Quantity": str(qty),
            "Price": str(price),
            "Order ID": order_id,
        },
        transport=transport,
    )


def alert_drawdown_breach(
    *,
    drawdown_pct: float,
    high_water_mark: Decimal,
    current_equity: Decimal,
    threshold_pct: float = 0.10,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """Drawdown alert. PRD §10: fires at 10% drawdown.

    `drawdown_pct` is the decimal fraction (0.10 == 10%).
    """
    sev: Severity = "critical" if drawdown_pct >= threshold_pct else "warning"
    return fire(
        title=f"Drawdown {drawdown_pct:.2%}",
        message=(
            f"Equity {current_equity} vs high-water mark {high_water_mark}. "
            f"Threshold = {threshold_pct:.0%}."
        ),
        severity=sev,
        fields={
            "Drawdown": f"{drawdown_pct:.2%}",
            "Equity": str(current_equity),
            "High-water mark": str(high_water_mark),
        },
        transport=transport,
    )


def alert_llm_spend(
    *,
    daily_spend_usd: Decimal,
    threshold_usd: Decimal = Decimal("5.00"),
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """LLM cost alert. PRD §10: fires when daily LLM spend > $5."""
    over = daily_spend_usd > threshold_usd
    return fire(
        title="LLM daily spend" + (" over budget" if over else " update"),
        message=f"Today's LLM spend: ${daily_spend_usd}. Threshold: ${threshold_usd}.",
        severity="warning" if over else "info",
        fields={"Spend (USD)": str(daily_spend_usd), "Threshold": str(threshold_usd)},
        transport=transport,
    )


def alert_reconciliation_drift(
    *,
    n_drift: int,
    summary: str,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """Reconciliation alert. PRD §10: any broker-vs-book mismatch is critical."""
    return fire(
        title=f"Reconciliation drift: {n_drift} discrepanc{'y' if n_drift == 1 else 'ies'}",
        message=summary[:4000],
        severity="critical",
        fields={"# drift entries": str(n_drift)},
        transport=transport,
    )


def alert_unhandled_exception(
    *,
    job: str,
    exc_type: str,
    detail: str,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """Unhandled-exception alert. PRD §10: any cron job blow-up is critical."""
    return fire(
        title=f"Unhandled exception in {job}",
        message=f"{exc_type}: {detail}"[:4000],
        severity="critical",
        fields={"Job": job, "Exception": exc_type},
        transport=transport,
    )
