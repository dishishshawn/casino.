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


def alert_order_submitted(
    *,
    run_id: str,
    symbol: str,
    side: str,
    qty: int,
    reference_price: Decimal,
    stop_price: Decimal,
    order_id: str,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """Fired when the runner submits a bracket order to the broker.

    Distinct from `alert_order_fill` which fires when the order actually
    fills. Submission != fill: paper orders placed after-hours sit
    ACCEPTED until next market open.
    """
    ref_str = _humanize_money(reference_price)
    stop_str = _humanize_money(stop_price)
    return fire(
        title=f"[{run_id}] Order submitted: {side.upper()} {qty} {symbol}",
        message=(
            f"Bracket order placed at ~{ref_str} with stop {stop_str}. "
            f"Awaiting fill (broker order {order_id})."
        ),
        severity="info",
        fields={
            "Run": run_id,
            "Symbol": symbol,
            "Side": side,
            "Qty": str(qty),
            "Reference": ref_str,
            "Stop": stop_str,
            "Order ID": order_id,
        },
        transport=transport,
    )


def alert_rebal_summary(
    *,
    run_id: str,
    rebal_date: str,
    nav: Decimal,
    n_orders_submitted: int,
    target_weights: list[dict[str, Any]],
    drift_after: int,
    forced: bool,
    dry_run: bool,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """End-of-rebal summary: orders submitted, target weights, NAV, drift.

    One alert per rebal cycle. Complements per-order `alert_order_submitted`
    by giving the operator a single roll-up of what happened.
    """
    weights_str = (
        ", ".join(f"{w['symbol']} {w['weight']:.1%}" for w in target_weights if w.get("weight"))
        or "(no nonzero weights)"
    )
    severity: Severity = "warning" if drift_after else "info"
    flags = []
    if forced:
        flags.append("FORCED")
    if dry_run:
        flags.append("DRY-RUN")
    flag_str = f" [{' '.join(flags)}]" if flags else ""
    return fire(
        title=f"[{run_id}] Rebal {rebal_date}{flag_str}: {n_orders_submitted} orders",
        message=(
            f"NAV ${nav}. Submitted {n_orders_submitted} bracket order(s). "
            f"Reconcile drift: {drift_after}. Targets: {weights_str}."
        ),
        severity=severity,
        fields={
            "Run": run_id,
            "Rebal date": rebal_date,
            "NAV": str(nav),
            "Orders submitted": str(n_orders_submitted),
            "Reconcile drift": str(drift_after),
            "Targets": weights_str[:1024],
        },
        transport=transport,
    )


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


def alert_external_position_change(
    *,
    symbols: list[str],
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """Fired when positions vanish at the broker with no bot sell order.

    The 2026-06-02/03 incident: 7 positions were liquidated overnight by a
    close that did not originate from the bot (Alpaca-generated order ids, no
    `kill_event`, the `trading_disabled` flag untouched). The morning book
    sync silently overwrote the book to match, so nothing alerted. This makes
    that case loud: a position the bot did not close disappearing is a
    control breach, not a routine sync.
    """
    sym_str = ", ".join(symbols) or "(none)"
    return fire(
        title=f"Positions closed OUTSIDE the bot: {sym_str}",
        message=(
            f"{len(symbols)} position(s) present in the book are gone at the "
            "broker with no matching bot sell order. Something flattened them "
            "outside the system (manual close, dashboard, or broker action). "
            "Investigate the broker activity log before the next rebal."
        ),
        severity="critical",
        fields={
            "Symbols": sym_str,
            "Count": str(len(symbols)),
        },
        transport=transport,
    )


def alert_stop_rearmed(
    *,
    armed: list[str],
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """Fired when the EOD guard finds an unprotected position and re-arms it.

    Every held position must carry a live broker-side stop (CLAUDE.md hard
    rule 3). A position showing up here means its stop was missing — usually
    because the bracket entry's DAY-tif stop leg expired — and the guard had
    to submit a fresh GTC stop. Healthy days arm nothing.
    """
    sym_str = ", ".join(armed) or "(none)"
    return fire(
        title=f"Re-armed missing stop(s): {sym_str}",
        message=(
            f"{len(armed)} held position(s) had no live broker stop and were "
            "re-armed with a GTC stop. If this recurs every day, the bracket "
            "entry's stop leg is expiring (DAY tif) and needs a durable fix."
        ),
        severity="warning",
        fields={
            "Symbols": sym_str,
            "Count": str(len(armed)),
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
    """Reconciliation alert. PRD §10: any broker-vs-book mismatch is critical.

    Wording is deliberately plain-English: the operator should be able to
    read the embed and know (a) what's wrong, (b) why it matters, and
    (c) where to look — without having to consult the source.
    """
    word = "issue" if n_drift == 1 else "issues"
    intro = (
        f"Alpaca and the system's internal records disagree on {n_drift} "
        f"{'position' if n_drift == 1 else 'positions'}. Until this is "
        "resolved the system can't trust its own state, so the next safety "
        "check may halt trading. Investigate per RUNBOOK §5."
    )
    body = f"{intro}\n\n• " + summary.replace("; ", "\n• ")
    return fire(
        title=f"Positions out of sync ({n_drift} {word})",
        message=body[:4000],
        severity="critical",
        fields={"Issues found": str(n_drift)},
        transport=transport,
    )


# Plain-English titles + one-line descriptions for each kill criterion.
# Keep the keys in sync with `casino.execution.tsmom_clock_check`. New
# criteria must be added here too — falling back to the criterion name is a
# bug, not a fallback (the whole point of this helper is no jargon).
_KILL_CRITERION_COPY: dict[str, tuple[str, str]] = {
    "drawdown": (
        "account is down too much from start",
        "Total paper-trading losses crossed the kill threshold.",
    ),
    "single_day": (
        "single-day loss exceeded the safety limit",
        "One day's loss was larger than the safety limit allows.",
    ),
    "cap_violation": (
        "a position exceeded portfolio size caps",
        "Either gross exposure or a single position grew past the configured cap.",
    ),
    "reconcile_drift": (
        "positions out of sync with broker",
        "Alpaca and the system's internal records disagree on positions by "
        "more than the safety limit. Sizing decisions can't be trusted until "
        "this is fixed (RUNBOOK §5).",
    ),
    "ks_test": (
        "paper returns no longer match the backtest",
        "The distribution of paper-trade daily returns has diverged "
        "significantly from the backtest baseline.",
    ),
}


def _humanize_pct(d: Decimal) -> str:
    """Render a fraction as a percentage with 2 decimals (e.g. 3.70%)."""
    return f"{float(d) * 100:.2f}%"


def _humanize_money(d: Decimal) -> str:
    """Render a money amount with $ and thousands separators."""
    try:
        f = float(d)
    except (TypeError, ValueError):
        return str(d)
    return f"${f:,.2f}"


def alert_kill_criterion(
    *,
    criterion: str,
    value: Decimal,
    threshold: Decimal,
    nav: Decimal,
    days_elapsed: int | None,
    detail: str,
    transport: WebhookTransport | None = None,
) -> AlertResult:
    """Fired by ``tsmom_clock_check`` when a kill criterion trips.

    Replaces the previous ad-hoc ``alerts.fire(title="... KILL CRITERION
    [reconcile_drift]", ...)`` call which read like a log line. The new
    embed leads with the plain-English reason, rounds raw decimals to
    something a human can scan, and tells the operator that trading has
    been disabled + where to recover.
    """
    plain_reason, explanation = _KILL_CRITERION_COPY.get(
        criterion,
        (criterion.replace("_", " "), f"Kill criterion {criterion!r} tripped."),
    )

    # Percent-style criteria render the value/threshold as percentages so
    # "0.0370 vs 0.0100" doesn't show up as a raw Decimal in the embed.
    pct_criteria = {"drawdown", "single_day", "cap_violation", "reconcile_drift"}
    if criterion in pct_criteria:
        value_str = _humanize_pct(value)
        threshold_str = _humanize_pct(threshold)
    else:
        # ks_test is a p-value, kept as a fixed-decimal number.
        try:
            value_str = f"{float(value):.4f}"
            threshold_str = f"{float(threshold):.4f}"
        except (TypeError, ValueError):
            value_str = str(value)
            threshold_str = str(threshold)

    body = (
        f"{explanation}\n\n"
        f"Trading has been disabled and any open orders cancelled. "
        f"Investigate the cause, then re-arm by clearing the "
        f"`trading_disabled` flag in `state.sqlite`.\n\n"
        f"Technical detail: {detail}"
    )

    fields: dict[str, str] = {
        "Reason": plain_reason,
        "Measured": value_str,
        "Safety limit": threshold_str,
        "NAV": _humanize_money(nav),
        "Status": "trading disabled",
    }
    if days_elapsed is not None:
        fields["Paper-run day"] = str(days_elapsed)

    return fire(
        title=f"Trading halted — {plain_reason}",
        message=body[:4000],
        severity="critical",
        fields=fields,
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
