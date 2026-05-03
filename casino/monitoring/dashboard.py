"""Streamlit monitoring dashboard.

Reads from the DuckDB store, the LLM audit log, the execution book, and the
broker (read-only) and renders:

* Live P&L (today / MTD / YTD)
* Open positions, with reconciliation flag vs the broker
* Last 50 LLM calls (model, tokens, cost, latency, parsed score)
* Daily LLM spend vs the monthly budget
* Drawdown from high-water mark
* Rolling 60-day Sharpe of the daily P&L series

The data-prep functions are pure and testable; the Streamlit `render`
function is the only place we touch the `streamlit` module. Tests cover
the data-prep functions per the Phase-3 instructions.

Run:

    uv run streamlit run casino/monitoring/dashboard.py
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from casino.config import get_config
from casino.execution import book, reconcile
from casino.execution.alpaca_broker import AlpacaBroker, BrokerPosition
from casino.llm import audit

# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class PnLSummary:
    """P&L roll-ups across windows."""

    today: Decimal
    mtd: Decimal
    ytd: Decimal
    equity_today: Decimal
    high_water_mark: Decimal
    drawdown: Decimal


@dataclass(frozen=True)
class PositionRow:
    """One position row, joined with reconciliation flag."""

    symbol: str
    side: str
    book_qty: int
    broker_qty: int | None
    avg_entry_price: Decimal
    market_price: Decimal | None
    market_value: Decimal | None
    unrealized_pl: Decimal | None
    in_sync: bool


@dataclass(frozen=True)
class LLMCallRow:
    """One LLM audit row, formatted for display."""

    timestamp_local: str  # converted to America/New_York
    model: str
    mode: str
    cost_usd: float
    input_tokens: int
    cached_read_tokens: int
    output_tokens: int
    latency_ms: int
    success: bool
    schema_name: str | None


_NY = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------- helpers


def _to_local(ts_utc: str | datetime) -> str:
    """Render a UTC timestamp string as America/New_York for display."""
    dt = ts_utc if isinstance(ts_utc, datetime) else datetime.fromisoformat(ts_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(_NY).strftime("%Y-%m-%d %H:%M:%S %Z")


def _start_of_month(d: date) -> date:
    return date(d.year, d.month, 1)


def _start_of_year(d: date) -> date:
    return date(d.year, 1, 1)


# ---------------------------------------------------------------------------- P&L


def compute_pnl_summary(
    history: Sequence[book.DailyPnLRow],
    *,
    today: date | None = None,
) -> PnLSummary:
    """Compute today/MTD/YTD P&L from the daily_pnl history table.

    `history` must be ordered most-recent first (matches `fetch_daily_pnl`).
    A missing today row gives a zero `today` field; missing earlier rows
    just shrink the window.
    """
    today = today or datetime.now(tz=UTC).date()
    today_str = today.isoformat()
    mtd_start = _start_of_month(today)
    ytd_start = _start_of_year(today)

    today_pnl = Decimal("0")
    mtd_pnl = Decimal("0")
    ytd_pnl = Decimal("0")
    equity_today = Decimal("0")
    hwm = Decimal("0")
    for row in history:
        d = date.fromisoformat(row.date)
        total = row.realized_pl + row.unrealized_pl
        if row.date == today_str:
            today_pnl += total
            equity_today = row.equity_close
        if d >= mtd_start:
            mtd_pnl += total
        if d >= ytd_start:
            ytd_pnl += total
        if row.equity_close > hwm:
            hwm = row.equity_close
    if equity_today == Decimal("0") and history:
        # No row for today yet: use the most-recent close as the equity.
        equity_today = history[0].equity_close

    drawdown = (hwm - equity_today) / hwm if hwm > Decimal("0") else Decimal("0")
    if drawdown < Decimal("0"):
        drawdown = Decimal("0")
    return PnLSummary(
        today=today_pnl,
        mtd=mtd_pnl,
        ytd=ytd_pnl,
        equity_today=equity_today,
        high_water_mark=hwm,
        drawdown=drawdown,
    )


def rolling_sharpe(
    history: Sequence[book.DailyPnLRow],
    *,
    window: int = 60,
    annualization: int = 252,
) -> float | None:
    """Rolling-`window`-day Sharpe of daily total P&L returns.

    Returns None when fewer than `window` days exist or the equity series
    is degenerate. Returns are computed as ``(realized + unrealized) / equity_open``.
    """
    if len(history) < window:
        return None
    rs: list[float] = []
    for row in list(history)[:window]:
        if row.equity_open <= Decimal("0"):
            continue
        total = row.realized_pl + row.unrealized_pl
        rs.append(float(total / row.equity_open))
    if len(rs) < window // 2:
        return None
    mu = sum(rs) / len(rs)
    var = sum((r - mu) ** 2 for r in rs) / max(len(rs) - 1, 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return None
    return (mu / sd) * math.sqrt(annualization)


# ---------------------------------------------------------------------------- positions


def merge_positions(
    *,
    book_positions: Sequence[book.StoredPosition],
    broker_positions: Sequence[BrokerPosition] = (),
) -> list[PositionRow]:
    """Join the internal book to broker positions for the dashboard table.

    Each row shows what the book thinks plus what the broker reports, with
    an `in_sync` flag (True iff sides + qty match within 1 share).
    """
    by_broker = {p.symbol.upper(): p for p in broker_positions}
    rows: list[PositionRow] = []
    seen: set[str] = set()
    for k in book_positions:
        bp = by_broker.get(k.symbol.upper())
        seen.add(k.symbol.upper())
        in_sync = bp is not None and bp.side == k.side and abs(bp.qty - k.qty) < 1
        rows.append(
            PositionRow(
                symbol=k.symbol,
                side=k.side,
                book_qty=k.qty,
                broker_qty=bp.qty if bp is not None else None,
                avg_entry_price=k.avg_entry_price,
                market_price=bp.market_price if bp is not None else None,
                market_value=bp.market_value if bp is not None else None,
                unrealized_pl=bp.unrealized_pl if bp is not None else None,
                in_sync=in_sync,
            )
        )
    # Broker-only positions: book missing → flagged in_sync=False
    for bp in broker_positions:
        if bp.symbol.upper() in seen:
            continue
        rows.append(
            PositionRow(
                symbol=bp.symbol,
                side=bp.side,
                book_qty=0,
                broker_qty=bp.qty,
                avg_entry_price=bp.avg_entry_price,
                market_price=bp.market_price,
                market_value=bp.market_value,
                unrealized_pl=bp.unrealized_pl,
                in_sync=False,
            )
        )
    rows.sort(key=lambda r: r.symbol)
    return rows


# ---------------------------------------------------------------------------- LLM ledger


def format_llm_calls(
    rows: Iterable[dict[str, Any]],
) -> list[LLMCallRow]:
    """Convert raw audit rows into display rows with localized timestamps."""
    out: list[LLMCallRow] = []
    for r in rows:
        out.append(
            LLMCallRow(
                timestamp_local=_to_local(str(r["timestamp_utc"])),
                model=str(r["model"]),
                mode=str(r["mode"]),
                cost_usd=float(r["cost_usd"]),
                input_tokens=int(r["input_tokens"]),
                cached_read_tokens=int(r["cache_read_tokens"]),
                output_tokens=int(r["output_tokens"]),
                latency_ms=int(r["latency_ms"]),
                success=bool(int(r["success"])),
                schema_name=str(r["schema_name"]) if r.get("schema_name") else None,
            )
        )
    return out


def daily_llm_spend_by_day(
    *,
    n_days: int = 30,
    db_path: Path | None = None,
) -> list[tuple[str, Decimal]]:
    """Return (date, total cost) for the last `n_days` of LLM activity."""
    target = db_path if db_path is not None else get_config().state_sqlite_path
    if not target.exists():
        return []
    sql = """
        SELECT substr(timestamp_utc, 1, 10) AS d,
               COALESCE(SUM(cost_usd), 0)   AS total
        FROM llm_calls
        GROUP BY d
        ORDER BY d DESC
        LIMIT ?
    """
    with sqlite3.connect(str(target)) as conn:
        rows = conn.execute(sql, (n_days,)).fetchall()
    return [(str(r[0]), Decimal(str(r[1]))) for r in rows]


def monthly_llm_spend(*, db_path: Path | None = None) -> Decimal:
    """Return month-to-date LLM spend in USD."""
    target = db_path if db_path is not None else get_config().state_sqlite_path
    if not target.exists():
        return Decimal("0")
    today = datetime.now(tz=UTC).date()
    month_prefix = f"{today.year:04d}-{today.month:02d}-"
    sql = """
        SELECT COALESCE(SUM(cost_usd), 0)
        FROM llm_calls
        WHERE substr(timestamp_utc, 1, 7) = ?
    """
    with sqlite3.connect(str(target)) as conn:
        row = conn.execute(sql, (month_prefix.rstrip("-"),)).fetchone()
    return Decimal(str(row[0] if row else 0))


# ---------------------------------------------------------------------------- glue (no streamlit)


@dataclass(frozen=True)
class DashboardSnapshot:
    """Snapshot of every metric the dashboard renders.

    Computed by `build_snapshot` so tests can verify all values without
    importing `streamlit`.
    """

    pnl: PnLSummary
    positions: list[PositionRow]
    recent_calls: list[LLMCallRow]
    daily_spend: list[tuple[str, Decimal]]
    monthly_spend: Decimal
    monthly_budget: Decimal
    rolling_sharpe_60d: float | None
    trading_disabled: bool


def build_snapshot(
    *,
    broker: AlpacaBroker | None = None,
    db_path: Path | None = None,
    monthly_budget_usd: Decimal = Decimal("100.00"),
    n_calls: int = 50,
) -> DashboardSnapshot:
    """Compute everything the dashboard renders. No streamlit imports."""
    history = book.fetch_daily_pnl(limit=2000, db_path=db_path)
    pnl = compute_pnl_summary(history)

    book_positions = book.fetch_positions(db_path=db_path)
    broker_positions: list[BrokerPosition] = []
    if broker is not None:
        try:
            broker_positions = broker.get_positions()
        except Exception:  # noqa: BLE001 — broker may be unreachable in dev
            broker_positions = []
    positions = merge_positions(
        book_positions=book_positions,
        broker_positions=broker_positions,
    )

    cfg = get_config()
    audit_db = cfg.state_sqlite_path
    recent = audit.fetch_recent_calls(limit=n_calls, db_path=audit_db)
    rendered = format_llm_calls(recent)
    daily = daily_llm_spend_by_day(db_path=audit_db)
    month = monthly_llm_spend(db_path=audit_db)

    sharpe = rolling_sharpe(history, window=60)
    return DashboardSnapshot(
        pnl=pnl,
        positions=positions,
        recent_calls=rendered,
        daily_spend=daily,
        monthly_spend=month,
        monthly_budget=monthly_budget_usd,
        rolling_sharpe_60d=sharpe,
        trading_disabled=book.is_trading_disabled(db_path=db_path),
    )


# ---------------------------------------------------------------------------- streamlit entry


# ---------------------------------------------------------------------------- render helpers
#
# The render layer below is a heavily-customized Streamlit page. It injects raw
# HTML + CSS so the page reads like a late-night trading desk: serif metric
# numbers (Fraunces), data in JetBrains Mono with tabular numbers, a single
# warm amber accent reserved for the brand mark, P&L green/red used only on
# values, sharp 1px rules instead of card chrome. The data-prep layer above
# is untouched — only `render` and its private helpers are visual.


_FONT_IMPORTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Fraunces:opsz,wght@9..144,300;9..144,500;9..144,700;9..144,900&"
    "family=JetBrains+Mono:wght@300;400;500;700&"
    "family=IBM+Plex+Sans+Condensed:wght@400;500;600;700&display=swap"
)

_CSS = """
<style>
@import url('%FONT_IMPORTS%');

:root {
    --bg-base:      #0a0a0a;
    --bg-panel:     #111111;
    --bg-elevated:  #151515;
    --bg-row-hover: #181818;
    --line:         #1f1f1f;
    --line-bright:  #2a2a2a;
    --line-rule:    #333333;
    --text:         #e8e6e3;
    --text-dim:     #888580;
    --text-faint:   #555149;
    --accent:       #e8a33d;
    --accent-dim:   #8a6224;
    --positive:     #3fb950;
    --negative:     #f85149;
    --warn:         #d29922;
}

html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    background: var(--bg-base) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', ui-monospace, 'Cascadia Code', monospace;
    font-feature-settings: "tnum" 1, "ss01" 1, "ss02" 1, "calt" 0;
    font-variant-numeric: tabular-nums;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"], footer { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

#MainMenu, div[data-testid="stStatusWidget"] { display: none !important; }

a, a:visited { color: var(--accent); text-decoration: none; }

/* ============================================ ticker tape */
.ticker-rail {
    background: var(--bg-base);
    border-top: 1px solid var(--line-bright);
    border-bottom: 1px solid var(--line-bright);
    overflow: hidden;
    height: 28px;
    position: relative;
    width: 100%;
}
.ticker-track {
    display: flex;
    gap: 3rem;
    white-space: nowrap;
    animation: ticker-scroll 60s linear infinite;
    padding-left: 100%;
}
@keyframes ticker-scroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-100%); }
}
.ticker-item {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    line-height: 28px;
    color: var(--text-dim);
    letter-spacing: 0.04em;
}
.ticker-item .sym { color: var(--text); font-weight: 500; }
.ticker-item .pl-pos { color: var(--positive); }
.ticker-item .pl-neg { color: var(--negative); }
.ticker-item .arrow { font-size: 9px; }

/* ============================================ brand header */
.brand-band {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    border-bottom: 1px solid var(--line-rule);
    padding: 2.4rem 1.6rem 1.4rem 1.6rem;
    margin-bottom: 0;
}
.brand-mark {
    font-family: 'Fraunces', 'Times New Roman', serif;
    font-weight: 900;
    font-size: 64px;
    line-height: 0.85;
    letter-spacing: -0.04em;
    color: var(--text);
    margin: 0;
    font-feature-settings: "ss01" 1;
}
.brand-mark .dot { color: var(--accent); }
.brand-sub {
    margin-top: 0.6rem;
    font-family: 'IBM Plex Sans Condensed', sans-serif;
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--text-dim);
}
.brand-sub .sep { color: var(--text-faint); margin: 0 0.5rem; }

.brand-meta {
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
    line-height: 1.7;
}
.brand-meta .label {
    font-family: 'IBM Plex Sans Condensed', sans-serif;
    font-weight: 600;
    font-size: 9px;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--text-faint);
    display: inline-block;
    min-width: 7em;
    text-align: left;
    margin-right: 1.2em;
}
.brand-meta .v { color: var(--text); }
.brand-meta .v-disabled { color: var(--negative); font-weight: 700; }
.brand-meta .v-ok { color: var(--positive); }

/* ============================================ kill-switch banner */
.kill-banner {
    background: linear-gradient(90deg, #2c0f10 0%, #1a0a0a 100%);
    border: 1px solid var(--negative);
    border-left-width: 4px;
    padding: 0.9rem 1.4rem;
    margin: 0 1.6rem 1.6rem 1.6rem;
    font-family: 'IBM Plex Sans Condensed', sans-serif;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--negative);
}
.kill-banner .pulse {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: var(--negative);
    border-radius: 50%;
    margin-right: 0.8em;
    animation: pulse 1.4s ease-in-out infinite;
    vertical-align: middle;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.25; }
}

/* ============================================ metric strip */
.metric-strip {
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr 1fr 1fr;
    gap: 0;
    border-bottom: 1px solid var(--line-rule);
    margin: 0 1.6rem;
    padding: 0;
}
.metric-cell {
    padding: 1.6rem 1.4rem 1.4rem 1.4rem;
    border-right: 1px solid var(--line);
    position: relative;
}
.metric-cell:last-child { border-right: none; }
.metric-cell .lbl {
    font-family: 'IBM Plex Sans Condensed', sans-serif;
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    color: var(--text-faint);
    display: block;
    margin-bottom: 0.4rem;
}
.metric-cell .num {
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: 38px;
    line-height: 1;
    letter-spacing: -0.025em;
    color: var(--text);
    font-feature-settings: "tnum" 1, "ss01" 1;
    font-variant-numeric: tabular-nums;
}
.metric-cell.lg .num { font-size: 48px; font-weight: 600; }
.metric-cell .num.pos { color: var(--positive); }
.metric-cell .num.neg { color: var(--negative); }
.metric-cell .num.warn { color: var(--warn); }
.metric-cell .sub {
    margin-top: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 0.02em;
}
.metric-cell .sub .lo { color: var(--text-faint); }

/* ============================================ ops bar (action buttons) */
.ops-bar-frame {
    margin: 1.4rem 1.6rem 0 1.6rem;
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: 0.6rem 0;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 1.5rem;
    align-items: center;
}
.ops-label {
    font-family: 'IBM Plex Sans Condensed', sans-serif;
    font-weight: 700;
    font-size: 9px;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--text-faint);
    padding-right: 0.8rem;
    border-right: 1px solid var(--line);
}
.action-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
    padding: 0.4rem 0.8rem;
    background: var(--bg-elevated);
    border-left: 2px solid var(--line-bright);
    margin-left: 0.4rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.action-log .v-ok { color: var(--positive); margin-right: 0.4em; }
.action-log .v-bad { color: var(--negative); margin-right: 0.4em; }

/* Streamlit button override — sharp, dark, monospace.
   Streamlit wraps button text in nested <div data-testid="stMarkdownContainer"><p>...</p></div>;
   we reset margins on every nested element so the icon stays inline and labels never wrap. */
[data-testid="stButton"] > button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"],
.stButton button {
    background: var(--bg-elevated) !important;
    color: var(--text) !important;
    border: 1px solid var(--line-bright) !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Sans Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0 12px !important;
    height: 34px !important;
    min-height: 34px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    line-height: 1 !important;
    transition: background 80ms ease, border-color 80ms ease, color 80ms ease;
    box-shadow: none !important;
    vertical-align: middle !important;
}
/* Reset nested elements so the icon + text are flex children of the button */
[data-testid="stButton"] > button > *,
[data-testid="stButton"] > button p,
[data-testid="stButton"] > button div,
[data-testid="stButton"] > button span,
.stButton button > *,
.stButton button p,
.stButton button div,
.stButton button span {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    font-family: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    letter-spacing: inherit !important;
    text-transform: inherit !important;
    color: inherit !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
}
[data-testid="stButton"] > button:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover,
.stButton button:hover {
    background: #1c1c1c !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
[data-testid="stButton"] > button[kind="primary"],
[data-testid="baseButton-primary"],
.stButton button[kind="primary"] {
    background: #2c0f10 !important;
    border-color: var(--negative) !important;
    color: var(--negative) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
.stButton button[kind="primary"]:hover {
    background: #3a1213 !important;
    color: #ff7b73 !important;
}
[data-testid="stButton"] > button:disabled,
.stButton button:disabled {
    opacity: 0.4 !important;
    cursor: not-allowed !important;
}
/* Fix vertical alignment of streamlit columns — they default to align-items: stretch
   which sometimes pushes button rows down. Force baseline within the ops bar. */
.ops-bar-frame [data-testid="stHorizontalBlock"] {
    align-items: center !important;
}

/* spinner styling */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ============================================ section heads */
.section {
    margin: 2.2rem 1.6rem 0.6rem 1.6rem;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 2rem;
    border-bottom: 1px solid var(--line);
    padding-bottom: 0.4rem;
}
.section h2 {
    font-family: 'IBM Plex Sans Condensed', sans-serif;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--text);
    margin: 0;
}
.section h2 .num-tag {
    color: var(--accent);
    margin-right: 0.7em;
    font-weight: 800;
}
.section .meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-faint);
    letter-spacing: 0.06em;
}

/* ============================================ data table */
.data {
    margin: 0 1.6rem;
    width: calc(100% - 3.2rem);
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.data th {
    text-align: left;
    font-family: 'IBM Plex Sans Condensed', sans-serif;
    font-weight: 600;
    font-size: 9px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-faint);
    padding: 1rem 0.8rem 0.6rem 0.8rem;
    border-bottom: 1px solid var(--line-bright);
    white-space: nowrap;
}
.data th.r { text-align: right; }
.data th.c { text-align: center; }
.data td {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid var(--line);
    color: var(--text);
    white-space: nowrap;
    font-feature-settings: "tnum" 1;
    font-variant-numeric: tabular-nums;
}
.data td.r { text-align: right; }
.data td.c { text-align: center; }
.data tr:hover td { background: var(--bg-row-hover); }
.data td.dim { color: var(--text-dim); }
.data td.faint { color: var(--text-faint); }
.data td.pos { color: var(--positive); }
.data td.neg { color: var(--negative); }
.data td.warn { color: var(--warn); }
.data td.sym { color: var(--text); font-weight: 500; letter-spacing: 0.02em; }
.data td.side-long { color: var(--positive); }
.data td.side-short { color: var(--negative); }
.data td .sync-ok { color: var(--positive); }
.data td .sync-bad { color: var(--negative); }
.data td .leader {
    color: var(--text-faint);
    letter-spacing: 0;
}
.data .empty {
    text-align: center;
    color: var(--text-faint);
    font-family: 'IBM Plex Sans Condensed', sans-serif;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 2rem 0.8rem !important;
    font-size: 10px;
}

/* ============================================ split body */
.split {
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    gap: 0;
    margin-top: 0;
}
.split > div {
    border-right: 1px solid var(--line);
}
.split > div:last-child { border-right: none; }

/* ============================================ spend bar */
.spend-row {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: 1rem;
    margin: 0.8rem 1.6rem 0 1.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
}
.spend-bar {
    height: 4px;
    background: var(--line);
    border-radius: 0;
    overflow: hidden;
    position: relative;
}
.spend-bar .fill {
    height: 100%;
    background: var(--accent);
}
.spend-bar .fill.warn { background: var(--warn); }
.spend-bar .fill.over { background: var(--negative); }
.spend-row .pct { font-feature-settings: "tnum" 1; color: var(--text); }

/* ============================================ chart container */
.chart-frame {
    margin: 0.6rem 1.6rem 0 1.6rem;
    border: 1px solid var(--line);
    background: var(--bg-panel);
    padding: 0;
}

/* ============================================ status bar */
.status {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--bg-base);
    border-top: 1px solid var(--line-bright);
    padding: 0.45rem 1.6rem;
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-faint);
    letter-spacing: 0.04em;
    z-index: 999;
}
.status .item .lbl {
    font-family: 'IBM Plex Sans Condensed', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.24em;
    color: var(--text-faint);
    margin-right: 0.5em;
}
.status .item .v { color: var(--text-dim); }
.status .item .v-ok { color: var(--positive); }
.status .item .v-warn { color: var(--warn); }
.status .item .v-bad { color: var(--negative); }
.status .item + .item { margin-left: 2rem; }
.dot-live {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 0.4em;
    background: var(--positive);
    box-shadow: 0 0 6px var(--positive);
}
.dot-live.bad { background: var(--negative); box-shadow: 0 0 6px var(--negative); }

/* hide streamlit's default H1 + paddings if any leak */
h1, h2, h3, h4 { font-family: 'Fraunces', serif !important; }
</style>
"""


def _fmt_money(v: Decimal | float, *, signed: bool = False, decimals: int = 2) -> str:
    """Format a money value with thousands separators and sign."""
    d = v if isinstance(v, Decimal) else Decimal(str(v))
    sign = ""
    if signed:
        if d > 0:
            sign = "+"
        elif d < 0:
            sign = "−"
            d = -d
    elif d < 0:
        sign = "−"
        d = -d
    quant = Decimal(10) ** -decimals
    q = d.quantize(quant)
    s = f"{q:,.{decimals}f}"
    return f"{sign}${s}"


def _fmt_pct(v: Decimal | float, *, decimals: int = 2, signed: bool = False) -> str:
    f = float(v) * 100
    if signed:
        sign = "+" if f > 0 else ("−" if f < 0 else "")
        return f"{sign}{abs(f):.{decimals}f}%"
    return f"{f:.{decimals}f}%"


def _pl_class(v: Decimal | float) -> str:
    f = float(v)
    if f > 0:
        return "pos"
    if f < 0:
        return "neg"
    return ""


def _ticker_tape_html(positions: Sequence[PositionRow]) -> str:
    """Build the looping ticker-tape header marquee."""
    if not positions:
        items_html = (
            "<span class='ticker-item'><span class='sym'>NO POSITIONS</span>"
            "<span>·</span><span>WAITING FOR NEXT EARNINGS BASKET</span></span>"
        )
    else:
        parts: list[str] = []
        for p in positions:
            arrow = "▲" if p.side == "long" else "▼"
            mark = _fmt_money(p.market_price, decimals=2) if p.market_price is not None else "—"
            if p.unrealized_pl is not None and p.market_value is not None and p.market_value != 0:
                pl_pct = float(p.unrealized_pl) / abs(float(p.market_value)) * 100
                pl_cls = "pl-pos" if pl_pct >= 0 else "pl-neg"
                pl_sign = "+" if pl_pct >= 0 else ""
                pl_str = f"<span class='{pl_cls}'>{pl_sign}{pl_pct:.2f}%</span>"
            else:
                pl_str = "<span>—</span>"
            parts.append(
                f"<span class='ticker-item'>"
                f"<span class='arrow'>{arrow}</span>"
                f"<span class='sym'>{p.symbol}</span>"
                f"<span>{mark}</span>"
                f"{pl_str}"
                f"</span>"
            )
        # duplicate so the loop is seamless
        items_html = "".join(parts) * 2
    return f"<div class='ticker-rail'><div class='ticker-track'>{items_html}</div></div>"


def _brand_header_html(snap: DashboardSnapshot, *, broker_connected: bool) -> str:
    """Top-of-page brand stamp + system meta."""
    now_ny = datetime.now(tz=UTC).astimezone(_NY)
    ts = now_ny.strftime("%a %b %d · %H:%M:%S %Z")
    trading = (
        "<span class='v-disabled'>DISABLED</span>"
        if snap.trading_disabled
        else "<span class='v-ok'>LIVE · PAPER</span>"
    )
    broker = (
        "<span class='v-ok'>CONNECTED</span>"
        if broker_connected
        else "<span class='v'>OFFLINE · BOOK ONLY</span>"
    )
    return f"""
    <div class='brand-band'>
      <div>
        <div class='brand-mark'>casino<span class='dot'>.</span></div>
        <div class='brand-sub'>
          Earnings-Drift LLM Long-Short
          <span class='sep'>//</span> v1 paper account
          <span class='sep'>//</span> Sonnet 4.6 + Haiku 4.5
        </div>
      </div>
      <div class='brand-meta'>
        <div><span class='label'>Session</span><span class='v'>{ts}</span></div>
        <div><span class='label'>Trading</span>{trading}</div>
        <div><span class='label'>Broker</span>{broker}</div>
      </div>
    </div>
    """


def _kill_banner_html() -> str:
    return (
        "<div class='kill-banner'>"
        "<span class='pulse'></span>"
        "Kill switch engaged · order entry disabled · "
        "click RE-ENABLE in the ops bar to resume"
        "</div>"
    )


def _render_ops_bar(snap: DashboardSnapshot) -> None:  # pragma: no cover
    """Action buttons for ops the operator otherwise runs from a shell.

    All actions run in-process. Long jobs (earnings_daily, reconcile_eod)
    block the page render — acceptable for a single-operator dashboard.
    Destructive actions (kill switch) require two clicks (arm + fire).
    """
    import streamlit as st  # noqa: PLC0415 — render-only

    # Initialize ops session_state defaults.
    st.session_state.setdefault("confirm_kill", False)
    st.session_state.setdefault("last_action", None)
    st.session_state.setdefault("last_action_ok", True)

    st.markdown(
        "<div class='ops-bar-frame'><div class='ops-label'>Operations</div><div>",
        unsafe_allow_html=True,
    )

    # Five equal-width button slots + one wide log slot.
    # Equal ratios keep all buttons on the same grid regardless of label length.
    cols = st.columns([1.4, 1.4, 1.4, 1.4, 1.4, 5])

    # Refresh — always safe.
    with cols[0]:
        if st.button("↻ Refresh", key="op_refresh", use_container_width=True):
            st.rerun()

    # EOD reconcile — safe, writes one daily_pnl row.
    with cols[1]:
        if st.button("▶ EOD Reconcile", key="op_eod", use_container_width=True):
            from casino.jobs.reconcile_eod import (  # noqa: PLC0415
                run_reconcile_eod,
            )

            with st.spinner("running reconcile_eod..."):
                try:
                    eod = run_reconcile_eod()
                    st.session_state["last_action"] = (
                        f"reconcile_eod ok · "
                        f"equity_close=${eod.equity_close} · "
                        f"drift={eod.drift_alerts} · "
                        f"dd={float(eod.drawdown) * 100:.2f}%"
                    )
                    st.session_state["last_action_ok"] = True
                except Exception as e:  # noqa: BLE001
                    st.session_state["last_action"] = f"reconcile_eod FAILED: {e}"
                    st.session_state["last_action_ok"] = False
            st.rerun()

    # Earnings daily — REAL Anthropic API spend + REAL paper orders.
    with cols[2]:
        clicked = st.button(
            "▶ Earnings Daily",
            key="op_earn",
            use_container_width=True,
            help="Scores transcripts via Claude (real API spend) + submits Alpaca paper orders",
            disabled=snap.trading_disabled,
        )
        if clicked:
            from casino.jobs.earnings_daily import (  # noqa: PLC0415
                run_earnings_daily,
            )

            with st.spinner("scoring transcripts + sizing orders... ~30-60s"):
                try:
                    earn = run_earnings_daily()
                    if earn.skipped_reason:
                        st.session_state["last_action"] = (
                            f"earnings_daily skipped · {earn.skipped_reason}"
                        )
                    else:
                        st.session_state["last_action"] = (
                            f"earnings_daily ok · "
                            f"candidates={earn.n_candidates} · "
                            f"scored={earn.n_scored} · "
                            f"L/S={earn.n_long}/{earn.n_short} · "
                            f"submitted={earn.n_submitted}"
                        )
                    st.session_state["last_action_ok"] = True
                except Exception as e:  # noqa: BLE001
                    st.session_state["last_action"] = f"earnings_daily FAILED: {e}"
                    st.session_state["last_action_ok"] = False
            st.rerun()

    # Kill switch / re-enable.
    with cols[3]:
        if snap.trading_disabled:
            if st.button(
                "⊕ Re-enable",
                key="op_reenable",
                use_container_width=True,
            ):
                from casino.execution.risk import (  # noqa: PLC0415
                    re_enable_trading,
                )

                re_enable_trading()
                st.session_state["last_action"] = "trading flag cleared · system armed"
                st.session_state["last_action_ok"] = True
                st.session_state["confirm_kill"] = False
                st.rerun()
        else:
            if st.button("⊘ Kill switch", key="op_kill", use_container_width=True):
                st.session_state["confirm_kill"] = True
                st.rerun()

    # Confirm kill — only visible while armed.
    with cols[4]:
        if st.session_state["confirm_kill"] and not snap.trading_disabled:
            confirm = st.button(
                "⊘ Confirm fire",
                key="op_kill_fire",
                type="primary",
                use_container_width=True,
            )
            if confirm:
                from casino.execution.risk import (  # noqa: PLC0415
                    flatten_and_disable,
                )

                with st.spinner("flattening positions..."):
                    try:
                        ks = flatten_and_disable(reason="dashboard manual kill")
                        st.session_state["last_action"] = (
                            f"KILL ENGAGED · cancelled={ks.cancelled_orders} · "
                            f"closed={ks.closed_positions} · flag_set={ks.flag_set}"
                        )
                        st.session_state["last_action_ok"] = True
                    except Exception as e:  # noqa: BLE001
                        st.session_state["last_action"] = f"kill switch FAILED: {e}"
                        st.session_state["last_action_ok"] = False
                st.session_state["confirm_kill"] = False
                st.rerun()

    # Last action log.
    with cols[5]:
        msg = st.session_state.get("last_action")
        if msg:
            cls = "v-ok" if st.session_state.get("last_action_ok") else "v-bad"
            glyph = "✓" if st.session_state.get("last_action_ok") else "✗"
            st.markdown(
                f"<div class='action-log'><span class='{cls}'>{glyph}</span> {msg}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div></div>", unsafe_allow_html=True)


def _metric_strip_html(snap: DashboardSnapshot) -> str:
    """Five metric tiles across the top: NAV / Today / MTD / YTD / DD."""
    today_cls = _pl_class(snap.pnl.today)
    mtd_cls = _pl_class(snap.pnl.mtd)
    ytd_cls = _pl_class(snap.pnl.ytd)
    dd_pct = float(snap.pnl.drawdown)
    dd_cls = "neg" if dd_pct >= 0.10 else ("warn" if dd_pct >= 0.05 else "")
    sharpe = f"{snap.rolling_sharpe_60d:+.2f}" if snap.rolling_sharpe_60d is not None else "—"
    return f"""
    <div class='metric-strip'>
      <div class='metric-cell lg'>
        <span class='lbl'>Equity / NAV</span>
        <div class='num'>{_fmt_money(snap.pnl.equity_today, decimals=2)}</div>
        <div class='sub'>HWM <span class='lo'>·</span> {_fmt_money(snap.pnl.high_water_mark, decimals=0)}</div>
      </div>
      <div class='metric-cell'>
        <span class='lbl'>Today</span>
        <div class='num {today_cls}'>{_fmt_money(snap.pnl.today, signed=True)}</div>
        <div class='sub'>session P&amp;L</div>
      </div>
      <div class='metric-cell'>
        <span class='lbl'>Month to date</span>
        <div class='num {mtd_cls}'>{_fmt_money(snap.pnl.mtd, signed=True)}</div>
        <div class='sub'>since {datetime.now(tz=UTC).strftime("%b 01")}</div>
      </div>
      <div class='metric-cell'>
        <span class='lbl'>Year to date</span>
        <div class='num {ytd_cls}'>{_fmt_money(snap.pnl.ytd, signed=True)}</div>
        <div class='sub'>rolling 60d Sharpe <span class='lo'>·</span> {sharpe}</div>
      </div>
      <div class='metric-cell'>
        <span class='lbl'>Drawdown</span>
        <div class='num {dd_cls}'>{_fmt_pct(snap.pnl.drawdown, decimals=2)}</div>
        <div class='sub'>from high water mark</div>
      </div>
    </div>
    """


def _positions_table_html(rows: Sequence[PositionRow]) -> str:
    """Render the positions table as raw HTML for full styling control."""
    if not rows:
        body = "<tr><td colspan='8' class='empty'>— no open positions —</td></tr>"
    else:
        trs: list[str] = []
        for p in rows:
            side_cls = "side-long" if p.side == "long" else "side-short"
            arrow = "▲" if p.side == "long" else "▼"
            mark = _fmt_money(p.market_price, decimals=2) if p.market_price is not None else "—"
            mv = _fmt_money(p.market_value, decimals=0) if p.market_value is not None else "—"
            if p.unrealized_pl is not None:
                pl_cls = _pl_class(p.unrealized_pl)
                pl_str = _fmt_money(p.unrealized_pl, signed=True, decimals=2)
            else:
                pl_cls = "faint"
                pl_str = "—"
            broker_qty = p.broker_qty if p.broker_qty is not None else "—"
            sync_glyph = (
                "<span class='sync-ok'>●</span>" if p.in_sync else "<span class='sync-bad'>◯</span>"
            )
            trs.append(
                f"<tr>"
                f"<td class='sym'>{p.symbol}</td>"
                f"<td class='c {side_cls}'>{arrow} {p.side.upper()}</td>"
                f"<td class='r'>{p.book_qty}</td>"
                f"<td class='r dim'>{broker_qty}</td>"
                f"<td class='r dim'>{_fmt_money(p.avg_entry_price, decimals=2)}</td>"
                f"<td class='r'>{mark}</td>"
                f"<td class='r dim'>{mv}</td>"
                f"<td class='r {pl_cls}'>{pl_str}</td>"
                f"<td class='c'>{sync_glyph}</td>"
                f"</tr>"
            )
        body = "".join(trs)
    return f"""
    <table class='data'>
      <thead><tr>
        <th>Symbol</th>
        <th class='c'>Side</th>
        <th class='r'>Book qty</th>
        <th class='r'>Brkr qty</th>
        <th class='r'>Avg entry</th>
        <th class='r'>Mark</th>
        <th class='r'>Mkt val</th>
        <th class='r'>Unreal P&amp;L</th>
        <th class='c'>Sync</th>
      </tr></thead>
      <tbody>{body}</tbody>
    </table>
    """


def _calls_table_html(rows: Sequence[LLMCallRow]) -> str:
    """Render the LLM ledger as raw HTML."""
    if not rows:
        body = "<tr><td colspan='7' class='empty'>— no LLM calls yet —</td></tr>"
    else:
        trs: list[str] = []
        for c in rows[:50]:
            ok = (
                "<span class='sync-ok'>●</span>" if c.success else "<span class='sync-bad'>◯</span>"
            )
            mode_cls = "warn" if c.mode == "backtest" else "dim"
            short_model = c.model.replace("claude-", "").replace("-20251001", "")
            cost_str = f"${c.cost_usd:.4f}"
            cache_pct = (
                int(c.cached_read_tokens / c.input_tokens * 100) if c.input_tokens > 0 else 0
            )
            trs.append(
                f"<tr>"
                f"<td class='c'>{ok}</td>"
                f"<td class='dim'>{c.timestamp_local[5:19]}</td>"
                f"<td class='sym'>{short_model}</td>"
                f"<td class='c {mode_cls}'>{c.mode}</td>"
                f"<td class='r'>{cost_str}</td>"
                f"<td class='r dim'>{c.input_tokens:,}<span class='leader'>/</span>"
                f"{cache_pct}%</td>"
                f"<td class='r dim'>{c.output_tokens:,}</td>"
                f"<td class='r faint'>{c.latency_ms} ms</td>"
                f"</tr>"
            )
        body = "".join(trs)
    return f"""
    <table class='data'>
      <thead><tr>
        <th class='c'>OK</th>
        <th>Time NY</th>
        <th>Model</th>
        <th class='c'>Mode</th>
        <th class='r'>Cost</th>
        <th class='r'>In tok / cache</th>
        <th class='r'>Out tok</th>
        <th class='r'>Latency</th>
      </tr></thead>
      <tbody>{body}</tbody>
    </table>
    """


def _equity_drawdown_series(
    history: Sequence[book.DailyPnLRow],
) -> tuple[list[str], list[float], list[float]]:
    """Build (dates_asc, equity, drawdown_pct) series from the daily P&L log.

    `history` is most-recent first per `fetch_daily_pnl`; we reverse to render.
    """
    rev = list(reversed(history))
    dates = [r.date for r in rev]
    eq = [float(r.equity_close) for r in rev]
    hwm = 0.0
    dd: list[float] = []
    for v in eq:
        if v > hwm:
            hwm = v
        if hwm > 0:
            dd.append((hwm - v) / hwm * 100.0)
        else:
            dd.append(0.0)
    return dates, eq, dd


def _render_equity_chart(history: Sequence[book.DailyPnLRow]) -> Any:
    """Build a Plotly figure for the equity + drawdown chart."""
    import plotly.graph_objects as go  # noqa: PLC0415 — render-only dep

    dates, eq, dd = _equity_drawdown_series(history)
    fig = go.Figure()
    if not dates:
        # Empty — keep the frame so the page composition holds.
        fig.add_annotation(
            text="— NO HISTORY —",
            font=dict(family="IBM Plex Sans Condensed", size=14, color="#555149"),
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=eq,
                mode="lines",
                name="Equity",
                line=dict(color="#e8a33d", width=1.6),
                hovertemplate="%{x}<br>%{y:$,.0f}<extra></extra>",
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=[-d for d in dd],
                mode="lines",
                name="Drawdown",
                line=dict(color="#f85149", width=1.0, dash="dot"),
                hovertemplate="%{x}<br>−%{customdata:.2f}%<extra></extra>",
                customdata=dd,
                yaxis="y2",
                fill="tozeroy",
                fillcolor="rgba(248,81,73,0.08)",
            )
        )
    fig.update_layout(
        height=300,
        margin=dict(t=24, b=36, l=64, r=64),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        showlegend=False,
        font=dict(family="JetBrains Mono", size=10, color="#888580"),
        hoverlabel=dict(
            bgcolor="#0a0a0a",
            bordercolor="#2a2a2a",
            font=dict(family="JetBrains Mono", size=11, color="#e8e6e3"),
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#2a2a2a",
            ticks="outside",
            tickcolor="#2a2a2a",
            tickfont=dict(size=9, color="#555149"),
        ),
        yaxis=dict(
            title=dict(
                text="EQUITY",
                font=dict(
                    family="IBM Plex Sans Condensed",
                    size=9,
                    color="#555149",
                ),
            ),
            showgrid=True,
            gridcolor="#1a1a1a",
            linecolor="#2a2a2a",
            tickfont=dict(size=9, color="#888580"),
            tickprefix="$",
            tickformat=",.0f",
            zeroline=False,
        ),
        yaxis2=dict(
            title=dict(
                text="DRAWDOWN",
                font=dict(
                    family="IBM Plex Sans Condensed",
                    size=9,
                    color="#555149",
                ),
            ),
            overlaying="y",
            side="right",
            showgrid=False,
            linecolor="#2a2a2a",
            tickfont=dict(size=9, color="#555149"),
            ticksuffix="%",
            zeroline=False,
            range=[-max(dd) * 1.4 if dd else -1, 0],
        ),
    )
    return fig


def _spend_bar_html(monthly: Decimal, budget: Decimal) -> str:
    pct_f = float(monthly / budget * 100) if budget > 0 else 0.0
    pct_clamped = min(pct_f, 100.0)
    cls = "fill"
    if pct_f >= 100:
        cls = "fill over"
    elif pct_f >= 75:
        cls = "fill warn"
    return f"""
    <div class='spend-row'>
      <span><span class='label' style='font-family:"IBM Plex Sans Condensed";
       font-weight:600;font-size:9px;letter-spacing:0.24em;text-transform:uppercase;
       color:var(--text-faint);margin-right:1em;'>MTD spend</span>
       <span class='pct'>{_fmt_money(monthly)}</span>
       <span style='color:var(--text-faint);'> / {_fmt_money(budget)}</span></span>
      <span class='spend-bar'>
        <span class='{cls}' style='width:{pct_clamped}%'></span>
      </span>
      <span class='pct'>{pct_f:.1f}%</span>
    </div>
    """


def _status_bar_html(snap: DashboardSnapshot, *, broker_connected: bool) -> str:
    """Bottom fixed status bar."""
    n_calls = len(snap.recent_calls)
    n_pos = len(snap.positions)
    out_of_sync = sum(1 for p in snap.positions if not p.in_sync)
    sync_cls = "v-ok" if out_of_sync == 0 else "v-bad"
    sync_label = "all reconciled" if out_of_sync == 0 else f"{out_of_sync} drift"
    today = sum(
        float(s) for _, s in snap.daily_spend if _ == datetime.now(tz=UTC).strftime("%Y-%m-%d")
    )
    today_spend_cls = "v-ok" if today < 5.0 else ("v-warn" if today < 10.0 else "v-bad")
    dot = "dot-live" if broker_connected and not snap.trading_disabled else "dot-live bad"
    return f"""
    <div class='status'>
      <div>
        <span class='item'><span class='{dot}'></span>
          <span class='lbl'>System</span>
          <span class='v'>{"online" if broker_connected and not snap.trading_disabled else "degraded"}</span>
        </span>
        <span class='item'><span class='lbl'>Positions</span><span class='v'>{n_pos}</span></span>
        <span class='item'><span class='lbl'>Reconcile</span><span class='{sync_cls}'>{sync_label}</span></span>
        <span class='item'><span class='lbl'>LLM calls (50)</span><span class='v'>{n_calls}</span></span>
        <span class='item'><span class='lbl'>Today LLM</span>
          <span class='{today_spend_cls}'>${today:.2f}</span></span>
      </div>
      <div>
        <span class='item'><span class='lbl'>Build</span><span class='v'>casino v0.1.0 / opus 4.7</span></span>
      </div>
    </div>
    """


def render(snapshot: DashboardSnapshot | None = None) -> None:  # pragma: no cover
    """Streamlit render layer.

    Editorial-financial dark UI: black canvas, single warm amber accent
    reserved for the brand mark, Fraunces serif for headline numbers,
    JetBrains Mono for tabular data, IBM Plex Sans Condensed for caps
    section labels. Only `render` and the `_*_html` helpers above import
    streamlit/plotly; the data-prep layer remains import-free of UI deps.
    """
    import streamlit as st  # noqa: PLC0415 — render-only

    st.set_page_config(
        page_title="casino · trading desk",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    snap = snapshot if snapshot is not None else build_snapshot(broker=None)
    broker_connected = False  # build_snapshot was called with broker=None

    st.markdown(_CSS.replace("%FONT_IMPORTS%", _FONT_IMPORTS), unsafe_allow_html=True)
    st.markdown(_ticker_tape_html(snap.positions), unsafe_allow_html=True)
    st.markdown(
        _brand_header_html(snap, broker_connected=broker_connected),
        unsafe_allow_html=True,
    )

    if snap.trading_disabled:
        st.markdown(_kill_banner_html(), unsafe_allow_html=True)

    # Operations action bar — buttons for jobs the operator otherwise runs from a shell.
    _render_ops_bar(snap)

    st.markdown(_metric_strip_html(snap), unsafe_allow_html=True)

    # --- Equity + drawdown chart
    st.markdown(
        "<div class='section'>"
        "<h2><span class='num-tag'>01</span>Equity curve · drawdown ribbon</h2>"
        "<span class='meta'>daily close · all-time</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='chart-frame'>", unsafe_allow_html=True)
    history = book.fetch_daily_pnl(limit=2000)
    fig = _render_equity_chart(history)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Positions
    st.markdown(
        "<div class='section'>"
        "<h2><span class='num-tag'>02</span>Open positions · book vs broker</h2>"
        f"<span class='meta'>{len(snap.positions)} open</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(_positions_table_html(snap.positions), unsafe_allow_html=True)

    # --- LLM ledger
    st.markdown(
        "<div class='section'>"
        "<h2><span class='num-tag'>03</span>LLM ledger · last 50 calls</h2>"
        f"<span class='meta'>{len(snap.recent_calls)} of 50</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(_calls_table_html(snap.recent_calls), unsafe_allow_html=True)

    # --- Spend
    st.markdown(
        "<div class='section'>"
        "<h2><span class='num-tag'>04</span>API spend · monthly budget</h2>"
        f"<span class='meta'>budget {_fmt_money(snap.monthly_budget)}/mo</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        _spend_bar_html(snap.monthly_spend, snap.monthly_budget),
        unsafe_allow_html=True,
    )

    # --- Daily spend mini table
    daily_rows = (
        "".join(
            f"<tr><td class='dim'>{d}</td><td class='r'>{_fmt_money(s, decimals=4)}</td></tr>"
            for d, s in snap.daily_spend[:14]
        )
        or "<tr><td colspan='2' class='empty'>— no spend recorded —</td></tr>"
    )
    st.markdown(
        f"""
        <table class='data' style='margin-top:0.6rem;max-width:480px'>
          <thead><tr>
            <th>Date</th>
            <th class='r'>Spend</th>
          </tr></thead>
          <tbody>{daily_rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        _status_bar_html(snap, broker_connected=broker_connected),
        unsafe_allow_html=True,
    )


# Streamlit entry point: `streamlit run casino/monitoring/dashboard.py`
if __name__ == "__main__":  # pragma: no cover
    render()


# Re-export so static analyzers see `reconcile` is imported (used by tests).
_ = reconcile
