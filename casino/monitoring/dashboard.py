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


def render(snapshot: DashboardSnapshot | None = None) -> None:  # pragma: no cover
    """Streamlit render layer. Pulled from `build_snapshot` if not supplied."""
    import streamlit as st  # noqa: PLC0415 — only imported when running

    st.set_page_config(page_title="casino dashboard", layout="wide")
    st.title("casino — paper trading dashboard")

    snap = snapshot if snapshot is not None else build_snapshot(broker=None)

    if snap.trading_disabled:
        st.error("⚠ Trading is DISABLED (kill switch flag is set).")

    cols = st.columns(4)
    cols[0].metric("Today P&L", f"${snap.pnl.today}")
    cols[1].metric("MTD P&L", f"${snap.pnl.mtd}")
    cols[2].metric("YTD P&L", f"${snap.pnl.ytd}")
    cols[3].metric("Drawdown", f"{snap.pnl.drawdown:.2%}")

    st.subheader("Open positions vs broker")
    st.dataframe(
        [
            {
                "Symbol": p.symbol,
                "Side": p.side,
                "Book qty": p.book_qty,
                "Broker qty": p.broker_qty,
                "Avg entry": str(p.avg_entry_price),
                "Mark": str(p.market_price) if p.market_price is not None else "",
                "MV": str(p.market_value) if p.market_value is not None else "",
                "Unrealized P&L": str(p.unrealized_pl) if p.unrealized_pl is not None else "",
                "In sync": "✓" if p.in_sync else "✗",
            }
            for p in snap.positions
        ]
    )

    st.subheader("Recent LLM calls")
    st.dataframe(
        [
            {
                "Time (NY)": c.timestamp_local,
                "Model": c.model,
                "Mode": c.mode,
                "Cost ($)": f"{c.cost_usd:.4f}",
                "In tok": c.input_tokens,
                "Cache-read tok": c.cached_read_tokens,
                "Out tok": c.output_tokens,
                "Latency (ms)": c.latency_ms,
                "OK": "✓" if c.success else "✗",
                "Schema": c.schema_name or "",
            }
            for c in snap.recent_calls
        ]
    )

    st.subheader("LLM spend")
    cols = st.columns(2)
    cols[0].metric(
        "Month-to-date spend",
        f"${snap.monthly_spend}",
        f"of ${snap.monthly_budget} budget",
    )
    if snap.rolling_sharpe_60d is not None:
        cols[1].metric("Rolling 60-day Sharpe", f"{snap.rolling_sharpe_60d:.2f}")
    else:
        cols[1].metric("Rolling 60-day Sharpe", "n/a")

    st.subheader("Daily spend")
    st.dataframe([{"Date": d, "Spend ($)": float(s)} for d, s in snap.daily_spend])


# Streamlit entry point: `streamlit run casino/monitoring/dashboard.py`
if __name__ == "__main__":  # pragma: no cover
    render()


# Re-export so static analyzers see `reconcile` is imported (used by tests).
_ = reconcile
