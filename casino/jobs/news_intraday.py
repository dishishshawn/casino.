"""Every-15-minute cron entry during market hours.

Classifies fresh news headlines via Haiku and stores the results so the
intraday strategy (later phase) can consume them. PRD §6 / §10:

* Always live mode; no anonymization on entities.
* Cap daily LLM spend at a configurable per-day USD budget (default $5,
  matching the alert threshold from PRD §10). When the cap is hit, the
  job logs a warning, fires a budget alert, and exits without making
  more calls.

The job does NOT submit orders in v1 — it only generates the headline
sentiment/materiality table. That table feeds future signals.

Run: ``uv run python -m casino.jobs.news_intraday``
Cron: every 15 minutes during US equity market hours (RUNBOOK).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from loguru import logger

from casino.config import get_config
from casino.data import store
from casino.execution.alpaca_broker import AlpacaBroker, build_default_broker
from casino.llm.client import LLMClient
from casino.llm.prompts.headline_class import classify_headline
from casino.monitoring import alerts

# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class HeadlineRow:
    id: str
    ticker: str | None
    headline: str
    published_at: datetime


@dataclass(frozen=True)
class JobResult:
    as_of: datetime
    n_fresh_headlines: int
    n_classified: int
    n_skipped: int
    cost_usd: Decimal
    budget_usd: Decimal
    budget_breached: bool
    market_open: bool
    skipped_reason: str | None


# ---------------------------------------------------------------------------- storage


_HEADLINE_CLASS_SCHEMA = """
CREATE TABLE IF NOT EXISTS headline_classifications (
    headline_id      TEXT    NOT NULL,
    ticker           TEXT,
    classified_at    TEXT    NOT NULL,
    sentiment        REAL    NOT NULL,
    is_material      INTEGER NOT NULL,
    relevance        TEXT    NOT NULL,
    rationale        TEXT,
    model            TEXT    NOT NULL,
    cost_usd         REAL    NOT NULL,
    PRIMARY KEY (headline_id)
);
CREATE INDEX IF NOT EXISTS idx_headline_class_ticker ON headline_classifications(ticker);
CREATE INDEX IF NOT EXISTS idx_headline_class_at     ON headline_classifications(classified_at);
"""


def _resolve_state_path(db_path: Path | None) -> Path:
    return db_path if db_path is not None else get_config().state_sqlite_path


@contextmanager
def _state_conn(db_path: Path | None) -> Iterator[sqlite3.Connection]:
    target = _resolve_state_path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    finally:
        conn.close()


def _ensure_schema(db_path: Path | None) -> None:
    with _state_conn(db_path) as conn:
        conn.executescript(_HEADLINE_CLASS_SCHEMA)


def already_classified_ids(
    candidate_ids: Sequence[str],
    *,
    db_path: Path | None = None,
) -> set[str]:
    """Return the subset of `candidate_ids` already present in the classification table."""
    if not candidate_ids:
        return set()
    _ensure_schema(db_path)
    placeholders = ",".join(["?"] * len(candidate_ids))
    sql = f"SELECT headline_id FROM headline_classifications WHERE headline_id IN ({placeholders})"
    with _state_conn(db_path) as conn:
        rows = conn.execute(sql, list(candidate_ids)).fetchall()
    return {str(r[0]) for r in rows}


def write_classification(
    *,
    headline_id: str,
    ticker: str | None,
    sentiment: float,
    is_material: bool,
    relevance: str,
    rationale: str | None,
    model: str,
    cost_usd: float,
    db_path: Path | None = None,
) -> None:
    _ensure_schema(db_path)
    sql = """
        INSERT OR REPLACE INTO headline_classifications
            (headline_id, ticker, classified_at, sentiment, is_material,
             relevance, rationale, model, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _state_conn(db_path) as conn:
        conn.execute(
            sql,
            (
                headline_id,
                ticker,
                datetime.now(tz=UTC).isoformat(),
                float(sentiment),
                1 if is_material else 0,
                relevance,
                rationale,
                model,
                float(cost_usd),
            ),
        )


def daily_llm_spend(*, day_utc: datetime, db_path: Path | None = None) -> Decimal:
    """Return the day's total LLM spend (across *all* call sites) in USD.

    Reads from the LLM audit log, summing `cost_usd` for rows whose
    `timestamp_utc` is on `day_utc`'s UTC date.
    """
    target = _resolve_state_path(db_path)
    if not target.exists():
        return Decimal("0")
    day_str = day_utc.strftime("%Y-%m-%d")
    sql = """
        SELECT COALESCE(SUM(cost_usd), 0)
        FROM llm_calls
        WHERE substr(timestamp_utc, 1, 10) = ?
    """
    with sqlite3.connect(str(target)) as conn:
        row = conn.execute(sql, (day_str,)).fetchone()
    return Decimal(str(row[0] if row and row[0] is not None else 0.0))


# ---------------------------------------------------------------------------- fetch


def fetch_fresh_headlines(
    *,
    since: datetime,
    db_path: Path | None = None,
) -> list[HeadlineRow]:
    """Return news rows with `published_at >= since`."""
    sql = """
        SELECT id, ticker, headline, published_at
        FROM news
        WHERE published_at >= ?
          AND headline IS NOT NULL
          AND headline <> ''
        ORDER BY published_at ASC
    """
    with store.get_duckdb_conn(db_path, read_only=False) as conn:
        rows = conn.execute(sql, [since]).fetchall()
    out: list[HeadlineRow] = []
    for r in rows:
        ts = r[3]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        out.append(
            HeadlineRow(
                id=str(r[0]),
                ticker=str(r[1]) if r[1] is not None else None,
                headline=str(r[2]),
                published_at=ts,
            )
        )
    return out


# ---------------------------------------------------------------------------- main flow


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def run_news_intraday(
    *,
    client: LLMClient | None = None,
    broker: AlpacaBroker | None = None,
    as_of: datetime | None = None,
    db_path: Path | None = None,
    lookback_minutes: int = 30,
    daily_budget_usd: Decimal = Decimal("5.00"),
    require_market_open: bool = True,
) -> JobResult:
    """One pass of the intraday news classifier.

    `lookback_minutes` is the freshness window — typically 2x the cron
    interval to make sure the job is robust to a missed tick.

    The cost guard reads the audit log's running daily total. We refuse
    to send another call once that total reaches `daily_budget_usd` —
    matching PRD §10's $5 alert threshold by default. Tests inject a
    smaller budget to drive the breach branch.
    """
    as_of = as_of if as_of is not None else _utc_now()
    cfg = get_config()
    if client is None:
        client = LLMClient(mode="live", audit_db_path=cfg.state_sqlite_path)
    if broker is None and require_market_open:
        broker = build_default_broker()

    try:
        # 0) market-hours gate (skipped in tests via require_market_open=False)
        if require_market_open and broker is not None:
            try:
                if not broker.is_market_open():
                    return JobResult(
                        as_of=as_of,
                        n_fresh_headlines=0,
                        n_classified=0,
                        n_skipped=0,
                        cost_usd=Decimal("0"),
                        budget_usd=daily_budget_usd,
                        budget_breached=False,
                        market_open=False,
                        skipped_reason="market closed",
                    )
            except Exception as e:  # noqa: BLE001 — broker outage shouldn't crash cron
                logger.warning("news_intraday: market-clock check failed ({}); proceeding", e)

        # 1) fresh headlines
        since = as_of - timedelta(minutes=lookback_minutes)
        headlines = fetch_fresh_headlines(since=since, db_path=db_path)
        if not headlines:
            return JobResult(
                as_of=as_of,
                n_fresh_headlines=0,
                n_classified=0,
                n_skipped=0,
                cost_usd=Decimal("0"),
                budget_usd=daily_budget_usd,
                budget_breached=False,
                market_open=True,
                skipped_reason="no fresh headlines",
            )
        already = already_classified_ids([h.id for h in headlines], db_path=cfg.state_sqlite_path)
        new = [h for h in headlines if h.id not in already]
        if not new:
            return JobResult(
                as_of=as_of,
                n_fresh_headlines=len(headlines),
                n_classified=0,
                n_skipped=len(already),
                cost_usd=Decimal("0"),
                budget_usd=daily_budget_usd,
                budget_breached=False,
                market_open=True,
                skipped_reason="all headlines already classified",
            )

        # 2) per-pass classification with running budget check
        spend_today = daily_llm_spend(day_utc=as_of, db_path=cfg.state_sqlite_path)
        if spend_today >= daily_budget_usd:
            alerts.alert_llm_spend(
                daily_spend_usd=spend_today,
                threshold_usd=daily_budget_usd,
            )
            return JobResult(
                as_of=as_of,
                n_fresh_headlines=len(headlines),
                n_classified=0,
                n_skipped=len(new),
                cost_usd=Decimal("0"),
                budget_usd=daily_budget_usd,
                budget_breached=True,
                market_open=True,
                skipped_reason="daily LLM budget already exceeded",
            )

        n_classified = 0
        cost_this_pass = Decimal("0")
        for h in new:
            running_total = spend_today + cost_this_pass
            if running_total >= daily_budget_usd:
                logger.warning(
                    "news_intraday: budget cap (${} >= ${}); halting",
                    running_total,
                    daily_budget_usd,
                )
                alerts.alert_llm_spend(
                    daily_spend_usd=running_total,
                    threshold_usd=daily_budget_usd,
                )
                return JobResult(
                    as_of=as_of,
                    n_fresh_headlines=len(headlines),
                    n_classified=n_classified,
                    n_skipped=len(new) - n_classified,
                    cost_usd=cost_this_pass,
                    budget_usd=daily_budget_usd,
                    budget_breached=True,
                    market_open=True,
                    skipped_reason="daily LLM budget exhausted mid-pass",
                )
            try:
                classification = classify_headline(
                    client=client,
                    ticker=h.ticker or "",
                    headline=h.headline,
                )
            except Exception as e:  # noqa: BLE001 — bad headline shouldn't kill the pass
                logger.warning(
                    "news_intraday: classify_headline failed for id={}: {}",
                    h.id,
                    e,
                )
                continue
            # The audit row already contains the cost; we re-read for the
            # running total below. For per-row attribution, query last row.
            last_cost = _last_audit_cost(db_path=cfg.state_sqlite_path)
            cost_this_pass += last_cost
            write_classification(
                headline_id=h.id,
                ticker=h.ticker,
                sentiment=classification.sentiment,
                is_material=classification.is_material,
                relevance=classification.relevance,
                rationale=classification.rationale,
                model=cfg.llm_haiku_model,
                cost_usd=float(last_cost),
            )
            n_classified += 1

        return JobResult(
            as_of=as_of,
            n_fresh_headlines=len(headlines),
            n_classified=n_classified,
            n_skipped=len(headlines) - n_classified - len(already),
            cost_usd=cost_this_pass,
            budget_usd=daily_budget_usd,
            budget_breached=False,
            market_open=True,
            skipped_reason=None,
        )
    except Exception as e:  # noqa: BLE001
        alerts.alert_unhandled_exception(
            job="news_intraday",
            exc_type=type(e).__name__,
            detail=str(e),
        )
        logger.exception("news_intraday: unhandled exception")
        raise


def _last_audit_cost(*, db_path: Path) -> Decimal:
    """Read the most recent llm_calls.cost_usd. Returns 0 if missing."""
    if not db_path.exists():
        return Decimal("0")
    sql = "SELECT cost_usd FROM llm_calls ORDER BY id DESC LIMIT 1"
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(sql).fetchone()
    except sqlite3.OperationalError:
        return Decimal("0")
    if row is None or row[0] is None:
        return Decimal("0")
    return Decimal(str(row[0]))


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.jobs.news_intraday",
        description="Classify fresh news headlines (intraday cron).",
    )
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=30,
        help="Headlines newer than this many minutes are considered fresh.",
    )
    parser.add_argument(
        "--budget-usd",
        type=str,
        default="5.00",
        help="Per-day USD spend cap.",
    )
    args = parser.parse_args(argv)
    result = run_news_intraday(
        lookback_minutes=int(args.lookback_minutes),
        daily_budget_usd=Decimal(args.budget_usd),
    )
    logger.info("news_intraday result: {}", result)
    return 0


# Re-export type for callers that want to type-narrow
_ = Any

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
