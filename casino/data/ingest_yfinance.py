"""yfinance ingestion: earnings calendar (actual + consensus EPS) → DuckDB.

Tiingo's free tier does not expose per-quarter EPS actuals/estimates, so the
`earnings` table is populated from yfinance instead. yfinance ships with the
project (declared in pyproject.toml) and reads the same Yahoo Finance feed
that powers most retail earnings calendars.

CLI:
    uv run python -m casino.data.ingest_yfinance --ticker AAPL
    uv run python -m casino.data.ingest_yfinance --tickers AAPL,MSFT,GOOGL
    uv run python -m casino.data.ingest_yfinance --tickers-file universe.txt

The fetcher returns rows shaped for `casino.data.store.upsert_earnings`:

    {
        "ticker":         "AAPL",
        "report_date":    datetime(2024, 10, 31, 20, 30, tzinfo=UTC),
        "period_end":     None,           # yfinance does not expose this here
        "actual_eps":     1.64,           # may be None for future dates
        "consensus_eps":  1.60,           # may be None for very old rows
        "revenue":        None,           # not provided by earnings_dates
        "source":         "yfinance",
    }
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from casino.data import store

_SOURCE = "yfinance"
_RATE_LIMIT_SEC = 0.5
_DEFAULT_OHLCV_START = "2018-01-01"


class _TickerLike(Protocol):
    """Minimal protocol over yfinance.Ticker so tests can inject fakes."""

    @property
    def earnings_dates(self) -> Any: ...  # pandas DataFrame at runtime

    def history(self, **kwargs: Any) -> Any: ...  # pandas DataFrame at runtime


def _default_factory(symbol: str) -> _TickerLike:
    """Lazy import yfinance so test environments without it can still load this module."""
    import yfinance as yf

    handle: _TickerLike = yf.Ticker(symbol)
    return handle


def _to_utc(value: Any) -> datetime | None:
    """Coerce a pandas Timestamp / datetime / NaT into a tz-aware UTC datetime."""
    if value is None:
        return None
    # pandas Timestamps expose .to_pydatetime; bare datetimes do not.
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        try:
            value = to_py()
        except Exception:  # noqa: BLE001 — defensive: pandas NaT raises here
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    out: datetime = value.astimezone(UTC)
    return out


def _to_float(value: Any) -> float | None:
    """Coerce numpy/pandas numerics into a plain float, mapping NaN to None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _normalize_row(ticker: str, ts: Any, row: Any) -> dict[str, object] | None:
    """Translate one yfinance earnings_dates row into the store-shaped dict."""
    report_date = _to_utc(ts)
    if report_date is None:
        return None

    # yfinance column names have shifted between versions: "EPS Estimate" /
    # "Reported EPS" are the stable forms. Fall back to lowercase variants.
    def _col(*names: str) -> Any:
        for n in names:
            if hasattr(row, "get"):
                v = row.get(n)
            else:
                v = getattr(row, n, None)
            if v is not None:
                return v
        return None

    actual = _to_float(_col("Reported EPS", "reportedEps", "EPS Actual"))
    consensus = _to_float(_col("EPS Estimate", "epsEstimate"))

    return {
        "ticker": ticker.upper(),
        "report_date": report_date,
        "period_end": None,
        "actual_eps": actual,
        "consensus_eps": consensus,
        "revenue": None,
        "source": _SOURCE,
    }


def fetch_earnings(
    ticker: str,
    *,
    factory: Any = _default_factory,
) -> list[dict[str, object]]:
    """Fetch earnings_dates for one ticker. Empty list on no data or fetch error."""
    try:
        handle = factory(ticker)
    except Exception as exc:  # noqa: BLE001 — yfinance raises bare RuntimeError
        logger.warning("yfinance ticker construction failed for {}: {}", ticker, exc)
        return []

    try:
        df = handle.earnings_dates
    except Exception as exc:  # noqa: BLE001 — Yahoo intermittently 404s on tickers
        logger.warning("yfinance earnings_dates failed for {}: {}", ticker, exc)
        return []

    if df is None or len(df) == 0:
        return []

    rows: list[dict[str, object]] = []
    # iterrows yields (index, Series); index is a pandas Timestamp.
    for ts, srow in df.iterrows():
        norm = _normalize_row(ticker, ts, srow)
        if norm is not None:
            rows.append(norm)
    return rows


def upsert_earnings(
    rows: list[dict[str, object]],
    *,
    db_path: Path | None = None,
) -> int:
    """Thin pass-through to store.upsert_earnings (kept for symmetry with other ingesters)."""
    return store.upsert_earnings(rows, db_path=db_path)


def ingest_tickers(
    tickers: list[str],
    *,
    factory: Any = _default_factory,
    db_path: Path | None = None,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
) -> dict[str, int]:
    """Fetch earnings for each ticker and upsert. Returns {ticker: row_count}."""
    store.create_schema(db_path=db_path)
    counts: dict[str, int] = {}
    for i, t in enumerate(tickers):
        if i > 0 and rate_limit_sec > 0:
            time.sleep(rate_limit_sec)
        rows = fetch_earnings(t, factory=factory)
        n = upsert_earnings(rows, db_path=db_path) if rows else 0
        counts[t.upper()] = n
        logger.info("yfinance earnings {}: {} rows", t.upper(), n)
    return counts


# ----------------------------------------------------------------- ohlcv path
def _normalize_ohlcv_row(ticker: str, ts: Any, row: Any) -> dict[str, object] | None:
    """Translate one yfinance .history() row into a store.upsert_ohlcv dict."""
    bar_ts = _to_utc(ts)
    if bar_ts is None:
        return None

    def _col(*names: str) -> Any:
        for n in names:
            if hasattr(row, "get"):
                v = row.get(n)
            else:
                v = getattr(row, n, None)
            if v is not None:
                return v
        return None

    open_ = _to_float(_col("Open", "open"))
    high = _to_float(_col("High", "high"))
    low = _to_float(_col("Low", "low"))
    close = _to_float(_col("Close", "close"))
    volume_raw = _col("Volume", "volume")
    adj_close = _to_float(_col("Adj Close", "adjClose", "adj_close"))

    if close is None:
        # A bar without a close is meaningless and breaks downstream PEAD math.
        return None

    try:
        volume = int(volume_raw) if volume_raw is not None else 0
    except (TypeError, ValueError):
        volume = 0

    return {
        "ticker": ticker.upper(),
        "ts": bar_ts,
        "open": open_ if open_ is not None else close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
        "close": close,
        "volume": volume,
        "adj_close": adj_close if adj_close is not None else close,
    }


def fetch_ohlcv(
    ticker: str,
    *,
    start: str | date | None = _DEFAULT_OHLCV_START,
    end: str | date | None = None,
    factory: Any = _default_factory,
) -> list[dict[str, object]]:
    """Fetch daily OHLCV bars. Empty list on no data or fetch error."""
    try:
        handle = factory(ticker)
    except Exception as exc:  # noqa: BLE001 — yfinance raises bare RuntimeError
        logger.warning("yfinance ticker construction failed for {}: {}", ticker, exc)
        return []

    kwargs: dict[str, Any] = {"auto_adjust": False, "actions": False}
    if start is not None:
        kwargs["start"] = start
    if end is not None:
        kwargs["end"] = end

    try:
        df = handle.history(**kwargs)
    except Exception as exc:  # noqa: BLE001 — Yahoo intermittently 404s on tickers
        logger.warning("yfinance history failed for {}: {}", ticker, exc)
        return []

    if df is None or len(df) == 0:
        return []

    rows: list[dict[str, object]] = []
    for ts, srow in df.iterrows():
        norm = _normalize_ohlcv_row(ticker, ts, srow)
        if norm is not None:
            rows.append(norm)
    return rows


def ingest_ohlcv(
    tickers: list[str],
    *,
    start: str | date | None = _DEFAULT_OHLCV_START,
    end: str | date | None = None,
    factory: Any = _default_factory,
    db_path: Path | None = None,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
) -> dict[str, int]:
    """Fetch OHLCV for each ticker and upsert into the ohlcv table."""
    store.create_schema(db_path=db_path)
    counts: dict[str, int] = {}
    for i, t in enumerate(tickers):
        if i > 0 and rate_limit_sec > 0:
            time.sleep(rate_limit_sec)
        rows = fetch_ohlcv(t, start=start, end=end, factory=factory)
        n = store.upsert_ohlcv(rows, db_path=db_path) if rows else 0
        counts[t.upper()] = n
        logger.info("yfinance ohlcv {}: {} bars", t.upper(), n)
    return counts


# ============================================================================ CLI
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="casino.data.ingest_yfinance",
        description="Ingest earnings and/or OHLCV from yfinance into DuckDB.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ticker", type=str, help="Single ticker (e.g. AAPL)")
    g.add_argument("--tickers", type=str, help="Comma-separated tickers (e.g. AAPL,MSFT)")
    g.add_argument(
        "--tickers-file",
        type=str,
        help="Path to a newline-delimited tickers file (blank lines and # comments OK)",
    )
    p.add_argument(
        "--rate-limit-sec",
        type=float,
        default=_RATE_LIMIT_SEC,
        help="Seconds to sleep between tickers (default 0.5)",
    )
    p.add_argument(
        "--mode",
        choices=("earnings", "ohlcv", "both"),
        default="earnings",
        help="What to ingest: earnings (default), ohlcv, or both.",
    )
    p.add_argument(
        "--ohlcv-start",
        type=str,
        default=_DEFAULT_OHLCV_START,
        help=f"OHLCV start date YYYY-MM-DD (default {_DEFAULT_OHLCV_START}).",
    )
    p.add_argument(
        "--ohlcv-end",
        type=str,
        default=None,
        help="OHLCV end date YYYY-MM-DD (default: today).",
    )
    return p


def _read_tickers_file(path: str) -> list[str]:
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.upper())
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = _read_tickers_file(args.tickers_file)

    if not tickers:
        logger.error("no tickers resolved from arguments")
        return 2

    summary: dict[str, int] = {}
    if args.mode in ("earnings", "both"):
        e_counts = ingest_tickers(tickers, rate_limit_sec=args.rate_limit_sec)
        summary["earnings_rows"] = sum(e_counts.values())
    if args.mode in ("ohlcv", "both"):
        o_counts = ingest_ohlcv(
            tickers,
            start=args.ohlcv_start,
            end=args.ohlcv_end,
            rate_limit_sec=args.rate_limit_sec,
        )
        summary["ohlcv_bars"] = sum(o_counts.values())
    logger.info("yfinance ingest summary: {} tickers, {}", len(tickers), summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
