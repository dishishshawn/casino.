"""Simulated broker — drop-in shadow for ``AlpacaBroker``.

Branch C amendment 2026-05-08 (Option B parallel experiment): the live
``tsmom_runner`` is hitting Alpaca paper for vanilla TSMOM. To compare
that head-to-head against the regime-filtered variant *without* paying
for a second paper account, the shadow path uses this in-process
simulator. It:

* Talks to the same DuckDB OHLCV store via ``casino.data.store`` (no
  separate market data path — point-in-time correctness flows through
  one seam, per CLAUDE.md §4.1).
* Persists state to the *same* ``state.sqlite`` as the live bot, but in
  separate tables prefixed ``sim_*``. All rows are keyed by ``run_id``
  so multiple sim runs cannot collide and the live tables are never
  touched by sim writes.
* Exposes the subset of the ``AlpacaBroker`` surface that
  ``tsmom_shadow_runner`` actually needs: ``get_account``,
  ``get_positions``, ``submit_bracket_order``, ``close_position``,
  ``cancel_all``, ``is_market_open``. Returned types match (``BrokerOrder``,
  ``BrokerPosition``, ``BrokerAccount``) so downstream reconcile/risk
  code can treat the sim and live brokers identically.

Fill model:

* Bracket orders submitted on day T fill at the day T+1 *open* (the next
  bar in DuckDB ≥ the submission's ``as_of`` cursor). This is the
  conservative interpretation of the live runner's monthly cadence — we
  never fill on the same bar we computed the signal from.
* Stops are evaluated bar-by-bar inside ``mark_to_market``. If a held
  long position's bar low ≤ ``stop_price`` (or short's high ≥ stop), the
  position is force-closed at exactly ``stop_price``. This matches a
  broker-side stop-loss order's behavior (the live broker would pick
  off the stop; the sim doesn't model worst-case slippage past the stop
  because the live runner doesn't either).
* No partial fills, no commissions inside the sim (the regime panel's
  baseline runs cost-aware backtests; the daily P&L mark-to-market here
  is gross-of-cost, mirroring how Alpaca paper reports unrealized).

NAV tracking:

* ``get_account()`` reports current equity = cash + sum(qty *
  market_price). ``mark_to_market`` is the only state-mutator that
  advances the clock; calling it idempotently for the same date is a
  no-op.

Hard rule from CLAUDE.md: ``data/store.py`` is the only DuckDB path.
The sim broker imports through that module — never opens DuckDB itself.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger

from casino.execution import book
from casino.execution.alpaca_broker import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
)

if TYPE_CHECKING:
    import pandas as pd

OrderSide = Literal["buy", "sell"]

# Default starting NAV mirrors the Alpaca paper account default.
DEFAULT_SIM_START_NAV: Decimal = Decimal("100000")


# ---------------------------------------------------------------------------- schema


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sim_account (
    run_id            TEXT    PRIMARY KEY,
    cash              TEXT    NOT NULL,
    last_clock_utc    TEXT,                  -- ISO-8601, the as-of date last marked
    started_at_utc    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS sim_orders (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    broker_order_id   TEXT    NOT NULL,
    client_order_id   TEXT,
    symbol            TEXT    NOT NULL,
    side              TEXT    NOT NULL,
    qty               INTEGER NOT NULL,
    stop_price        TEXT    NOT NULL,
    submitted_at_utc  TEXT    NOT NULL,
    status            TEXT    NOT NULL,       -- 'pending' | 'filled' | 'cancelled'
    UNIQUE(run_id, broker_order_id)
);
CREATE INDEX IF NOT EXISTS idx_sim_orders_run ON sim_orders(run_id);
CREATE INDEX IF NOT EXISTS idx_sim_orders_status ON sim_orders(status);

CREATE TABLE IF NOT EXISTS sim_fills (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    broker_order_id   TEXT    NOT NULL,
    symbol            TEXT    NOT NULL,
    side              TEXT    NOT NULL,
    qty               INTEGER NOT NULL,
    price             TEXT    NOT NULL,
    filled_at_utc     TEXT    NOT NULL,
    reason            TEXT                    -- 'open' | 'close' | 'stop'
);
CREATE INDEX IF NOT EXISTS idx_sim_fills_run ON sim_fills(run_id);

CREATE TABLE IF NOT EXISTS sim_positions (
    run_id            TEXT    NOT NULL,
    symbol            TEXT    NOT NULL,
    side              TEXT    NOT NULL,
    qty               INTEGER NOT NULL,
    avg_entry_price   TEXT    NOT NULL,
    stop_price        TEXT    NOT NULL,
    last_mark_price   TEXT    NOT NULL,
    opened_at_utc     TEXT    NOT NULL,
    last_update_utc   TEXT    NOT NULL,
    PRIMARY KEY (run_id, symbol)
);

CREATE TABLE IF NOT EXISTS sim_nav_history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    as_of_date        TEXT    NOT NULL,       -- YYYY-MM-DD
    cash              TEXT    NOT NULL,
    positions_mv      TEXT    NOT NULL,
    equity            TEXT    NOT NULL,
    realized_pl_today TEXT    NOT NULL,
    UNIQUE(run_id, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_sim_nav_run ON sim_nav_history(run_id);
"""


def init_schema(db_path: Path | None = None) -> None:
    """Idempotently create the sim_* tables in state.sqlite."""
    book.init_schema(db_path)  # base schema (positions, run_state, daily_pnl)
    with book.get_book_conn(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)


# ---------------------------------------------------------------------------- helpers


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _to_date(d: date | datetime | str) -> date:
    if isinstance(d, datetime):
        return d.astimezone(UTC).date() if d.tzinfo else d.date()
    if isinstance(d, str):
        return date.fromisoformat(d[:10])
    return d


# ---------------------------------------------------------------------------- bar lookup


def _load_bars_for(
    symbol: str,
    *,
    start: date,
    end: date,
    duckdb_path: Path | None = None,
) -> pd.DataFrame:
    """Pull (open, high, low, close) bars for ``symbol`` between two dates.

    Goes through ``casino.data.store`` per CLAUDE.md (single DuckDB seam).
    Returns a DataFrame indexed by date with float OHLC columns. Empty
    DataFrame if no rows.
    """
    import pandas as pd  # noqa: PLC0415

    from casino.data import store  # noqa: PLC0415

    start_ts = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    # End is inclusive: extend by one day so the boundary bar is included.
    end_ts = datetime.combine(end + timedelta(days=2), datetime.min.time(), tzinfo=UTC)
    sql = """
        SELECT ts, open, high, low, COALESCE(adj_close, close) AS close
        FROM ohlcv
        WHERE ticker = ? AND ts BETWEEN ? AND ?
        ORDER BY ts
    """
    with store.get_duckdb_conn(duckdb_path, read_only=True) as conn:
        df = conn.execute(sql, [symbol.upper(), start_ts, end_ts]).df()
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["bar_date"] = df["ts"].dt.date
    return df.set_index("bar_date").sort_index()


# ---------------------------------------------------------------------------- broker


@dataclass
class SimBroker:
    """Drop-in shadow broker with broker-side stops and deterministic fills.

    Methods mirror ``AlpacaBroker`` so ``tsmom_shadow_runner`` can route
    through them with the same call sites as the live runner. The
    differences:

    * ``submit_bracket_order`` records a *pending* order; it does not fill
      until the next ``mark_to_market`` advances the clock past the
      submission date.
    * ``mark_to_market(as_of)`` is the simulator's tick — call once per
      simulated trading day in chronological order. Idempotent: re-marking
      the same date is a no-op.
    """

    run_id: str
    db_path: Path | None = None
    duckdb_path: Path | None = None
    start_nav: Decimal = DEFAULT_SIM_START_NAV
    _initialized: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------ init
    def _ensure_init(self) -> None:
        if self._initialized:
            return
        init_schema(self.db_path)
        with book.get_book_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT cash FROM sim_account WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO sim_account (run_id, cash, last_clock_utc, started_at_utc)
                    VALUES (?, ?, NULL, ?)
                    """,
                    (self.run_id, str(self.start_nav), _utc_now_iso()),
                )
                logger.info(
                    "sim_broker[{}]: opened account, start_nav=${}",
                    self.run_id,
                    self.start_nav,
                )
        self._initialized = True

    # ------------------------------------------------------------------ helpers
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        self._ensure_init()
        with book.get_book_conn(self.db_path) as conn:
            yield conn

    def _get_cash(self) -> Decimal:
        with self._conn() as conn:
            r = conn.execute(
                "SELECT cash FROM sim_account WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
        if r is None:
            return self.start_nav
        return Decimal(str(r[0]))

    def _set_cash(self, cash: Decimal) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sim_account SET cash = ? WHERE run_id = ?",
                (str(cash), self.run_id),
            )

    def _last_clock(self) -> datetime | None:
        with self._conn() as conn:
            r = conn.execute(
                "SELECT last_clock_utc FROM sim_account WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
        if r is None or r[0] is None:
            return None
        return _parse_iso(str(r[0]))

    def _set_last_clock(self, ts: datetime) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sim_account SET last_clock_utc = ? WHERE run_id = ?",
                (ts.isoformat(), self.run_id),
            )

    def _get_positions_raw(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT symbol, side, qty, avg_entry_price, stop_price,
                       last_mark_price, opened_at_utc, last_update_utc
                FROM sim_positions
                WHERE run_id = ?
                ORDER BY symbol
                """,
                (self.run_id,),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "symbol": str(r[0]),
                    "side": str(r[1]),
                    "qty": int(r[2]),
                    "avg_entry_price": Decimal(str(r[3])),
                    "stop_price": Decimal(str(r[4])),
                    "last_mark_price": Decimal(str(r[5])),
                    "opened_at_utc": _parse_iso(str(r[6])),
                    "last_update_utc": _parse_iso(str(r[7])),
                }
            )
        return out

    def _upsert_position(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        avg_entry_price: Decimal,
        stop_price: Decimal,
        last_mark_price: Decimal,
        opened_at_utc: datetime,
    ) -> None:
        sql = """
            INSERT INTO sim_positions (
                run_id, symbol, side, qty, avg_entry_price, stop_price,
                last_mark_price, opened_at_utc, last_update_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, symbol) DO UPDATE SET
                side = excluded.side,
                qty = excluded.qty,
                avg_entry_price = excluded.avg_entry_price,
                stop_price = excluded.stop_price,
                last_mark_price = excluded.last_mark_price,
                last_update_utc = excluded.last_update_utc
        """
        with self._conn() as conn:
            conn.execute(
                sql,
                (
                    self.run_id,
                    symbol.upper(),
                    side,
                    int(qty),
                    str(avg_entry_price),
                    str(stop_price),
                    str(last_mark_price),
                    opened_at_utc.isoformat(),
                    _utc_now_iso(),
                ),
            )

    def _delete_position(self, symbol: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM sim_positions WHERE run_id = ? AND symbol = ?",
                (self.run_id, symbol.upper()),
            )

    # ------------------------------------------------------------------ public
    def get_account(self) -> BrokerAccount:
        """Return current account snapshot (cash + positions market value)."""
        cash = self._get_cash()
        positions = self._get_positions_raw()
        positions_mv = sum(
            (Decimal(p["qty"]) * p["last_mark_price"] for p in positions),
            start=Decimal("0"),
        )
        equity = cash + positions_mv
        return BrokerAccount(
            account_number=f"sim-{self.run_id}",
            status="ACTIVE",
            equity=equity,
            cash=cash,
            buying_power=cash,
            last_equity=equity,
            pattern_day_trader=False,
            trading_blocked=False,
        )

    def get_positions(self) -> list[BrokerPosition]:
        out: list[BrokerPosition] = []
        for p in self._get_positions_raw():
            qty = int(p["qty"])
            mp = p["last_mark_price"]
            mv = Decimal(qty) * mp
            cb = Decimal(qty) * p["avg_entry_price"]
            unr = mv - cb
            side: Literal["long", "short"] = "short" if p["side"] == "short" else "long"
            out.append(
                BrokerPosition(
                    symbol=p["symbol"],
                    qty=abs(qty),
                    side=side,
                    avg_entry_price=p["avg_entry_price"],
                    market_price=mp,
                    market_value=mv,
                    unrealized_pl=unr,
                    cost_basis=cb,
                )
            )
        return out

    def list_positions(self) -> list[BrokerPosition]:
        """Alias for ``get_positions`` (matches the spec's surface name)."""
        return self.get_positions()

    def is_market_open(self) -> bool:  # noqa: D401 — stub
        """Sim has no clock; pretend the market is always open."""
        return True

    def submit_bracket_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: OrderSide,
        stop_price: Decimal,
        take_profit_price: Decimal | None = None,  # noqa: ARG002 — sim ignores
        time_in_force: str = "day",  # noqa: ARG002 — sim ignores
        client_order_id: str | None = None,
    ) -> BrokerOrder:
        """Record a pending bracket order. Fills at the next mark-to-market.

        CLAUDE.md hard rule 3: every position has a broker-side stop. The
        ``stop_price`` is required and the sim's ``mark_to_market`` enforces
        it bar-by-bar.
        """
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if stop_price <= Decimal("0"):
            raise ValueError(f"stop_price must be positive, got {stop_price}")
        order_id = f"sim-{uuid.uuid4().hex[:16]}"
        coid = client_order_id or f"sim-coid-{uuid.uuid4().hex[:12]}"
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sim_orders (
                    run_id, broker_order_id, client_order_id, symbol, side,
                    qty, stop_price, submitted_at_utc, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    self.run_id,
                    order_id,
                    coid,
                    symbol.upper(),
                    side,
                    int(qty),
                    str(stop_price),
                    now,
                ),
            )
        logger.info(
            "sim_broker[{}]: submitted {} {} {} (stop=${}) id={}",
            self.run_id,
            side,
            qty,
            symbol,
            stop_price,
            order_id,
        )
        return BrokerOrder(
            id=order_id,
            client_order_id=coid,
            symbol=symbol.upper(),
            side=side,
            qty=int(qty),
            filled_qty=0,
            status="accepted",
            order_type="market",
            submitted_at=_utc_now(),
            filled_at=None,
            filled_avg_price=None,
            stop_price=stop_price,
            limit_price=None,
            legs=(),
        )

    def cancel_all(self) -> int:
        """Cancel every pending order. Returns count cancelled."""
        with self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE sim_orders SET status = 'cancelled'
                WHERE run_id = ? AND status = 'pending'
                """,
                (self.run_id,),
            )
            return int(cur.rowcount or 0)

    def close_position(self, symbol: str) -> BrokerOrder:
        """Submit a market-close for one position; fills at next mark-to-market.

        Implemented as a sell-side bracket-style request with stop_price set
        to ``avg_entry_price`` (the close path doesn't actually use the stop;
        it just needs a valid value for the schema).
        """
        sym = symbol.upper()
        positions = {p["symbol"]: p for p in self._get_positions_raw()}
        if sym not in positions:
            raise ValueError(f"sim_broker[{self.run_id}]: no position to close for {sym}")
        pos = positions[sym]
        # Submit as sell (long-only paper assumption).
        order_id = f"sim-close-{uuid.uuid4().hex[:12]}"
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sim_orders (
                    run_id, broker_order_id, client_order_id, symbol, side,
                    qty, stop_price, submitted_at_utc, status
                ) VALUES (?, ?, ?, ?, 'sell', ?, ?, ?, 'pending')
                """,
                (
                    self.run_id,
                    order_id,
                    f"sim-close-coid-{uuid.uuid4().hex[:12]}",
                    sym,
                    int(pos["qty"]),
                    str(pos["avg_entry_price"]),
                    now,
                ),
            )
        logger.info("sim_broker[{}]: queued close {} qty={}", self.run_id, sym, pos["qty"])
        return BrokerOrder(
            id=order_id,
            client_order_id="",
            symbol=sym,
            side="sell",
            qty=int(pos["qty"]),
            filled_qty=0,
            status="accepted",
            order_type="market",
            submitted_at=_utc_now(),
            filled_at=None,
            filled_avg_price=None,
            stop_price=pos["avg_entry_price"],
            limit_price=None,
            legs=(),
        )

    def close_all_positions(self, *, cancel_orders: bool = True) -> list[BrokerOrder]:
        """Best-effort close of every open position. Used by the kill switch."""
        if cancel_orders:
            self.cancel_all()
        out: list[BrokerOrder] = []
        for p in self._get_positions_raw():
            try:
                out.append(self.close_position(p["symbol"]))
            except Exception as e:  # noqa: BLE001
                logger.warning("sim_broker[{}]: close {} failed: {}", self.run_id, p["symbol"], e)
        return out

    # ------------------------------------------------------------------ mark
    def mark_to_market(self, as_of: date | datetime) -> dict[str, object]:
        """Advance the simulated clock to ``as_of``.

        Steps in order:

        1. Fill any pending orders submitted *before* ``as_of`` at that
           date's open. (Submission day T → fill at T+1 open or later.)
        2. Walk each held position's bar between last_clock and as_of:
           if a long's low <= stop_price, force-close at stop_price.
        3. Mark every still-open position to ``as_of``'s close.
        4. Record one ``sim_nav_history`` row keyed (run_id, as_of_date).
        5. Update ``last_clock_utc``.

        Idempotent: marking the same date twice is a no-op (the
        ``UNIQUE(run_id, as_of_date)`` on ``sim_nav_history`` and the
        last_clock check both prevent double-application).

        Returns a small dict summary for logging/tests.
        """
        as_of_dt = (
            as_of
            if isinstance(as_of, datetime)
            else datetime.combine(as_of, datetime.min.time(), tzinfo=UTC)
        )
        if as_of_dt.tzinfo is None:
            as_of_dt = as_of_dt.replace(tzinfo=UTC)
        as_of_date = _to_date(as_of_dt)

        last = self._last_clock()
        if last is not None and as_of_dt <= last:
            return {"as_of": as_of_date.isoformat(), "no_op": True, "fills": 0, "stops": 0}

        # ------------------- 1) fill pending orders submitted before today
        fills_executed = 0
        with self._conn() as conn:
            pending_rows = conn.execute(
                """
                SELECT broker_order_id, symbol, side, qty, stop_price, submitted_at_utc
                FROM sim_orders
                WHERE run_id = ? AND status = 'pending'
                ORDER BY id ASC
                """,
                (self.run_id,),
            ).fetchall()

        for o in pending_rows:
            broker_order_id = str(o[0])
            sym = str(o[1])
            side = str(o[2])
            qty = int(o[3])
            stop_price = Decimal(str(o[4]))
            submitted_dt = _parse_iso(str(o[5]))
            sub_date = _to_date(submitted_dt)
            # Fill at first bar with date > submission date (next-open rule).
            bars = _load_bars_for(
                sym,
                start=sub_date,
                end=as_of_date + timedelta(days=1),
                duckdb_path=self.duckdb_path,
            )
            fill_bar = None
            fill_date = None
            for bd, brow in bars.iterrows():
                if bd > sub_date and bd <= as_of_date:
                    fill_bar = brow
                    fill_date = bd
                    break
            if fill_bar is None:
                # No bar yet — leave pending.
                continue

            fill_price = Decimal(str(float(fill_bar["open"])))
            fill_ts = datetime.combine(fill_date, datetime.min.time(), tzinfo=UTC)
            if side == "buy":
                # Open / increase a long position.
                self._apply_long_buy(
                    sym=sym,
                    qty=qty,
                    fill_price=fill_price,
                    stop_price=stop_price,
                    fill_ts=fill_ts,
                    broker_order_id=broker_order_id,
                )
            else:  # sell — close
                self._apply_long_sell(
                    sym=sym,
                    qty=qty,
                    fill_price=fill_price,
                    fill_ts=fill_ts,
                    broker_order_id=broker_order_id,
                    reason="close",
                )
            with self._conn() as conn:
                conn.execute(
                    """
                    UPDATE sim_orders SET status = 'filled'
                    WHERE run_id = ? AND broker_order_id = ?
                    """,
                    (self.run_id, broker_order_id),
                )
            fills_executed += 1

        # ------------------- 2) walk bars from last->as_of, fire stops
        # We need to walk each held position day by day so a stop fired
        # mid-week force-closes BEFORE we mark to as_of's close.
        stops_fired = 0
        if last is None:
            # First mark — only inspect the single ``as_of`` bar for stops.
            walk_start = as_of_date
        else:
            walk_start = _to_date(last) + timedelta(days=1)
        # Iterate over each calendar day from walk_start..as_of (inclusive).
        cur_d = walk_start
        while cur_d <= as_of_date:
            # snapshot positions
            for p in list(self._get_positions_raw()):
                bars = _load_bars_for(
                    p["symbol"],
                    start=cur_d,
                    end=cur_d,
                    duckdb_path=self.duckdb_path,
                )
                if bars.empty:
                    # weekend / no bar — skip this day for this symbol
                    continue
                row = bars.iloc[0]
                low = Decimal(str(float(row["low"])))
                high = Decimal(str(float(row["high"])))
                stop = p["stop_price"]
                if p["side"] == "long" and low <= stop:
                    fill_ts = datetime.combine(cur_d, datetime.min.time(), tzinfo=UTC)
                    self._apply_long_sell(
                        sym=p["symbol"],
                        qty=int(p["qty"]),
                        fill_price=stop,
                        fill_ts=fill_ts,
                        broker_order_id=f"sim-stop-{uuid.uuid4().hex[:12]}",
                        reason="stop",
                    )
                    stops_fired += 1
                elif p["side"] == "short" and high >= stop:
                    fill_ts = datetime.combine(cur_d, datetime.min.time(), tzinfo=UTC)
                    self._apply_short_cover(
                        sym=p["symbol"],
                        qty=int(p["qty"]),
                        fill_price=stop,
                        fill_ts=fill_ts,
                        broker_order_id=f"sim-stop-{uuid.uuid4().hex[:12]}",
                    )
                    stops_fired += 1
            cur_d = cur_d + timedelta(days=1)

        # ------------------- 3) mark survivors to as_of close
        for p in self._get_positions_raw():
            bars = _load_bars_for(
                p["symbol"],
                start=as_of_date - timedelta(days=10),  # forgive missing bar (weekend)
                end=as_of_date,
                duckdb_path=self.duckdb_path,
            )
            if bars.empty:
                continue
            mark_price = Decimal(str(float(bars.iloc[-1]["close"])))
            self._upsert_position(
                symbol=p["symbol"],
                side=p["side"],
                qty=int(p["qty"]),
                avg_entry_price=p["avg_entry_price"],
                stop_price=p["stop_price"],
                last_mark_price=mark_price,
                opened_at_utc=p["opened_at_utc"],
            )

        # ------------------- 4) record nav history
        cash = self._get_cash()
        positions_now = self._get_positions_raw()
        positions_mv = sum(
            (Decimal(p["qty"]) * p["last_mark_price"] for p in positions_now),
            start=Decimal("0"),
        )
        equity = cash + positions_mv
        # Realized P&L today: sum of fills.reason in {'close','stop'} on this date
        # minus their cost basis. We approximate via NAV diff vs last NAV row.
        with self._conn() as conn:
            prev = conn.execute(
                """
                SELECT equity FROM sim_nav_history
                WHERE run_id = ? ORDER BY as_of_date DESC LIMIT 1
                """,
                (self.run_id,),
            ).fetchone()
            prev_equity = Decimal(str(prev[0])) if prev else self.start_nav
            realized_today = equity - prev_equity
            conn.execute(
                """
                INSERT INTO sim_nav_history (
                    run_id, as_of_date, cash, positions_mv, equity,
                    realized_pl_today
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, as_of_date) DO UPDATE SET
                    cash = excluded.cash,
                    positions_mv = excluded.positions_mv,
                    equity = excluded.equity,
                    realized_pl_today = excluded.realized_pl_today
                """,
                (
                    self.run_id,
                    as_of_date.isoformat(),
                    str(cash),
                    str(positions_mv),
                    str(equity),
                    str(realized_today),
                ),
            )

        # ------------------- 5) advance clock
        self._set_last_clock(as_of_dt)
        return {
            "as_of": as_of_date.isoformat(),
            "fills": fills_executed,
            "stops": stops_fired,
            "equity": str(equity),
            "cash": str(cash),
            "positions": len(positions_now),
        }

    # ------------------------------------------------------------------ fill helpers
    def _apply_long_buy(
        self,
        *,
        sym: str,
        qty: int,
        fill_price: Decimal,
        stop_price: Decimal,
        fill_ts: datetime,
        broker_order_id: str,
    ) -> None:
        positions = {p["symbol"]: p for p in self._get_positions_raw()}
        cash = self._get_cash()
        cost = Decimal(qty) * fill_price
        if cost > cash:
            # Cap qty to what cash can fund (no margin).
            new_qty = int((cash / fill_price).to_integral_value(rounding=ROUND_DOWN))
            if new_qty <= 0:
                logger.warning(
                    "sim_broker[{}]: insufficient cash ${} for {} @ ${}, skipping",
                    self.run_id,
                    cash,
                    sym,
                    fill_price,
                )
                return
            qty = new_qty
            cost = Decimal(qty) * fill_price

        if sym in positions:
            existing = positions[sym]
            new_qty = int(existing["qty"]) + qty
            new_avg = (
                existing["avg_entry_price"] * Decimal(existing["qty"]) + fill_price * Decimal(qty)
            ) / Decimal(new_qty)
            self._upsert_position(
                symbol=sym,
                side="long",
                qty=new_qty,
                avg_entry_price=new_avg,
                stop_price=stop_price,
                last_mark_price=fill_price,
                opened_at_utc=existing["opened_at_utc"],
            )
        else:
            self._upsert_position(
                symbol=sym,
                side="long",
                qty=qty,
                avg_entry_price=fill_price,
                stop_price=stop_price,
                last_mark_price=fill_price,
                opened_at_utc=fill_ts,
            )
        self._set_cash(cash - cost)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sim_fills (
                    run_id, broker_order_id, symbol, side, qty, price,
                    filled_at_utc, reason
                ) VALUES (?, ?, ?, 'buy', ?, ?, ?, 'open')
                """,
                (
                    self.run_id,
                    broker_order_id,
                    sym,
                    int(qty),
                    str(fill_price),
                    fill_ts.isoformat(),
                ),
            )
        logger.info(
            "sim_broker[{}]: filled BUY {} qty={} @ ${} (stop=${})",
            self.run_id,
            sym,
            qty,
            fill_price,
            stop_price,
        )

    def _apply_long_sell(
        self,
        *,
        sym: str,
        qty: int,
        fill_price: Decimal,
        fill_ts: datetime,
        broker_order_id: str,
        reason: str,
    ) -> None:
        positions = {p["symbol"]: p for p in self._get_positions_raw()}
        if sym not in positions:
            return
        existing = positions[sym]
        sell_qty = min(int(qty), int(existing["qty"]))
        proceeds = Decimal(sell_qty) * fill_price
        cash = self._get_cash()
        self._set_cash(cash + proceeds)

        remaining = int(existing["qty"]) - sell_qty
        if remaining <= 0:
            self._delete_position(sym)
        else:
            self._upsert_position(
                symbol=sym,
                side="long",
                qty=remaining,
                avg_entry_price=existing["avg_entry_price"],
                stop_price=existing["stop_price"],
                last_mark_price=fill_price,
                opened_at_utc=existing["opened_at_utc"],
            )

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sim_fills (
                    run_id, broker_order_id, symbol, side, qty, price,
                    filled_at_utc, reason
                ) VALUES (?, ?, ?, 'sell', ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    broker_order_id,
                    sym,
                    int(sell_qty),
                    str(fill_price),
                    fill_ts.isoformat(),
                    reason,
                ),
            )
        logger.info(
            "sim_broker[{}]: filled SELL {} qty={} @ ${} ({})",
            self.run_id,
            sym,
            sell_qty,
            fill_price,
            reason,
        )

    def _apply_short_cover(
        self,
        *,
        sym: str,
        qty: int,
        fill_price: Decimal,
        fill_ts: datetime,
        broker_order_id: str,
    ) -> None:
        # Symmetric to long_sell; the sim is long-only by default but we
        # handle short in case a future variant needs it.
        positions = {p["symbol"]: p for p in self._get_positions_raw()}
        if sym not in positions:
            return
        existing = positions[sym]
        cover_qty = min(int(qty), int(existing["qty"]))
        cost = Decimal(cover_qty) * fill_price
        cash = self._get_cash()
        self._set_cash(cash - cost)

        remaining = int(existing["qty"]) - cover_qty
        if remaining <= 0:
            self._delete_position(sym)
        else:
            self._upsert_position(
                symbol=sym,
                side="short",
                qty=remaining,
                avg_entry_price=existing["avg_entry_price"],
                stop_price=existing["stop_price"],
                last_mark_price=fill_price,
                opened_at_utc=existing["opened_at_utc"],
            )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sim_fills (
                    run_id, broker_order_id, symbol, side, qty, price,
                    filled_at_utc, reason
                ) VALUES (?, ?, ?, 'buy', ?, ?, ?, 'stop')
                """,
                (
                    self.run_id,
                    broker_order_id,
                    sym,
                    int(cover_qty),
                    str(fill_price),
                    fill_ts.isoformat(),
                ),
            )


# ---------------------------------------------------------------------------- helpers


def fetch_sim_nav_history(
    *,
    run_id: str,
    db_path: Path | None = None,
) -> list[dict]:
    """Return the chronological NAV-history rows for a sim run."""
    init_schema(db_path)
    with book.get_book_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT as_of_date, cash, positions_mv, equity, realized_pl_today
            FROM sim_nav_history
            WHERE run_id = ?
            ORDER BY as_of_date ASC
            """,
            (run_id,),
        ).fetchall()
    return [
        {
            "as_of_date": str(r[0]),
            "cash": Decimal(str(r[1])),
            "positions_mv": Decimal(str(r[2])),
            "equity": Decimal(str(r[3])),
            "realized_pl_today": Decimal(str(r[4])),
        }
        for r in rows
    ]
