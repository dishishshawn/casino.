# Design: Handle positions that fell past their protective stop

**Date:** 2026-06-15
**Branch:** fix/book-recovery-stops-fills
**Status:** Approved (design)

## Plain-English summary

Every position is supposed to carry a broker-side "sell if it drops too far"
order (a protective stop). A position can fall *past* that level while it is
temporarily unprotected — exactly what happened to USO on 2026-06-15, which sat
~0.5% below its stop level. When the EOD guard then tries to arm a normal stop,
the broker rejects it (you can't place a sell-stop at or above the current
price), the guard errors out, and the position is left naked.

This change makes the guard handle that case deliberately: it never crashes, it
warns the operator the same evening, and it automatically sells the breached
position the next morning when the market is open and prices are trustworthy.

This is the "Option A (liquidate, guarded)" decision from the brainstorming
session.

## Background / root cause

- Protective stops are armed by `reconcile.ensure_protective_stops`, called at
  EOD from `jobs/reconcile_eod.py`. It computes the stop as
  `avg_entry * (1 - stop_fraction)` (stop_fraction = 0.10) and submits a GTC
  sell-stop for any open long without a live stop.
- A standalone sell-stop is only valid when `stop_price < current price`. If the
  position has already fallen to/below the stop level, Alpaca returns
  `422 / 42210000 "stop price must be less than current market_price"`.
- That error was unhandled and propagated out of the whole loop, so:
  1. the EOD reconcile logged a critical exception, and
  2. any position after the failing one in the loop would have been skipped
     (USO happened to be last, so the other six were armed by luck).
- The deeper cause (a position being naked long enough to drift past its stop)
  is the expired DAY-tif bracket leg. The durable cure — arming the stop the
  instant a position is filled — is a separate fast-follow, not this change.

## Timing constraint (drives the split)

The EOD reconcile runs ~17:15 CDT, **after** market close. Selling into a closed
market would fill at an unreliable price, so liquidation cannot happen at EOD. It
must happen at the next market-open run. The morning book-sync job
(`jobs/sync_book.py`) runs ~09:45 CDT, shortly after the open, while the market
is open — that is where liquidation belongs.

## Behavior

For each open long position, compare the intended stop price `stop_px` against
the current `market_price`:

1. **`stop_px < market_price` (normal):** arm a GTC sell-stop. Unchanged from
   today.
2. **Breached but within the cushion** (`market_price` is below `stop_px` by less
   than the cushion, 0.25%): cannot arm a valid stop, but the breach is within
   quote noise. Do **not** sell. Emit a warning/alert and re-evaluate next run.
3. **Materially breached** (`market_price <= stop_px * (1 - 0.0025)`):
   - **At EOD (market closed):** do not trade. Fire a **critical** alert:
     "<SYM> is below its protective stop and unprotected — it will be sold at the
     next market open."
   - **At morning sync (market open):** sell the position at market via
     `broker.close_position(sym)`, record the order in the book, and fire a
     **critical** alert: "Sold <SYM> at market — it was X% below its stop."

Each position is handled in its own try/except so one symbol's broker error can
never abort protection for the others.

The cushion (0.25%) exists so ordinary price jitter never triggers an automatic
sale.

## Components

- **`casino/execution/reconcile.py` — `ensure_protective_stops`**
  - Add a `liquidate_breached: bool = False` parameter.
  - Read `position.market_price` (already on `BrokerPosition`).
  - Per-position try/except; collect failures instead of raising.
  - Classify each position as normal / within-cushion / materially-breached per
    the rules above.
  - When `liquidate_breached` is True **and** `broker.is_market_open()` is True,
    call `broker.close_position(sym)` for materially-breached positions, record
    the order, and return a result marked liquidated.
  - Extend the per-position result type with flags: `breached`, `liquidated`
    (plus existing `armed` / `already_protected`).

- **`casino/jobs/reconcile_eod.py`** (runs after close)
  - Call `ensure_protective_stops(..., liquidate_breached=False)`.
  - For any `breached` (and not liquidated) result, fire the critical
    "will be sold at next open" alert. Keep existing re-arm alert for armed ones.

- **`casino/jobs/sync_book.py`** (runs at open)
  - After syncing the book, call
    `ensure_protective_stops(..., liquidate_breached=True)`.
  - For any `liquidated` result, fire the critical "sold at market" alert.

- **`casino/monitoring/alerts.py`**
  - `alert_below_stop_unprotected(symbols)` — critical, EOD wording.
  - `alert_position_liquidated_below_stop(symbol, pct_below, price)` — critical,
    morning wording.

## Configuration

- Cushion margin: `Decimal("0.0025")` (0.25%). Define as a named constant near
  the stop logic; not yet surfaced to `.env` (YAGNI until a second value is
  needed).

## Error handling

- Broker errors arming or closing a single position are caught per-position,
  logged, and surfaced via the unhandled-exception alert, without aborting the
  rest or crashing the job.
- If the market is unexpectedly closed during the morning run, breached
  positions are not sold; they alert and are retried next run.

## Testing

In `tests/test_reconcile.py` (+ alerts tests):
- Normal position → arms GTC stop (unchanged).
- Materially breached + market closed (`liquidate_breached=False`) → flagged
  breached, **no** order submitted, no raise.
- Materially breached + market open (`liquidate_breached=True`) →
  `close_position` called once, result marked liquidated, alert fired.
- Breach within the cushion → no arm, no sale, alert only.
- One position raises a broker error → the others are still processed.
- Alert helpers produce the expected severity/title/fields.

## Out of scope (fast-follow)

- Arming the protective stop immediately on fill (the durable cure that makes
  "drifted past the stop" rare). Its natural home is the same morning sync slot.
- Any change to the strategy itself or the 30-day experiment framing.
