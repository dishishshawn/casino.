# casino — operations runbook

This runbook covers the day-to-day and emergency procedures for the
`casino` paper-trading deployment. It is written for the single
operator (the author) and assumes the system is running on a single
Hetzner VPS, with cron driving the daily and intraday jobs.

---

## 1. System overview

| Layer            | Module / file                              | Purpose                                                     |
| ---------------- | ------------------------------------------ | ----------------------------------------------------------- |
| Market data      | `casino/data/store.py` (DuckDB)            | OHLCV, fundamentals, news, transcripts, EDGAR filings       |
| Run state        | `casino/execution/book.py` (SQLite)        | Orders, fills, positions, daily P&L, kill-switch flag       |
| LLM audit        | `casino/llm/audit.py` (SQLite)             | One row per Anthropic call (model, tokens, cost, score)     |
| Broker           | `casino/execution/alpaca_broker.py`        | Alpaca paper wrapper; bracket orders only                   |
| Risk             | `casino/execution/risk.py`                 | Sizing, stops, kill switch, exposure caps                   |
| Reconcile        | `casino/execution/reconcile.py`            | Broker-vs-book truth source; drift detection                |
| Daily cron       | `casino/jobs/earnings_daily.py`            | Score reports, place baskets, EOD reconcile                 |
| Intraday cron    | `casino/jobs/news_intraday.py`             | Headline classifier (every 15 min during RTH)               |
| EOD cron         | `casino/jobs/reconcile_eod.py`             | Reconcile + write `daily_pnl` + drawdown alerts             |
| Dashboard        | `casino/monitoring/dashboard.py`           | Streamlit; reads everything; renders P&L + ledger           |
| Alerts           | `casino/monitoring/alerts.py`              | Discord webhook dispatcher                                  |
| Kill switch      | `casino/execution/kill_switch.py`          | `python -m` CLI: cancel + flatten + disable                 |

All money is `Decimal`. Times are stored UTC; rendered America/New_York.

---

## 2. Deployment

### 2.1 Prerequisites

- Hetzner CX22 (or similar) VPS, Ubuntu 24.04, 2 vCPU / 4 GB RAM minimum.
- Outbound HTTPS to: `api.tiingo.com`, `api.alpaca.markets`,
  `paper-api.alpaca.markets`, `api.anthropic.com`, `discord.com`,
  `www.sec.gov`.
- Discord channel + webhook URL.
- Alpaca paper account (Live API keys must NOT be used until 6+ months of
  profitable paper history exists per CLAUDE.md hard rule 7).
- Anthropic API key with monthly cap set on the Anthropic dashboard.
- Tiingo API key.

### 2.2 First-time install

```bash
sudo apt-get update && sudo apt-get install -y git build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/<owner>/casino.git
cd casino
uv sync --frozen
```

### 2.3 Environment file

Create `/etc/casino/casino.env` (root-owned, mode 0640) with:

```
ANTHROPIC_API_KEY=...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
TIINGO_API_KEY=...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
CASINO_DATA_DIR=/var/lib/casino/data
CASINO_DUCKDB_PATH=/var/lib/casino/data/casino.duckdb
CASINO_STATE_SQLITE_PATH=/var/lib/casino/data/state.sqlite
SEC_USER_AGENT=casino-trading <your-email>@example.com
```

> Do not toggle `ALPACA_BASE_URL` to live until the gating conditions in
> §1 of CLAUDE.md ("Hard Rules") are met. The risk-management module
> rejects margin and full-Kelly configurations regardless of broker
> mode, so flipping the URL is a deliberate, reviewed change.

Then bootstrap the database:

```bash
uv run python -c "from casino.data.store import create_schema; create_schema()"
uv run python -c "from casino.execution.book import init_schema; init_schema()"
uv run python -c "from casino.llm.audit import init_audit_schema; init_audit_schema()"
```

### 2.4 Cron schedule (UTC)

US equity regular trading hours are 13:30–20:00 UTC during EDT,
14:30–21:00 UTC during EST. The cron entries below assume the host
clock is UTC (`timedatectl set-timezone UTC`).

```cron
# Daily ingestion (post-close + overnight)
30 21 * * 1-5  cd /opt/casino && /root/.local/bin/uv run python -m casino.data.ingest_tiingo --universe sp500 --days 1

# Intraday news classifier — every 15 min during US RTH (EDT/EST tolerant)
*/15 13-21 * * 1-5  cd /opt/casino && /root/.local/bin/uv run python -m casino.jobs.news_intraday

# Daily earnings basket (15 min before close)
45 19 * * 1-5  cd /opt/casino && /root/.local/bin/uv run python -m casino.jobs.earnings_daily

# End-of-day reconcile + P&L row
30 21 * * 1-5  cd /opt/casino && /root/.local/bin/uv run python -m casino.jobs.reconcile_eod
```

Set `EnvironmentFile=/etc/casino/casino.env` (systemd timer) or
`. /etc/casino/casino.env` (cron) so config is loaded.

### 2.5 Dashboard

Run interactively over an SSH tunnel:

```bash
# on the VPS
uv run streamlit run casino/monitoring/dashboard.py --server.address 127.0.0.1 --server.port 8501

# locally
ssh -L 8501:127.0.0.1:8501 casino-vps
# then visit http://localhost:8501
```

Do not expose port 8501 publicly. Streamlit has no auth.

---

## 3. Daily routine

Each weekday morning:

1. Open the dashboard. Confirm the **trading-disabled** banner is
   absent. Check the prior-day's daily P&L row exists.
2. Open the Discord channel; scroll the prior 24h. Any `critical`
   alert (red) is investigated before the market opens.
3. Eyeball the `Recent LLM calls` table. Schema-validation failures or
   latency spikes (>5 s on Haiku) point at a model/version drift.
4. Eyeball the `Open positions vs broker` table. Any `✗` in the **In
   sync** column means the EOD reconcile flagged drift — see §5.

Each weekday after the close:

1. Check the `Daily spend` panel — confirm spend stayed under the
   monthly budget trajectory.
2. Quickly sanity-check today's basket against open positions. If the
   daily-job basket exceeded `n_submitted = 0` (i.e. orders went out),
   ensure the bracket stops are visible in Alpaca's portal too.

---

## 4. Emergency: kill switch

CLAUDE.md hard rule 4 / PRD §8: **the kill switch must remain a single
command**. The implementation flattens positions and disables order
entry.

### 4.1 Engage

```bash
uv run python -m casino.execution.kill_switch --reason "drawdown breach"
```

This:

1. Cancels every open Alpaca order (`broker.cancel_all`).
2. Submits market-close for every open position (`broker.close_all_positions`).
3. Writes `trading_disabled=1` to `state.sqlite`. Future calls to
   `casino.execution.risk.submit_order` raise `TradingDisabledError`
   without touching the broker.

The CLI prints a summary; the same summary is logged.

### 4.2 Verify

After running the kill switch:

```bash
# Verify flag is set
uv run python -c "from casino.execution.book import is_trading_disabled; print(is_trading_disabled())"
# True

# Verify broker is flat
uv run python -c "from casino.execution.alpaca_broker import build_default_broker; print(len(build_default_broker().get_positions()))"
# 0
```

### 4.3 Re-enable

Only after the operator (you) has performed the post-mortem documented
in §6 and has decided trading should resume:

```bash
uv run python -m casino.execution.kill_switch --reenable
```

---

## 5. Reconciliation drift

The `casino.execution.reconcile.reconcile` function runs at the end of
the daily cron and inside the EOD job. Drift kinds:

| Kind             | Severity     | Meaning                                                 |
| ---------------- | ------------ | ------------------------------------------------------- |
| `broker_only`    | critical     | Broker holds something the book doesn't know about.     |
| `book_only`      | critical     | Book has a row the broker is missing.                   |
| `qty_mismatch`   | critical     | Both sides hold the symbol but qty differs.             |
| `side_mismatch`  | critical     | Long vs short disagreement.                             |
| `price_drift`    | informational | Mark-to-market differs from book entry-cost by ≥ $1.    |

When a critical alert fires:

1. **Stop**. Do not let another cron tick run with bad state.
2. Engage the kill switch (§4.1) if the drift looks like it could grow
   another order in the next cycle (e.g. a stale book entry could lead
   to oversizing).
3. Capture both sides for the post-mortem:
   ```bash
   uv run python -c "from casino.execution.alpaca_broker import build_default_broker as b; print([p for p in b().get_positions()])"
   uv run python -c "from casino.execution import book; print(book.fetch_positions())"
   ```
4. Decide which side is right. Default assumption: the broker is the
   source of truth, the book is what's broken.
5. Resync the book from the broker:
   ```bash
   uv run python -c "from casino.execution.reconcile import sync_book_from_broker; from casino.execution.alpaca_broker import build_default_broker; sync_book_from_broker(broker=build_default_broker())"
   ```
6. Re-run reconcile and confirm `in_sync=True`.
7. Re-enable trading (§4.3) only if the drift root cause is understood
   and addressed in code.

---

## 6. Drawdown procedures

PRD §8 / §10 thresholds:

- **Drawdown > 10%** — Discord critical alert; review same-day. Review
  the basket selection logic, recent regime flips, and whether the
  cost guard is firing on the LLM ledger. Continue trading only if the
  loss source is understood (e.g. a known sector blow-up the strategy
  was on the wrong side of, not a code bug).

- **Drawdown > 35%** — Hard stop. Engage the kill switch immediately.
  Do not trade until:
  1. A full post-mortem is written.
  2. The historical Sharpe / Deflated Sharpe is recomputed including
     the drawdown period.
  3. If the strategy still passes the bar, the kill switch is cleared
     by hand. If not, retire the strategy and re-enter research.

The dashboard shows drawdown as `(high_water_mark - equity_today) /
high_water_mark` over the entire `daily_pnl` history.

---

## 7. Cost guard procedures

The intraday news job is configured with a `daily_budget_usd` (default
$5, matching PRD §10's alert threshold). When the cumulative LLM spend
on the day reaches the cap, the job:

1. Logs a warning.
2. Fires a `warning` Discord alert (`alert_llm_spend`).
3. Exits without making more calls. Headlines accumulated during the
   skipped window will be picked up by the next cron tick if the
   budget rolls over (it does not — caps are daily, not hourly).

If the cap is breached on multiple consecutive days:

1. Confirm headline ingestion volume hasn't blown out (DuckDB
   `news` table count vs prior weeks).
2. Audit the LLM ledger for retry storms (`schema_name` repeats with
   `success=0`).
3. Either raise the budget ceiling (and the §10 alert threshold to
   match) or cap intraday ticker coverage in `news_intraday.py`.

The budget enforcer reads from the audit log — it sees *all* LLM
spend, including the daily earnings basket. The earnings basket
typically dominates and runs once per day; the intraday job's job is
to stay within the residual headroom.

---

## 8. Recipients + escalation

The Discord webhook is the only paging channel. The operator (the
author) is the only on-call. There is no second-line.

| Severity     | What to do                                                     |
| ------------ | -------------------------------------------------------------- |
| `info`       | Logged in Discord. Nothing required.                           |
| `warning`    | Review same-day. Common: cost cap, near-threshold drawdowns.   |
| `critical`   | Page-equivalent. Investigate within 1 hour during market hours. |

If the system is left unattended for >24 hours, the operator engages
the kill switch before stepping away and clears it on return.

---

## 9. Backups

- `state.sqlite` and `casino.duckdb` are backed up nightly via cron:
  ```cron
  0 2 * * *  rsync -a /var/lib/casino/data /backups/casino-$(date +\%F)/
  ```
- Keep 30 days of daily snapshots; keep monthly snapshots indefinitely
  (they are small).
- The LLM audit log is the only record of model / prompt / score
  versions; do not let the SQLite file get corrupted. Use
  `sqlite3 state.sqlite ".backup state.sqlite.bak"` if you need a
  hot copy without stopping crons.

---

## 10. Common commands cheatsheet

```bash
# Quality gates (also run by CI)
uv run ruff check .
uv run ruff format --check .
uv run mypy casino
uv run pytest

# Manual runs of the cron jobs
uv run python -m casino.jobs.earnings_daily
uv run python -m casino.jobs.news_intraday
uv run python -m casino.jobs.reconcile_eod

# Kill switch
uv run python -m casino.execution.kill_switch --reason "manual review"
uv run python -m casino.execution.kill_switch --reenable

# Dashboard
uv run streamlit run casino/monitoring/dashboard.py
```
