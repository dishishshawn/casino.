# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

Pre-implementation. The repo currently contains only `README.md` — no source, no `pyproject.toml`, no tests, no CI. Treat the README as the design spec; treat the layout in README "Repository Layout" as the *target* structure to build into, not a reflection of what exists.

The on-disk directory is `WALLACE/`; the README and Python package use the name `casino/`. When scaffolding the package, use `casino/` (matches README, prompts, and references) and do not rename to match the directory.

## Project Identity

`casino` is a single-developer, Claude-powered retail trading system. The v1 strategy is **Earnings-Drift LLM Long-Short**: classical PEAD (post-earnings-announcement drift) signal augmented with Claude-scored earnings-call transcripts. The system is intentionally narrow and opinionated; do not generalize it into a "trading framework."

This is a *real-money* project for the author. Mistakes that pass tests but break trading invariants (look-ahead bias, missing stops, broker reconciliation drift) cause direct financial loss. Treat the invariants in the "Hard Rules" section as load-bearing, not aspirational.

## Stack and Tooling

- **Python 3.11+**, package-managed by **`uv`** (not pip/poetry). Use `uv add` / `uv sync` / `uv run`.
- **Data**: DuckDB + Parquet for market/research data; SQLite for orders and run state.
- **Backtest**: `vectorbt` for fast research sweeps; `backtrader` for event-driven validation. Both layers are required — vectorbt sweeps that look promising must be re-confirmed in backtrader before going to paper.
- **Broker**: `alpaca-py` (paper for v1, live cash account later); `ib_insync` is a future migration target, not for v1.
- **LLM**: Anthropic SDK. Default to **Sonnet 4.6**; use Haiku 4.5 for high-volume classification; reserve Opus 4.7 for rare deep analysis. Always enable prompt caching on system/instruction blocks.
- **Lint/type/test**: `ruff`, `mypy`, `pytest`. CI runs all three on every push.
- **Logging**: `loguru` + Discord webhook.

## Common Commands

```bash
# Environment
uv sync                                  # install/update deps from lockfile
uv add <pkg>                             # add a dep (do not edit pyproject.toml by hand)

# Quality gates (must all pass before commit)
uv run ruff check .
uv run ruff format --check .
uv run mypy casino
uv run pytest                            # full suite
uv run pytest tests/test_signals.py      # one file
uv run pytest -k "no_lookahead"          # one test by name
uv run pytest -x --ff                    # stop on first fail, failed-first ordering

# Data ingestion (one-off and cron entrypoints)
uv run python -m casino.data.ingest_tiingo --ticker AAPL --days 30
uv run python -m casino.data.ingest_edgar --form 8-K --days 7

# Jobs (these are the cron targets in production)
uv run python -m casino.jobs.earnings_daily
uv run python -m casino.jobs.news_intraday
uv run python -m casino.jobs.reconcile_eod

# Monitoring
uv run streamlit run casino/monitoring/dashboard.py
```

## Architecture (target shape)

The system is a one-way data pipeline with a tightly controlled LLM seam in the middle:

```
ingest (data/)  →  signals/  →  backtest/  ──┐
                       │                     │
                       └──→  llm/ (cached) ──┤
                                             ├──→ execution/ → broker
                                             │         │
                       monitoring/  ←────────┘         └──→ reconcile (SQLite book)
```

Key boundaries that future changes must respect:

- **`data/store.py` is the only path to DuckDB/Parquet.** Other modules import helpers from it; they don't open DuckDB connections directly. This keeps point-in-time correctness enforceable in one place.
- **`llm/client.py` is the only path to the Anthropic API.** It owns retries, caching headers, the cost log, and the per-call audit row written for the dashboard. Signal modules call typed wrappers in `llm/prompts/*`, never `anthropic.Anthropic()` directly.
- **`signals/*` consume historical data and produce numeric scores.** They must be callable from both backtest and live-trading paths with identical inputs producing identical outputs — no environment-dependent branching.
- **`backtest/` never imports from `execution/`** and vice versa. Crossing that line is how look-ahead bias and live/sim divergence creep in.
- **`execution/risk.py` is authoritative for sizing, stops, and the kill switch.** Brokers and signals are *inputs* to risk; risk is not an "advisory layer" downstream of them.
- **`execution/reconcile.py` is the source of truth for "what we actually hold."** Any module reasoning about positions reads from reconcile, not from the broker API directly and not from local order state alone.

### LLM call discipline

Every LLM call must:

1. Use prompt caching on the system prompt and any stable instruction block (`cache_control: {"type": "ephemeral"}`).
2. Request structured output validated by a Pydantic schema in `casino/llm/schemas.py`. No free-text scoring.
3. Be logged to the cost/audit table with: prompt hash, model, input/cached/output tokens, USD cost, latency, and the parsed score.
4. In backtest contexts, **anonymize entities** (replace ticker and company name with `<COMPANY>`) and **assert via `tests/test_backtest_no_lookahead.py` that no prompt contains a date inside the backtest window**. This test is a release gate; do not skip, xfail, or "temporarily disable" it.

## Hard Rules (do not violate without an explicit user instruction)

These come from the README's risk section and are the project's non-negotiable invariants:

1. **No live trading without:** 3+ months of paper-trading history, Deflated Sharpe > 0, and Bailey/Lopez de Prado correction applied to the trial count.
2. **No backtest that overlaps the model's training cutoff** unless entity anonymization is in effect.
3. **Broker-side stop-loss on every position.** Code-only stops are not sufficient — bugs and outages happen.
4. **Kill switch must remain a single command** that flattens positions and disables order entry. If you change execution code, re-verify the kill switch path still works end-to-end.
5. **Per-trade risk ≤ 1.5% of NAV; max single-name 10%; gross exposure ≤ 100%.** Sizing helpers in `risk.py` enforce these; do not add code paths that bypass them.
6. **Fractional Kelly only (¼ to ½).** Full-Kelly sizing is forbidden.
7. **Cash account, not margin,** until 6+ months of profitable live history exist.
8. **Cost ceiling:** if monthly API + data spend exceeds 2% of expected monthly P&L, downsize before adding features.

## Backtest Hygiene (the easy way to lose money)

When working on `backtest/` or `signals/`, default to the most paranoid interpretation:

- Use **point-in-time** universe constituents (no survivorship bias). Never join on today's S&P 500 membership.
- Apply realistic costs: assume 5–10 bps round-trip on liquid US equities, more on small caps. A backtest that doesn't break under realistic costs is wrong, not good.
- Run **walk-forward** in `backtest/walk_forward.py`; never report a single-split in-sample Sharpe as a result.
- Apply **Deflated Sharpe** (`backtest/deflated_sharpe.py`) accounting for the number of trials run during research. Most retail edges vanish here — that is the point.
- For LLM-dependent signals, run prompts at temperature > 0 several times per observation; report variance, not just mean.

## Conventions

- **Money is `Decimal`**, not `float`, end-to-end through orders, fills, and the SQLite book. Floats are acceptable for research-only score computation in `signals/`.
- **All times stored UTC.** Convert to America/New_York only at display boundaries (dashboard, alerts).
- **Configuration centralizes in `casino/config.py`,** which loads from `.env`. Do not read `os.environ` from feature modules.
- **Notebooks are exploratory only.** Promote any logic worth keeping into `casino/` with tests; never let a notebook become the source of truth for a signal or a backtest result.
