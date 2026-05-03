# Casino — A Claude-Powered Retail Trading System

> **Read this first.** This is gambling money. Real academic studies show ~1% of retail day traders are predictably profitable after fees. LLMs do not change that base rate. The point of this project is to build something *narrow, well-validated, and cheap to run*, paper-trade it for a quarter, and only deploy capital you can fully lose. If you're looking for a money printer, close this tab.

---

## What This Is

A pragmatic, single-developer trading system that uses Claude (via API) for signal generation, classical quant infrastructure for execution, and a strict discipline of bias-controlled backtesting. Built to be developed in [Claude Code](https://docs.claude.com/en/docs/claude-code/overview).

**Core thesis:** the only retail edges that survive in 2026 are (a) narrow event-driven plays with LLM-parsed catalysts and (b) small structural edges in capacity-limited niches. Everything else is noise plus fees.

**v1 strategy:** Earnings-Drift LLM Long-Short (PEAD enhanced with Claude-scored transcripts).

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for quant; best Claude SDK support |
| Package mgmt | `uv` | Fast, reproducible, modern |
| Data store | DuckDB + Parquet | Zero-config, blazing fast on a laptop |
| State / orders | SQLite | Simple, durable, one file |
| Backtest (research) | `vectorbt` | Vectorized, fast parameter sweeps |
| Backtest (validation) | `backtrader` | Event-driven, more realistic execution |
| Broker (start) | Alpaca via `alpaca-py` | Free, clean REST/WS, paper trading |
| Broker (scale) | IBKR via `ib_insync` | Best execution, global, futures/options |
| LLM | Anthropic Claude SDK | Sonnet 4.6 default, Haiku 4.5 high-volume, Opus rare |
| Market data | Tiingo ($10/mo) + SEC EDGAR (free) + yfinance fallback | Best price/value retail stack |
| Scheduler | `cron` + `systemd` → `Prefect` later | Start dumb, graduate when needed |
| Logging | `loguru` + Discord webhook | One-line setup, mobile alerts |
| Hosting | $5/mo Hetzner VPS (or Mac mini) | Avoid cloud-lock-in early |
| CI | GitHub Actions + `ruff` + `mypy` + `pytest` | Catch dumb bugs before they trade |

**Monthly operating cost target: $50–80 all-in.** If costs exceed 2% of expected monthly P&L, downsize.

---

## Repository Layout

```
casino/
├── README.md                    # this file
├── pyproject.toml               # uv / poetry config
├── .env.example                 # API keys template (never commit .env)
├── .github/workflows/ci.yml     # lint + type + test on every push
│
├── casino/
│   ├── __init__.py
│   ├── config.py                # central config, loads from .env
│   │
│   ├── data/
│   │   ├── ingest_edgar.py      # SEC filings (8-K, 10-K, 10-Q, Form 4)
│   │   ├── ingest_tiingo.py     # OHLCV + fundamentals + news
│   │   ├── ingest_transcripts.py# earnings call transcripts (FMP)
│   │   └── store.py             # DuckDB read/write helpers
│   │
│   ├── llm/
│   │   ├── client.py            # Anthropic client w/ caching, retries, cost log
│   │   ├── prompts/
│   │   │   ├── earnings_score.py# transcript scoring (Sonnet)
│   │   │   ├── headline_class.py# news classifier (Haiku)
│   │   │   └── thesis_gen.py    # weekly thesis (Opus, optional)
│   │   └── schemas.py           # Pydantic models for structured outputs
│   │
│   ├── signals/
│   │   ├── pead.py              # standardized unexpected earnings (SUE)
│   │   ├── llm_earnings.py      # combine SUE × Claude transcript score
│   │   └── regime.py            # simple market regime filter
│   │
│   ├── backtest/
│   │   ├── vbt_research.py      # vectorbt sweeps
│   │   ├── bt_validate.py       # backtrader event-driven validation
│   │   ├── walk_forward.py      # strict walk-forward harness
│   │   └── deflated_sharpe.py   # Bailey & Lopez de Prado correction
│   │
│   ├── execution/
│   │   ├── alpaca_broker.py     # paper + live wrapper
│   │   ├── risk.py              # position sizing, stop-losses, kill switch
│   │   └── reconcile.py         # broker positions vs internal book
│   │
│   ├── monitoring/
│   │   ├── dashboard.py         # Streamlit P&L + LLM-call audit
│   │   └── alerts.py            # Discord/Telegram webhook
│   │
│   └── jobs/
│       ├── earnings_daily.py    # cron entry: run before close
│       ├── news_intraday.py     # cron entry: every 15 min during market
│       └── reconcile_eod.py     # cron entry: end of day
│
├── notebooks/                   # exploratory only — never the source of truth
│   └── 01_pead_baseline.ipynb
│
└── tests/
    ├── test_signals.py
    ├── test_backtest_no_lookahead.py  # critical
    └── test_risk.py
```

---

## Setup

### Prerequisites

- Python 3.11+
- A US brokerage account at [Alpaca](https://alpaca.markets) (paper trading is free, no minimum)
- An [Anthropic API key](https://console.anthropic.com)
- A [Tiingo](https://www.tiingo.com) account ($10/mo Starter recommended; free tier works to start)
- Optional: a [Financial Modeling Prep](https://site.financialmodelingprep.com) API key for transcripts

### Install

```bash
# clone and enter
git clone <your-fork> casino && cd casino

# uv handles the venv and lockfile
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# copy env template and fill in keys
cp .env.example .env
$EDITOR .env

# verify everything works
uv run pytest
uv run python -m casino.data.ingest_tiingo --ticker AAPL --days 30
```

### Required environment variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # paper for v1
TIINGO_API_KEY=...
FMP_API_KEY=...                                     # optional
DISCORD_WEBHOOK_URL=...                             # optional alerts
```

---

## The v1 Strategy: Earnings-Drift LLM Long-Short

**Hypothesis:** Post-earnings-announcement drift (PEAD) is a documented anomaly worth ~5–8% annualized historically. LLM-parsed transcript sentiment materially improves the signal (Meursault et al. 2021; ExtractAlpha 2024). Combining standardized unexpected earnings (SUE) with Claude-scored transcript tone produces a higher-conviction signal than either alone.

**Universe:** S&P 500 + Russell 1000 mid-caps. Avoid micro-caps (slippage kills you).

**Signal pipeline:**

1. After every earnings release, fetch the 8-K + transcript.
2. Compute classical SUE (actual EPS − consensus, scaled by historical surprise std).
3. Send transcript to Claude Sonnet 4.6 with cached system prompt; request structured JSON:
   - `beat_quality`: −2 to +2 (was the beat clean or noisy?)
   - `guidance_tone`: −2 to +2
   - `qa_defensiveness`: −2 to +2
   - `confidence`: 0 to 1
   - `reasoning`: short explanation
4. Combined score = `0.5 * normalized_SUE + 0.5 * llm_composite`.
5. **Trade only when SUE and LLM score agree in sign and both exceed thresholds.**

**Position rules:**

- Long top quintile, short bottom quintile (long-only if shorting is impractical at your account size)
- Equal-weighted, max 10 positions per side
- Hold 5–20 trading days; exit on signal decay or stop-loss
- Risk per name: 1% of NAV; total gross: ≤ 100%; max single-name: 10%

**Why this strategy first:**

- Documented academic edge that survives out-of-sample
- LLM materially improves it (multiple 2024–2025 papers)
- Low look-ahead bias risk if restricted to dates after the model's training cutoff
- Capacity-friendly for retail (you'll never compete with Citadel here)
- Fits a part-time workflow — earnings happen on a schedule

---

## Avoiding LLM Look-Ahead Bias (Read This Twice)

This is where 95% of "AI trading" backtests fall apart. The model has read the future during pretraining. Defenses, in order of effectiveness:

1. **Restrict backtest start to *after* the model's training cutoff.** Cleanest defense. For Sonnet 4.6, that means backtesting only on data the model couldn't have seen.
2. **Anonymize entities in prompts.** Replace ticker and company name with `<COMPANY>`. Glasserman & Lin (2023) found anonymized headlines actually outperformed even in-sample — the model's general knowledge of named companies is a *distraction* more than a leak.
3. **Use a chronologically frozen model** (e.g., ChronoBERT, ChronoGPT, or a local Llama snapshot) for serious backtest validation. Use Claude live for forward signal generation only.
4. **Strict walk-forward** with no calendar-date leakage in prompts.
5. **Monte Carlo over LLM stochasticity:** run each prompt 5–10x at temperature > 0; the variance is informative about signal robustness.
6. **Apply Deflated Sharpe Ratio** (Bailey & Lopez de Prado): correct for the number of trials you ran. Most retail backtests are statistically indistinguishable from noise after deflation.

There is a `tests/test_backtest_no_lookahead.py` that asserts the LLM is never called with a date inside the backtest window. **Do not skip this test.**

---

## Cost Model

Claude pricing (per million tokens, retail tier as of 2026):

| Model | Input | Cached input | Output |
|---|---|---|---|
| Haiku 4.5 | $1 | $0.10 | $5 |
| Sonnet 4.6 | $3 | $0.30 | $15 |
| Opus 4.7 | $5 | $0.50 | $25 |

**Always use prompt caching** for system prompts and instruction blocks (~90% saving). **Use Batch API** for non-latency-sensitive jobs (50% saving). **Default to Sonnet 4.6, not Opus** — Opus is rarely worth 5× the cost for trading NLP.

Realistic monthly cost for v1:

| Item | Cost |
|---|---|
| Tiingo Starter | $10 |
| Claude API (transcripts + headlines, w/ caching) | $30–50 |
| Hetzner VPS | $5 |
| FMP (optional) | $15 |
| **Total** | **$60–80/month** |

**Break-even math:** at $80/mo = $960/yr operating cost, you need 9.6% gross return on a $10k account just to cover costs. Below ~$25k of deployed capital, expected value is negative. Below ~$10k, do not run paid services — use the free stack (yfinance, EDGAR, Alpaca free, Claude with hard $20/mo cap).

---

## 30-60-90 Day Plan

**Days 0–30: Foundation**
- Set up environment with Claude Code; ingest SEC EDGAR + Tiingo into DuckDB
- Build a vectorbt backtest of pure SUE-based PEAD (no LLM) to confirm baseline edge
- Read Lopez de Prado *Advances in Financial Machine Learning* ch. 1–7 for proper CV
- Read Ernie Chan *Quantitative Trading*

**Days 31–60: Add the LLM**
- Ingest earnings transcripts; design Claude scoring prompt with strict JSON schema and ticker-anonymized examples
- Backtest SUE × LLM combined signal restricted to post-cutoff dates only
- Compute Deflated Sharpe; run walk-forward
- **Only proceed if signal is statistically significant after deflation**
- Begin paper-trading live for the next earnings season (one full quarter)

**Days 61–90: Honest reality check**
- Compare paper-trade live results to backtest expectations
- **If they differ by more than 50% of backtest Sharpe, do NOT deploy real money**
- If they match: deploy $2,500–$5,000 in a cash account, trade for one quarter
- Reassess scaling

---

## Risk Management Hard Rules

These do not bend.

1. **Never risk more than 1.5% of account on a single trade.**
2. **Never run a strategy live without 3+ months of paper-trading and a Deflated Sharpe > 0.**
3. **Never trade a strategy whose backtest overlaps your LLM's training cutoff without entity anonymization.**
4. **Always have a broker-side stop-loss.** Never rely solely on your code to exit. Bugs and outages happen.
5. **Treat any month where API + data spend > 2% of expected monthly P&L as a red flag.** Downsize.
6. **Use fractional Kelly (¼ to ½), never full Kelly.** Edge estimates are noisy; full Kelly produces 50–80% drawdowns even when correct.
7. **Cash account, not margin, until you have 6+ months of profitable live history.**
8. **Wash sale tracking:** if you trade more than 50 times/year, get TradeLog or GainsKeeper. The wash sale rule applies across *all* your accounts including a spouse's IRA. It will silently destroy your tax efficiency.
9. **Kill switch:** a single command that flattens all positions and disables order entry. Test it.

---

## Monitoring

A single Streamlit dashboard (`uv run streamlit run casino/monitoring/dashboard.py`) showing:

- Live P&L (today, MTD, YTD)
- Open positions vs. broker (reconciliation flag if mismatch)
- Last 50 LLM calls: prompt hash, model, input/output tokens, cost, score, latency
- Daily LLM spend vs. monthly budget
- Drawdown from high-water mark
- Strategy Sharpe (rolling 60-day)
- Discord webhook fires on: order fills, drawdown > 10%, daily LLM spend > $5, broker reconciliation mismatch, unhandled exceptions

---

## Triggers

**Scale up when:** 6 consecutive months of live performance within 30% of backtest *and* Deflated Sharpe > 0.5 *and* max drawdown < 25%.

**Scale down or stop when:** drawdown > 35%, three consecutive losing months in a regime where backtest predicted gains, or live-vs-backtest divergence > 50% of expected Sharpe. These mean your edge is gone or never existed. Do not "let it work itself out."

---

## Pitfalls Graveyard

The ways retail AI traders blow up, ranked by how often they kill the strategy:

1. **Ignoring slippage and commissions.** Assume 5–10 bps round-trip on liquid US equities, 25–50 bps on small caps, 50–200 bps on options, 5–15 bps on top-10 crypto pairs. If your backtest doesn't break with realistic costs, you haven't modeled them realistically.
2. **Overfitting / multiple testing.** With 100 trials, an in-sample Sharpe of 2.0 may be statistically equivalent to noise. Always Deflate.
3. **LLM look-ahead bias.** Covered above.
4. **Survivorship bias in the universe.** Test against historical (point-in-time) constituents, not today's S&P 500. This routinely inflates backtests by 100–300 bps.
5. **Regime change.** Every strategy that worked 2010–2019 broke in 2020–2022. Every COVID strategy broke in 2023. Every 2023 LLM strategy is decaying now.
6. **Position sizing errors.** Full Kelly with a noisy edge is a path to ruin.
7. **Operational risk.** API downtime, exchange outages, your VPS rebooting mid-trade. Always have a kill switch and broker-side stops.
8. **Sample-size lies.** Any "win rate" claimed on < 100 trades is meaningless. You need 666+ trades for 99% confidence.
9. **Course/Twitter grifters.** Real edges aren't sold for $497.

---

## What This Project Is Not

- A signal you can copy-paste and print money. There is no such signal at retail scale.
- A general-purpose framework. It's opinionated and narrow on purpose.
- A reason to skip the math. If you don't want to learn cross-sectional statistics, deflated Sharpe, walk-forward CV, and proper transaction-cost modeling, you will lose money no matter how good the LLM is.
- A substitute for index investing. If you have $50k and no specific edge, VOO and chill is mathematically the better bet for ~99% of people.

---

## References

**Books**
- Lopez de Prado, *Advances in Financial Machine Learning* (2018) — required
- Ernie Chan, *Quantitative Trading* and *Algorithmic Trading* — required
- Lopez de Prado, *Machine Learning for Asset Managers* (2020) — strongly recommended

**Papers**
- Barber, Lee, Liu & Odean (Taiwan day-trading study) — base rates for retail profitability
- Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" — how to correct for selection bias
- Lopez-Lira & Tang (arXiv:2304.07619) — ChatGPT headline sentiment, with decay analysis
- Glasserman & Lin (2023) — anonymization and LLM distraction effects
- Meursault et al. (2021), "PEAD.txt" — NLP on earnings transcripts improves PEAD
- Li et al. (arXiv:2505.07078, 2025) — honest evaluation of LLM trading agents in long-run

**Open-source repos worth studying**
- `TauricResearch/TradingAgents` — multi-agent architecture inspiration
- `AI4Finance-Foundation/FinGPT` — local-LLM option for high-volume sentiment
- `QuantConnect/Lean` — production-grade structure
- `freqtrade/freqtrade` — if you go crypto

---

## License

MIT. Use at your own risk. Nothing here is investment advice. The author is not your financial advisor and is not responsible for your losses.

---

*Last reviewed: building it now. If you find yourself ignoring the rules in this README, that's the moment to stop, not the moment to deploy capital.*
