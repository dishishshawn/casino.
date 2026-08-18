# Backlog: Monte Carlo Permutation Testing (MCPT) for backtest significance

**Filed:** 2026-06-16
**Status:** Not started — saved for a future agent
**Origin:** Review of https://github.com/neurotrader888/mcpt (neurotrader888), MIT-licensed.

## Why

`CLAUDE.md` backtest-hygiene already mandates Deflated Sharpe + Bailey/López de
Prado trial-count correction + walk-forward. MCPT is a complementary, empirical
significance test that prices in **data-mining / parameter-selection bias**
directly (by re-optimizing on each permutation) rather than via a closed-form
correction. It also supports **multi-market** permutation with correlations
preserved-then-broken — a natural fit for the 7-ETF TSMOM basket. Good fit for
trend/momentum strategies specifically (the permutation destroys serial
structure while keeping per-bar distributions).

Immediate motivation: get an honest read on whether the TSMOM strategy has a
detectable edge or is living inside the noise band (the Branch C paper run is
"sorta failing"; this informs whether to continue).

## What to take from the source repo

- **`bar_permute.py::get_permutation(ohlc, start_index=0, seed=None)`** — the one
  genuinely reusable asset. Algorithm: convert OHLC to log prices; decompose
  each bar into a *gap* term (open vs previous close) and *intrabar* terms
  (high/low/close vs this bar's open); independently shuffle the gap series and
  the intrabar series; rebuild bars anchored at `start_index`; exponentiate
  back. Accepts a single DataFrame or a list of aligned DataFrames (multi-market,
  shuffled in lockstep). `start_index` anchors a fixed prefix (used so
  walk-forward only permutes the out-of-sample/post-train region).
- **The driver pattern** (from `insample_donchian_mcpt.py` /
  `walkforward_donchian_mcpt.py`): optimize on real data → re-run on N≈1000
  permutations → `p_value = (# perms with best_metric >= real_best_metric) / N`.
  Methodology to reimplement, NOT code to import.

## What to skip

- The driver scripts verbatim — they are demo-grade: hardcoded BTC parquet path,
  `matplotlib.show()`, `tqdm`, profit-factor metric, no functions/tests.
- Profit Factor as the metric — use casino's Sharpe / Deflated Sharpe instead.
- Any `pip install` expectation — the repo is not packaged.

## Deliverables (suggested; run through brainstorming → writing-plans → TDD)

1. **Vendor the engine:** `casino/backtest/bar_permutation.py` with
   `get_permutation(...)` — add type hints, docstring, MIT attribution to the
   source, and tests (original has none). Respect the rule that `backtest/`
   never imports from `execution/`.
2. **MCPT runner:** `casino/backtest/mcpt.py` exposing something like
   `run_mcpt(prices, optimize_fn, *, n_permutations=1000, metric=..., mode="insample"|"walkforward", start_index=...)`
   returning the real metric, the permutation distribution, and the p-value.
   Metric should be casino's Sharpe/DSR, not profit factor. Reuse / sit beside
   `casino/backtest/deflated_sharpe.py` and `casino/backtest/walk_forward.py`.
3. **Wire as an optional release gate** alongside Deflated Sharpe (a strategy must
   clear MCPT before promotion), consistent with the existing hygiene gates.
4. **One-off TSMOM check (cheap, do this FIRST):** run in-sample + walk-forward
   MCPT on the current TSMOM lookback/params over the research history and report
   the p-value, before investing in full integration. If TSMOM can't beat its own
   permutations, that's decisive input for the experiment-continuation decision.

## Caveats

- Educational single-author code; verify the permutation math against the source
  and add tests rather than trusting it blind.
- Permutation assumes you want to destroy serial dependence — correct for TSMOM;
  think before applying to mean-reversion on the same bars.
- Reproducibility: thread the `seed` through so MCPT runs are deterministic.
