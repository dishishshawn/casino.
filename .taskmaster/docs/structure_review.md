# Casino structure review — 2026-05-11

<context>
# Overview

Audit of the `casino/` package, `tests/`, `scripts/`, and `.taskmaster/`
focused on **module boundaries**, **dependency direction**, **naming
consistency**, and **test organization**. Findings are prioritized P0/P1/P2
where:

- **P0** = actively breaks a documented invariant or creates a footgun the
  next change will trip over. Fix before adding features.
- **P1** = structural drift; not currently broken but compounds with each
  new feature.
- **P2** = cleanup / consistency; safe to defer.

The codebase is single-developer and real-money paper-trading; correctness-
critical invariants (DuckDB single-path, Anthropic single-path, Alpaca
single-path, backtest⊥execution) **mostly** hold. The exceptions are the
P0s below.

# Scope

In scope: `casino/`, `tests/`, `scripts/`, `pyproject.toml`,
`.taskmaster/`. Source files counted: 56 in `casino/`, 38 in `tests/`,
7 PowerShell wrappers in `scripts/`.

Out of scope: `docs/`, `notebooks/`, `data/`, `reports/`, `logs/`, and
the v1 EDLLLS PRD body (only its invariants are referenced).

# Findings overview

The hard invariants from `CLAUDE.md` largely hold:

- `casino/backtest/` does NOT import from `casino/execution/` ✅
- `casino/execution/` does NOT import from `casino/backtest/` ✅
- `anthropic` is only imported in `casino/llm/client.py` ✅
- `alpaca` is only imported in `casino/execution/alpaca_broker.py` ✅
- `duckdb.connect` happens in **two** places — `data/store.py` (sanctioned)
  and `jobs/heartbeat.py` (P0 violation).

The structural pain concentrates in `casino/execution/`, which has grown
into a mix of primitives (`book`, `risk`, `reconcile`, `alpaca_broker`,
`paper_clock`, `sim_broker`) and CLI entrypoints (`tsmom_runner`,
`tsmom_clock_check`, `tsmom_shadow_runner`, `kill_switch`). The runners
also re-export constants and utilities consumed by other runners, which
makes the dependency graph among entrypoints non-obvious.

Test organization has one critical issue: `FakeTradingClient` lives in
`tests/test_risk.py` and is imported by **7 other test files** — a
hidden-shared-fixture pattern that makes `test_risk.py` non-deletable
and brittle to refactor.

</context>

<PRD>

# P0 — fix before next feature

## 1. `jobs/heartbeat.py` opens DuckDB directly

**Problem.** `casino/data/__init__.py` and `CLAUDE.md` both state
`data/store.py` is the only path to DuckDB; `casino/jobs/heartbeat.py:36`
imports `duckdb` and `duckdb.connect()`s a read-only handle in
`_latest_ohlcv_date`. This is the second place in the package that opens
DuckDB and silently breaks the "point-in-time correctness is enforced in
one file" guarantee.

**Proposed change.** Add `latest_ohlcv_date(*, universe, db_path)` to
`casino/data/store.py`; heartbeat imports it. Delete the local
`duckdb.connect` path.

**Files affected.** `casino/data/store.py` (+helper), `casino/jobs/heartbeat.py`
(swap import + call), `tests/test_store.py` (cover new helper).

**Risk.** Low — heartbeat is read-only; the helper just relocates the SQL.

## 2. `FakeTradingClient` imported across 7 test files from `tests/test_risk.py`

**Problem.** `tests/test_alpaca_broker.py`, `test_heartbeat.py`,
`test_jobs.py`, `test_kill_switch.py`, `test_reconcile.py`,
`test_tsmom_clock_check.py`, and `test_tsmom_runner.py` all do
`from tests.test_risk import FakeTradingClient`. Tests-importing-tests
makes `test_risk.py` non-deletable, hides the fake's responsibilities
behind a domain-specific test file, and causes import-order surprises
under pytest collection.

**Proposed change.** Move `FakeTradingClient` and its `_FakeRawOrder`,
`_account_to_raw`, `_position_to_raw` helpers to `tests/_fakes.py` (or
expose via `conftest.py` as a `broker_factory` fixture, which 5 of the 7
already want anyway). Update the 7 import sites.

**Files affected.** New `tests/_fakes.py`; `tests/test_risk.py` (remove
helpers); 7 test files (rewrite imports).

**Risk.** Low — mechanical rename. Tests already pass and just need
their imports updated.

## 3. `JobResult` defined three times with different shapes

**Problem.** `casino/jobs/reconcile_eod.py:37`, `casino/jobs/news_intraday.py:53`,
and `casino/jobs/earnings_daily.py:73` each define `class JobResult` with
different fields. A future aggregator (e.g., a unified job-orchestration
module) that imports two of them will silently shadow. The names also
break grep — searching for "JobResult" produces three classes that share
nothing.

**Proposed change.** Rename to `ReconcileEodResult`, `NewsIntradayResult`,
`EarningsDailyResult`. Same shape, just disambiguated.

**Files affected.** Each `jobs/*.py` + the corresponding `tests/test_jobs.py`
assertions.

**Risk.** Very low — module-private types, no external consumers.

# P1 — structural drift; address before more execution code lands

## 4. `execution/` mixes primitives and CLI entrypoints

**Problem.** `casino/execution/` contains 11 files. Six are primitives
(`book`, `risk`, `reconcile`, `alpaca_broker`, `paper_clock`, `sim_broker`).
Five have `def main(...)` CLI entrypoints and fire Discord alerts
(`tsmom_runner`, `tsmom_clock_check`, `tsmom_shadow_runner`, `kill_switch`,
plus arguably `risk.py` which exposes `flatten_and_disable`). The
runners look like jobs but live in the same directory as `book.py`,
muddling the boundary that `casino/jobs/` is supposed to enforce.

**Proposed change.** Move the four runner modules to `casino/jobs/`
(or, alternatively, create `casino/execution/runners/`). `kill_switch.py`
is already a thin CLI shim and naturally belongs in jobs. Keep
`risk.flatten_and_disable` as the library function, leave the CLI in
`jobs/kill_switch.py`.

**Files affected.** Four `execution/tsmom_*` files moved or re-homed;
their tests; the RUNBOOK; any scripts that reference `python -m
casino.execution.tsmom_runner` (several `.ps1` wrappers).

**Risk.** Medium. The scheduled-task `.ps1` scripts call modules by
fully qualified name; every one needs to be updated atomically. The
RUNBOOK §A.2 documents these commands explicitly. Worth doing as a
single PR with no behavior changes, just module relocations + script
edits.

## 5. `tsmom_clock_check` imports `assert_paper_account` from `tsmom_runner`

**Problem.** `casino/execution/tsmom_clock_check.py:73` imports
`assert_paper_account` from a peer runner module. The guard is a
defense-in-depth check on `ALPACA_BASE_URL`; it has nothing to do with
TSMOM rebal logic. Re-exporting safety primitives from "the runner
that happens to have defined it first" inverts the layering.

**Proposed change.** Move `assert_paper_account` and `NotPaperAccountError`
to either `casino/execution/alpaca_broker.py` (next to the URL config) or
a tiny `casino/execution/safety.py`. Two runners + future runners import
from one canonical home.

**Files affected.** `casino/execution/alpaca_broker.py` (or new
`safety.py`); `casino/execution/tsmom_runner.py`, `tsmom_clock_check.py`,
`tsmom_shadow_runner.py` (if it touches it).

**Risk.** Low — a function move with no logic change.

## 6. `tsmom_shadow_runner` imports constants and types from `tsmom_runner`

**Problem.** `tsmom_shadow_runner.py:66-71` pulls `DEFAULT_STOP_FRACTION`,
`HISTORY_DAYS`, `TSMOM_UNIVERSE`, and `TargetWeight` out of the live
runner. The shadow runner is the *sim* path and shouldn't depend on the
*live* path; today it works because the live runner happens to also be
where these were first defined.

**Proposed change.** Promote the shared pieces to a `casino/execution/
tsmom_common.py` (or, for `TSMOM_UNIVERSE`, to `signals/ts_momentum.py`
which already owns `load_ohlcv_panel` and `compute_tsmom_panel`). Both
runners import from the new home.

**Files affected.** New `tsmom_common.py` (or extend
`signals/ts_momentum.py`); `tsmom_runner.py`, `tsmom_shadow_runner.py`,
`tsmom_clock_check.py`, `monitoring/dashboard.py` (if it grabs constants).

**Risk.** Low — extraction of constants. The `TargetWeight` dataclass
is referenced in tests but renaming it is moot since the move is purely
relocational.

## 7. `TSMOM_UNIVERSE` duplicated in two places

**Problem.** `casino/execution/tsmom_runner.py:100` and
`casino/jobs/heartbeat.py:57` each define a 10-symbol tuple by hand. If
the universe ever changes (e.g., adding USO replacement), one side will
drift and the heartbeat freshness check will read a different set than
the runner trades.

**Proposed change.** Same as #6 — make `signals/ts_momentum.py` (or
`execution/tsmom_common.py`) the single source.

**Files affected.** Same as #6.

**Risk.** Low. Resolves alongside #6.

## 8. `run_id` constants are scattered

**Problem.** `paper_clock.DEFAULT_RUN_ID = "DiCaprio"` is the live-bot id;
`tsmom_shadow_runner.SHADOW_RUN_ID = "Belfort"` is the sim-bot id;
`monitoring/dashboard.py` hardcodes both strings in multiple places
(lines 745-746, 2104-2105). Adding a third bot means touching at least
four files.

**Proposed change.** Centralize in `paper_clock.py`:
`DEFAULT_RUN_ID = "DiCaprio"`, `SHADOW_RUN_ID = "Belfort"`, plus a
`KNOWN_RUN_IDS: tuple[str, ...]` registry the dashboard iterates over.

**Files affected.** `casino/execution/paper_clock.py` (add constants),
`casino/execution/tsmom_shadow_runner.py` (drop local constant),
`casino/monitoring/dashboard.py` (read from registry).

**Risk.** Low.

## 9. `mypy` overrides disable almost all checking for 9 modules

**Problem.** `pyproject.toml:71-88` lists nine modules (mostly
backtest/signals) with `disable_error_code = ["arg-type", "operator",
"index", "return-value", "no-untyped-def", "call-overload",
"attr-defined", "unreachable", "type-arg"]` plus `disallow_untyped_defs
= false`. That's not "relaxed" — that's mypy off. `arg-type` and
`return-value` in particular would have caught real bugs (e.g., the
pre-existing mypy errors in `heartbeat.py:111-113` that snuck past
because of the broader bypass culture).

**Proposed change.** Tighten in stages. First, drop `arg-type` and
`return-value` from the disable list and fix the resulting errors per
module. Once green, drop `attr-defined` and `index`. The pandas/Decimal
interop issues that motivated the original bypass are real, but
narrow `# type: ignore[...]` annotations at the call site are
preferable to module-wide blanket suppression.

**Files affected.** `pyproject.toml`; each of the nine modules in the
override block (`casino/backtest/*_baseline.py`, `casino/signals/ts_momentum*.py`,
`casino/signals/carry.py`, `casino/execution/sim_broker.py`).

**Risk.** Medium. Some of the original bypasses are load-bearing
(backtrader's `Strategy.params` descriptor genuinely defeats mypy).
Stage the cleanup; don't try to fix all 9 in one pass.

## 10. No test categorization markers (slow/integration/unit)

**Problem.** `pyproject.toml:94-96` registers exactly one marker
(`release_gate`). The 38-file test suite includes fast unit tests, slow
TSMOM walk-forward tests, and integration tests that hit a real DuckDB
schema. There's no way to run `pytest -m unit` in dev or
`pytest -m "not slow"` in pre-commit. Today everything runs every time.

**Proposed change.** Register `unit`, `integration`, `slow`,
`network` markers in `pyproject.toml`. Tag the obvious slow-tests
(`test_walk_forward.py`, `test_vbt_research.py`, `test_bt_validate.py`,
`test_bt_tsmom_validate.py`, `test_tsmom_sensitivity.py`,
`test_carry_baseline.py`, `test_tsmom_shadow_runner.py`) with
`@pytest.mark.slow`. Update CI to run both jobs in parallel.

**Files affected.** `pyproject.toml`; ~7 test files (decorator only).

**Risk.** Low. Markers are additive; nothing breaks if a test is
untagged.

# P2 — cleanup / consistency

## 11. `*Row` naming inconsistent across SQL-row and display-row dataclasses

**Problem.** `DailyPnLRow`, `PaperClockRow`, `KillEventRow`,
`RebalEventRow`, `StoredOrder`, `StoredPosition` model SQLite rows;
`PositionRow`, `LLMCallRow`, `CriterionPanelRow`, `HeadlineRow`,
`EarningsRow`, `CandidateRow` are display/intermediate types that
happen to share the `*Row` suffix. A reader can't tell from the name
which is persisted and which is ephemeral.

**Proposed change.** Reserve `*Row` for SQL rows. Rename the display
dataclasses to drop the suffix or use a different one (`PositionView`,
`LLMCallView`, `HeadlineRecord`, `EarningsRecord`, `CandidateRecord`).

**Files affected.** `monitoring/dashboard.py` (PositionRow, LLMCallRow,
CriterionPanelRow), `jobs/news_intraday.py` (HeadlineRow),
`jobs/earnings_daily.py` (CandidateRow), `signals/pead.py`
(EarningsRow), plus tests.

**Risk.** Low — module-local renames.

## 12. `monitoring/dashboard.py` reaches into `alpaca_broker.BrokerPosition`

**Problem.** Line 36 imports `BrokerPosition` from
`execution.alpaca_broker`. The monitoring layer now knows the Alpaca SDK
type shape, which weakens the "broker is replaceable" abstraction
that `_TradingClientLike` (alpaca_broker.py:112) carefully builds.

**Proposed change.** Either re-export the broker types from
`casino/execution/__init__.py` (a small step that says "this is the
public broker interface") or build a `monitoring`-facing facade in
`execution/views.py` that returns plain dicts.

**Files affected.** `casino/execution/__init__.py` (re-exports) or new
`execution/views.py`; `casino/monitoring/dashboard.py`.

**Risk.** Low. Aesthetic.

## 13. Small utilities (`_round_money`, `_utc_now`) duplicated across modules

**Problem.** `tsmom_runner.py:236-237` and `tsmom_shadow_runner.py:127-128`
both define `_round_money(d: Decimal) -> Decimal`. `_utc_now()` is
re-implemented in `reconcile.py`, `book.py`, `heartbeat.py`,
`tsmom_runner.py`, `tsmom_clock_check.py`, `paper_clock.py`,
`alerts.py`. The duplicates are tiny but multiply with every new module.

**Proposed change.** Create `casino/_internal/time.py` (or extend
`casino/config.py` with a `now_utc()` helper) and a `casino/_internal/
money.py` with `round_money(d)`. Replace duplicates.

**Files affected.** New `_internal/` package; ~7 modules.

**Risk.** Low. Pure deduplication.

## 14. `casino/llm/prompts/` is a subpackage but `casino/data/` is flat

**Problem.** `casino/llm/prompts/` has 3 files (`earnings_score`,
`headline_class`, `thesis_gen`) plus `__init__.py`. `casino/data/`
has 8 files at the top level — `store.py` + 6 `ingest_*.py` + 2
`edgar_*.py` — with no internal grouping. The 6 ingesters share
nothing in common with `store.py` or the edgar client/parser.

**Proposed change.** Move ingesters under `casino/data/ingest/`
(producing `data/ingest/tiingo.py`, `data/ingest/yfinance.py`,
`data/ingest/edgar.py`, `data/ingest/fred.py`,
`data/ingest/transcripts.py`, `data/ingest/transcripts_hf.py`). Keep
`store.py`, `edgar_client.py`, `edgar_parser.py` at `data/` root.

**Files affected.** Six `data/ingest_*.py` files moved; their tests;
any `.ps1` wrapper that invokes `python -m casino.data.ingest_*`.

**Risk.** Medium — same `.ps1` invocation concern as #4. Worth pairing
with #4 in a single "rename pass" PR.

## 15. `tests/` is flat with 38 files

**Problem.** No subdirectory structure. Finding "all tests that touch
execution" requires grep. Adding mirrored layout (`tests/execution/`,
`tests/signals/`, `tests/jobs/`, `tests/data/`, `tests/llm/`,
`tests/backtest/`) would scale with the package.

**Proposed change.** Mirror the package layout. With pytest's
auto-discovery this is purely a filesystem move; no test code changes
beyond the `from tests.test_risk import FakeTradingClient` lines (which
are getting fixed in #2 anyway).

**Files affected.** All 37 test files moved (conftest.py stays at
`tests/` root); CI config (none, pytest auto-discovers).

**Risk.** Low. Mechanical move; pytest collection is path-agnostic.

# Logical dependency chain

Recommended order:

1. **#2 (FakeTradingClient)** — pure mechanical, unblocks future test
   refactors, no risk.
2. **#3 (JobResult rename)** — trivial, removes a future footgun.
3. **#1 (heartbeat DuckDB)** — fixes a documented invariant; small.
4. **#5 (assert_paper_account move)** + **#6 (tsmom_common extraction)**
   + **#7 (TSMOM_UNIVERSE dedup)** + **#8 (run_id registry)** — these
   four are the same shape of work (move constants/helpers to a sensible
   home). Best done as one PR.
5. **#4 (execution → jobs migration)** + **#14 (data/ingest/ subpackage)**
   — both touch `.ps1` wrappers and RUNBOOK; best done atomically as a
   "rename pass" PR.
6. **#10 (test markers)** — additive; can land any time.
7. **#15 (tests/ layout)** — pairs naturally with #2 cleanup.
8. **#9 (mypy tightening)** — multi-PR effort; queue last.
9. **#11–#13 (naming, abstraction, util dedup)** — cleanup; opportunistic.

# Risks and mitigations

- **The `.ps1` wrappers reference fully-qualified module paths.** Any
  rename of an `execution/tsmom_*` or `data/ingest_*` module touches
  Windows Task Scheduler. Mitigation: pair every module move with the
  corresponding `.ps1` edit and RUNBOOK update; manually run
  `.\daily_branch_c.ps1` before pushing.
- **`pyproject.toml`'s mypy override block is load-bearing.** Tightening
  it in one pass will produce hundreds of errors. Mitigation: drop one
  disabled code per PR, fix the resulting errors, move on.
- **`FakeTradingClient` is implicitly the test-suite's broker mock.**
  Moving it could change import ordering in subtle ways. Mitigation:
  run the full test suite (`uv run pytest`) after the move and before
  cutting the PR.

# Appendix — observations not promoted to action items

- `casino/execution/kill_switch.py` is a 62-line CLI shim around
  `risk.flatten_and_disable` / `risk.re_enable_trading`. It's justified
  per its own docstring ("library callers want the function; operators
  want the command"). Leave it; just relocate to `jobs/` in #4.
- `signals/ts_momentum.py` and `signals/ts_momentum_regime.py` are a
  base signal + regime overlay; the naming reflects layering and is
  fine.
- The `release_gate` pytest marker is well-used (`test_backtest_
  no_lookahead.py`); good model for adding `slow` / `integration`.
- `casino/__init__.py` declares `__version__ = "0.1.0"`; consider
  adopting it as the source of truth for any future deploy tagging.

</PRD>
