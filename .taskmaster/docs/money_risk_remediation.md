# Money-risk remediation — 2026-05-11

Follow-up to `.taskmaster/docs/structure_review.md`. That review surfaced
15 structural issues; **three of them have a path to losing real money**
and are addressed in this PRD. The other 12 are engineering-time debt
and queued separately.

## Why these three

The system is single-developer paper-trading on its way to live cash.
Engineering-time debt is acceptable; **arithmetic and config drift on the
signal path are not**. Every issue here is on the seam between the
signal that decides weights and the broker that executes them.

| Task | Risk | Mechanism |
|------|------|-----------|
| 44 — mypy on signals | High | Untyped weight/return computation; one bad cast or NaN slip silently changes sizing. mypy is off for the signal modules today (`arg-type`, `return-value`, `attr-defined`, …). |
| 45 — TSMOM_UNIVERSE single source | Medium | The constant is duplicated in `execution/tsmom_runner.py` and `jobs/heartbeat.py`. If the runner's universe is updated and heartbeat's isn't, you trade a position the freshness gate isn't watching. |
| 46 — `_round_money` single source | Low | The helper is duplicated in `execution/tsmom_runner.py` and `execution/tsmom_shadow_runner.py`. The day one drifts (e.g., `ROUND_DOWN` for conservatism on one side), live and shadow sizing diverge and contaminate the Belfort-vs-DiCaprio comparison. |

## Task 46 — `_round_money` canonical helper

**Problem.** `_round_money(d: Decimal) -> Decimal` is defined identically
in `casino/execution/tsmom_runner.py:236-237` and
`casino/execution/tsmom_shadow_runner.py:127-128`. Both use
`d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)`.

**Proposed change.** Create `casino/_money.py` with a public
`round_money(d: Decimal) -> Decimal`. Replace both private duplicates
with imports. Add a unit test pinning the rounding policy (HALF_UP, two
decimals) — this is the contract that has to hold between runners.

**Files affected.**
- New: `casino/_money.py`
- New: `tests/test_money.py`
- Modified: `casino/execution/tsmom_runner.py` (remove local `_round_money`, import)
- Modified: `casino/execution/tsmom_shadow_runner.py` (same)

**Risk level.** Very low. Pure-function move with pinned semantics.

**Done when.**
- `grep "_round_money" casino/` returns zero matches.
- `tests/test_money.py` passes with explicit cases (e.g., `Decimal("0.125") → Decimal("0.13")`).
- Existing test suites (`test_tsmom_runner.py`, `test_tsmom_shadow_runner.py`) still pass unchanged.

## Task 45 — `TSMOM_UNIVERSE` single source

**Problem.** The 10-symbol tuple is hardcoded in two places:
- `casino/execution/tsmom_runner.py:100`
- `casino/jobs/heartbeat.py:57`

`casino/execution/tsmom_shadow_runner.py:69` already imports
`TSMOM_UNIVERSE` from `tsmom_runner`, which would also be fixed by this
change.

If the operator adds, removes, or substitutes a ticker (e.g., replacing
`USO` with a newer commodity ETF) and updates only the runner, the
heartbeat's OHLCV-freshness gate would silently monitor the wrong set.
The system would trade a ticker the freshness gate isn't watching; a
yfinance ingest failure on the new ticker would not trip the daily
warning.

**Proposed change.** Promote `TSMOM_UNIVERSE` to
`casino/signals/ts_momentum.py` (which already owns
`load_ohlcv_panel` and `compute_tsmom_panel` — the canonical TSMOM
signal module). All three consumers import from there.

**Files affected.**
- Modified: `casino/signals/ts_momentum.py` (define `TSMOM_UNIVERSE`)
- Modified: `casino/execution/tsmom_runner.py` (remove local definition, import)
- Modified: `casino/execution/tsmom_shadow_runner.py` (rewrite import path)
- Modified: `casino/jobs/heartbeat.py` (remove local definition, import)

**Risk level.** Low. Pure constant relocation; the tuple value is
unchanged.

**Done when.**
- `grep -E "^TSMOM_UNIVERSE" casino/` returns exactly one match
  (`signals/ts_momentum.py`).
- `tests/test_heartbeat.py` still references `heartbeat.TSMOM_UNIVERSE`
  via the module attribute (or is updated to import the canonical one).
  A test asserts the three known consumers agree.

## Task 44 — Tighten mypy on signal modules

**Problem.** `pyproject.toml:71-88` declares the following bypass for
nine modules:

```toml
[[tool.mypy.overrides]]
module = [
    "casino.backtest.sue_baseline",
    "casino.backtest.composite_baseline",
    "casino.backtest.finbert_baseline",
    "casino.backtest.tsmom_baseline",
    "casino.backtest.carry_baseline",
    "casino.signals.ts_momentum",
    "casino.signals.ts_momentum_regime",
    "casino.signals.carry",
    "casino.execution.sim_broker",
]
disallow_untyped_defs = false
warn_return_any = false
warn_unused_ignores = false
disable_error_code = [
    "arg-type", "operator", "index", "return-value",
    "no-untyped-def", "call-overload", "attr-defined",
    "unreachable", "type-arg",
]
```

That's not "relaxed" — that's mypy off. The three modules in the
`casino.signals.*` block compute the **weight vector and per-symbol
target dollars the runner trades on**. A NaN-contaminated panel or a
wrong-type return path can silently misweight a position. The disabled
checks include the two most useful: `arg-type` (passes the wrong type
into a function) and `return-value` (returns the wrong shape).

**Scope of this task.** Tighten mypy on the three **signal** modules
only:

- `casino/signals/ts_momentum.py`
- `casino/signals/ts_momentum_regime.py`
- `casino/signals/carry.py`

Specifically:

1. Move those three modules out of the broad-disable block into a
   narrower one that re-enables `arg-type` and `return-value`.
2. Fix the resulting errors with proper type hints, narrowed casts via
   `typing.cast`, or **narrow per-line `# type: ignore[code]`
   annotations at the call site** where pandas/Decimal interop legitimately
   defeats the checker. No module-level blanket suppression.
3. Backtest baselines and `sim_broker` stay on the legacy override for
   now — they are correctness-critical too but have heavier pandas
   interop pain; that's a separate follow-up.

**Files affected.**
- Modified: `pyproject.toml` (split the override block).
- Modified: `casino/signals/ts_momentum.py` (type fixes).
- Modified: `casino/signals/ts_momentum_regime.py` (type fixes).
- Modified: `casino/signals/carry.py` (type fixes).
- Possibly modified: tests if any rely on the previous looseness.

**Risk level.** Medium. Pandas interop is genuinely hard to type. The
mitigation is to use narrow `# type: ignore` annotations rather than
silently broadening the bypass — every ignore is a documented exception
the reader can audit.

**Done when.**
- `uv run mypy casino/signals/ts_momentum.py casino/signals/ts_momentum_regime.py casino/signals/carry.py` exits 0 with `arg-type` and `return-value` checks **enabled**.
- The legacy `[[tool.mypy.overrides]]` block in `pyproject.toml` no
  longer includes those three modules.
- All existing tests still pass.

## Logical order

1. **Task 46** first. Smallest change, unrelated to the other two,
   establishes the `casino/_money.py` pattern that #45 may reference.
2. **Task 45** second. Touches `tsmom_runner.py`, which #44 may also
   touch (signals/ts_momentum.py imports change). Sequencing avoids
   merge churn.
3. **Task 44** last. The longest of the three; touches type system; do
   it on a clean tree.

## Risks and mitigations

- **#44 may surface latent bugs.** Goal: surface them, not hide them.
  Every new mypy error is either a bug to fix or a documented `# type:
  ignore[code]` exception. No module-wide bypass for newly-found pain.
- **#45 changes import paths.** The `.ps1` wrappers do not reference
  `TSMOM_UNIVERSE` directly (they invoke modules), so scripts are
  unaffected. Tests use `heartbeat.TSMOM_UNIVERSE` as an attribute,
  which keeps working through re-export.
- **#46 is the safest.** The only meaningful risk is forgetting an
  import site; one grep at the end catches that.

## Appendix — what's NOT in this PRD

Items from `structure_review.md` deferred to a later pass:

- P0 #1 (heartbeat opens DuckDB directly) — invariant violation, but
  read-only and not a money risk; queue with the
  data/ingest/ subpackage refactor.
- P0 #2 (FakeTradingClient cross-imports) — engineering hygiene.
- P0 #3 (`JobResult` name collision) — engineering hygiene.
- P1 #4–8, #10 — structural drift; not on the money path.
- P2 #11–15 — naming and layout polish.
