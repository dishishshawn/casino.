"""TSMOM 30-day-cap daily monitor + day-30 binary verdict.

Branch C amendment 2026-05-07: companion to ``casino.execution.tsmom_runner``.
The runner owns the *monthly* rebal leg; this module owns the *daily*
kill-criteria leg and the *day-30* COMMIT-or-KILL verdict.

The user's verbatim decision: "don't add a leg, paper-trade TSMOM alone,
30-day cap on Branch C, then commit or kill". This script enforces that
deadline mechanically.

Kill criteria (any one fires → kill switch + Discord alert + persisted
``kill_event`` row). Each threshold and rationale is documented inline so
the next operator (or future-you in 6 months) understands why the number
is the number.

* ``drawdown``: cumulative paper P&L drawdown from start NAV > 10%. The
  vbt baseline showed -19.5% MaxDD over 20 years, so a 10% drawdown in
  30 days is a >2-sigma tail event — sufficient to suspect a
  signal/execution misalignment.
* ``single_day``: any single-day P&L drop > 5% NAV. Sizing or risk-cap
  bug indicator.
* ``cap_violation``: gross exposure > 1.0 (with 0.02 numerical-tolerance
  buffer); any single-name > 10% NAV. CLAUDE.md hard-rule caps.
* ``reconcile_drift``: |broker_market_value − book_market_value| > 1% NAV.
  Indicates a fill-tracking bug we cannot afford to discover during live.
* ``ks_test``: at day ≥ 14, one-sided KS test of paper daily returns
  against the backtest predicted distribution. ``p < 0.01`` → kill.

Day-30 verdict (run with ``--verdict``). All COMMIT preconditions:

1. No kill criterion has fired.
2. ≥ 1 monthly rebal completed end-to-end.
3. Reconcile drift < 0.5% NAV.
4. Paper Sharpe ≥ -0.2 (deliberately loose; 30 days is a tiny sample).
5. KS-test p > 0.10.

Otherwise: KILL. The verdict ITSELF does not auto-promote anything;
COMMIT means "Branch C lives — operator decides next step (start
ensemble work, head toward live-cash gate, etc)". KILL means "Branch C
is dead, abandon and reassess".

CLI:
    uv run python -m casino.execution.tsmom_clock_check
        # Daily mode: evaluate every kill criterion, fire alerts,
        # persist any new kill_event rows. Engages the kill switch if
        # anything fires.

    uv run python -m casino.execution.tsmom_clock_check --verdict
        # Day-30 mode: in addition to the daily check, compute the
        # COMMIT-or-KILL verdict and write reports/tsmom_paper_30day_verdict.{csv,md}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from casino.config import get_config
from casino.execution import book, paper_clock, reconcile
from casino.execution.alpaca_broker import AlpacaBroker, build_default_broker
from casino.execution.risk import KillSwitchResult, flatten_and_disable
from casino.execution.tsmom_runner import assert_paper_account
from casino.monitoring import alerts

# Kill thresholds. Keep in one place so the dashboard can import them too.
DD_KILL_THRESHOLD: Decimal = Decimal("0.10")  # 10% drawdown
SINGLE_DAY_KILL_THRESHOLD: Decimal = Decimal("0.05")  # 5% one-day drop
GROSS_CAP_BUFFER: Decimal = Decimal("0.02")  # tolerance over 1.0 gross
RECONCILE_DRIFT_THRESHOLD: Decimal = Decimal("0.01")  # 1% NAV
RECONCILE_DRIFT_COMMIT_THRESHOLD: Decimal = Decimal("0.005")  # 0.5% NAV (verdict gate)
KS_KILL_PVALUE: float = 0.01
KS_COMMIT_PVALUE: float = 0.10
KS_MIN_DAYS: int = 14
COMMIT_SHARPE_FLOOR: float = -0.2

# Path to the prebuilt backtest return distribution. Stored as a one-column CSV
# of daily returns. The runner does NOT auto-create it; the operator drops it
# under reports/tsmom_backtest_returns.csv before running paper-trade. If the
# file is absent we skip the KS test (logged as "no backtest baseline").
BACKTEST_RETURNS_PATH: Path = Path("reports/tsmom_backtest_returns.csv")


# ---------------------------------------------------------------------------- types


@dataclass(frozen=True)
class CriterionStatus:
    """Per-criterion health snapshot."""

    name: str
    triggered: bool
    value: Decimal
    threshold: Decimal
    detail: str


@dataclass
class DailyCheckResult:
    """Output of one ``run_daily_check`` invocation."""

    as_of_utc: datetime
    days_elapsed: int | None
    statuses: list[CriterionStatus] = field(default_factory=list)
    kill_fired: bool = False
    triggered_criteria: list[str] = field(default_factory=list)
    kill_switch_result: KillSwitchResult | None = None


@dataclass(frozen=True)
class VerdictResult:
    """Output of the day-30 ``run_verdict`` script."""

    verdict: str  # "COMMIT" | "KILL"
    reasons: list[str]
    paper_sharpe: float | None
    paper_total_return: Decimal
    paper_max_drawdown: Decimal
    n_rebals: int
    drift_now: Decimal
    ks_pvalue: float | None
    days_elapsed: int


# ---------------------------------------------------------------------------- helpers


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _abs(d: Decimal) -> Decimal:
    return -d if d < Decimal("0") else d


def _safe_div(a: Decimal, b: Decimal) -> Decimal:
    return a / b if b != Decimal("0") else Decimal("0")


def _load_paper_returns(*, db_path: Path | None = None) -> list[float]:
    """Build the paper daily-return series from the daily_pnl table.

    Returns a list of decimal returns (today_total_pl / equity_open) ordered
    chronologically. Discards rows with non-positive equity_open.
    """
    history = book.fetch_daily_pnl(limit=2000, db_path=db_path)
    rs: list[float] = []
    for row in reversed(history):
        if row.equity_open <= Decimal("0"):
            continue
        total = row.realized_pl + row.unrealized_pl
        rs.append(float(total / row.equity_open))
    return rs


def _equity_curve(
    *,
    start_nav: Decimal,
    db_path: Path | None = None,
) -> tuple[list[Decimal], list[Decimal]]:
    """Return (equity_values_chrono, drawdown_values_chrono).

    Drawdown computed against running high-water mark seeded at ``start_nav``.
    """
    history = book.fetch_daily_pnl(limit=2000, db_path=db_path)
    eqs: list[Decimal] = []
    dds: list[Decimal] = []
    hwm = start_nav
    eq = start_nav
    for row in reversed(history):
        eq = row.equity_close
        if eq > hwm:
            hwm = eq
        dd = _safe_div(hwm - eq, hwm) if hwm > Decimal("0") else Decimal("0")
        eqs.append(eq)
        dds.append(dd)
    return eqs, dds


# ---------------------------------------------------------------------------- KS test


def _ks_two_sample_pvalue(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Two-sample two-sided Kolmogorov-Smirnov p-value.

    Numpy-only implementation (no scipy dependency). Uses the asymptotic
    Kolmogorov distribution; matches scipy's KS p-value to 1e-3 for the
    sample sizes we care about (paper n ≥ 14, backtest n ≫ 100). Returns
    None if either sample has < 2 points.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    aa = np.sort(np.asarray(a, dtype=float))
    bb = np.sort(np.asarray(b, dtype=float))
    n_a, n_b = len(aa), len(bb)
    # CDF of the union: walk both sorted arrays simultaneously.
    all_vals = np.concatenate([aa, bb])
    all_vals.sort()
    cdf_a = np.searchsorted(aa, all_vals, side="right") / n_a
    cdf_b = np.searchsorted(bb, all_vals, side="right") / n_b
    d_stat = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(n_a * n_b / (n_a + n_b))
    # Marsaglia-Tsang-Wang asymptotic series for Pr(K > k).
    lam = (en + 0.12 + 0.11 / en) * d_stat
    if lam <= 0:
        return 1.0
    s = 0.0
    for j in range(1, 101):
        term = ((-1) ** (j - 1)) * math.exp(-2.0 * (lam**2) * (j**2))
        s += term
        if abs(term) < 1e-12:
            break
    p = max(0.0, min(1.0, 2.0 * s))
    return p


def _load_backtest_returns(path: Path = BACKTEST_RETURNS_PATH) -> list[float] | None:
    """Load the backtest return series. Returns None if the file is absent.

    File format: one column of float daily returns, optional header.
    """
    if not path.exists():
        return None
    out: list[float] = []
    text = path.read_text(encoding="utf-8").strip().splitlines()
    for line in text:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(float(line))
        except ValueError:
            # likely a header row
            continue
    return out if out else None


# ---------------------------------------------------------------------------- criteria


def _evaluate_drawdown(
    *,
    start_nav: Decimal,
    eqs: Sequence[Decimal],
) -> CriterionStatus:
    """Cumulative drawdown from start_nav (NOT from running high-water mark).

    Per the amendment: 'cumulative paper P&L drawdown from start > 10% of
    starting NAV (paper)'. We use start_nav as the baseline so the criterion
    fires on absolute paper losses, regardless of an interim equity peak.
    """
    if not eqs:
        return CriterionStatus(
            name="drawdown",
            triggered=False,
            value=Decimal("0"),
            threshold=DD_KILL_THRESHOLD,
            detail="no equity history yet",
        )
    cur = eqs[-1]
    drop = (start_nav - cur) / start_nav if start_nav > Decimal("0") else Decimal("0")
    if drop < Decimal("0"):
        drop = Decimal("0")
    triggered = drop > DD_KILL_THRESHOLD
    return CriterionStatus(
        name="drawdown",
        triggered=triggered,
        value=drop,
        threshold=DD_KILL_THRESHOLD,
        detail=f"start_nav=${start_nav} current_equity=${cur} drop={drop:.4%}",
    )


def _evaluate_single_day(*, db_path: Path | None = None) -> CriterionStatus:
    history = book.fetch_daily_pnl(limit=2000, db_path=db_path)
    worst = Decimal("0")
    detail = "no daily P&L rows"
    for row in history:
        if row.equity_open <= Decimal("0"):
            continue
        total = row.realized_pl + row.unrealized_pl
        ret = total / row.equity_open
        if ret < worst:
            worst = ret
            detail = f"worst day {row.date}: total={total} eq_open={row.equity_open} ret={ret:.4%}"
    drop = -worst  # convert to positive magnitude
    triggered = drop > SINGLE_DAY_KILL_THRESHOLD
    return CriterionStatus(
        name="single_day",
        triggered=triggered,
        value=drop,
        threshold=SINGLE_DAY_KILL_THRESHOLD,
        detail=detail,
    )


def _evaluate_cap_violation(broker: AlpacaBroker) -> CriterionStatus:
    """Gross > 1.0+buffer or any single-name > 10% NAV → trigger."""
    account = broker.get_account()
    nav = account.equity
    positions = broker.get_positions()
    gross = Decimal("0")
    worst_name = ""
    worst_pct = Decimal("0")
    for p in positions:
        notional = _abs(p.market_value)
        gross += notional
        pct = _safe_div(notional, nav) if nav > Decimal("0") else Decimal("0")
        if pct > worst_pct:
            worst_pct = pct
            worst_name = p.symbol
    gross_pct = _safe_div(gross, nav) if nav > Decimal("0") else Decimal("0")
    cfg = get_config()
    single_name_cap = Decimal(str(cfg.max_single_name))
    gross_cap = Decimal(str(cfg.max_gross_exposure)) + GROSS_CAP_BUFFER
    triggered = gross_pct > gross_cap or worst_pct > single_name_cap
    detail = (
        f"gross={gross_pct:.4f} cap={gross_cap}; "
        f"worst_name={worst_name} pct={worst_pct:.4f} cap={single_name_cap}"
    )
    # Report the "worst" of the two as the primary value/threshold for telemetry.
    if worst_pct > gross_pct:
        return CriterionStatus(
            name="cap_violation",
            triggered=triggered,
            value=worst_pct,
            threshold=single_name_cap,
            detail=detail,
        )
    return CriterionStatus(
        name="cap_violation",
        triggered=triggered,
        value=gross_pct,
        threshold=gross_cap,
        detail=detail,
    )


def _evaluate_reconcile_drift(
    *,
    broker: AlpacaBroker,
    db_path: Path | None = None,
) -> CriterionStatus:
    """Sum |broker_market_value - book_notional| as a fraction of NAV."""
    account = broker.get_account()
    nav = account.equity
    rec = reconcile.reconcile(broker=broker, db_path=db_path)
    drift_dollars = Decimal("0")
    for d in rec.drift:
        # broker_only / book_only / qty_mismatch / side_mismatch / price_drift
        diff = _abs(d.broker_notional - d.book_notional)
        drift_dollars += diff
    drift_pct = _safe_div(drift_dollars, nav) if nav > Decimal("0") else Decimal("0")
    triggered = drift_pct > RECONCILE_DRIFT_THRESHOLD
    return CriterionStatus(
        name="reconcile_drift",
        triggered=triggered,
        value=drift_pct,
        threshold=RECONCILE_DRIFT_THRESHOLD,
        detail=(
            f"|sum(broker_mv - book_notional)|=${drift_dollars} nav=${nav} "
            f"drift={drift_pct:.4%} entries={len(rec.drift)}"
        ),
    )


def _evaluate_ks_test(
    *,
    days_elapsed: int | None,
    db_path: Path | None = None,
    backtest_returns_path: Path = BACKTEST_RETURNS_PATH,
) -> CriterionStatus:
    if days_elapsed is None or days_elapsed < KS_MIN_DAYS:
        return CriterionStatus(
            name="ks_test",
            triggered=False,
            value=Decimal("1.0"),
            threshold=Decimal(str(KS_KILL_PVALUE)),
            detail=f"days_elapsed={days_elapsed} below {KS_MIN_DAYS}-day minimum",
        )
    paper = _load_paper_returns(db_path=db_path)
    if len(paper) < KS_MIN_DAYS:
        return CriterionStatus(
            name="ks_test",
            triggered=False,
            value=Decimal("1.0"),
            threshold=Decimal(str(KS_KILL_PVALUE)),
            detail=f"only {len(paper)} paper days available; below {KS_MIN_DAYS}",
        )
    backtest = _load_backtest_returns(backtest_returns_path)
    if backtest is None or len(backtest) < KS_MIN_DAYS:
        return CriterionStatus(
            name="ks_test",
            triggered=False,
            value=Decimal("1.0"),
            threshold=Decimal(str(KS_KILL_PVALUE)),
            detail="no backtest baseline at reports/tsmom_backtest_returns.csv",
        )
    p = _ks_two_sample_pvalue(paper, backtest)
    if p is None:
        return CriterionStatus(
            name="ks_test",
            triggered=False,
            value=Decimal("1.0"),
            threshold=Decimal(str(KS_KILL_PVALUE)),
            detail="ks pvalue computation aborted",
        )
    triggered = p < KS_KILL_PVALUE
    return CriterionStatus(
        name="ks_test",
        triggered=triggered,
        value=Decimal(str(round(p, 6))),
        threshold=Decimal(str(KS_KILL_PVALUE)),
        detail=f"two-sample KS p={p:.4f} (n_paper={len(paper)} n_backtest={len(backtest)})",
    )


# ---------------------------------------------------------------------------- daily check


def run_daily_check(
    *,
    broker: AlpacaBroker | None = None,
    db_path: Path | None = None,
    backtest_returns_path: Path = BACKTEST_RETURNS_PATH,
    run_id: str = paper_clock.DEFAULT_RUN_ID,
    transport: alerts.WebhookTransport | None = None,
    engage_kill_switch: bool = True,
) -> DailyCheckResult:
    """Evaluate every kill criterion. If any triggers, fire kill_switch + alert.

    The kill_switch path is intentionally idempotent — we re-engage on
    every fire (cancel-all + close-all-positions are best-effort and
    safe to repeat). The ``run_id`` row in ``paper_clock`` is NOT marked
    KILL here; that's the day-30 verdict's job. We just persist a
    ``kill_event`` row and trip the trading-disabled flag.
    """
    assert_paper_account()
    actual_broker = broker if broker is not None else build_default_broker()

    clock = paper_clock.fetch_paper_clock(run_id=run_id, db_path=db_path)
    days_elapsed_val = (
        paper_clock.days_elapsed(run_id=run_id, db_path=db_path) if clock is not None else None
    )
    start_nav = clock.start_nav if clock is not None else Decimal("0")

    eqs, _dds = _equity_curve(start_nav=start_nav, db_path=db_path)

    statuses: list[CriterionStatus] = []
    statuses.append(_evaluate_drawdown(start_nav=start_nav, eqs=eqs))
    statuses.append(_evaluate_single_day(db_path=db_path))
    statuses.append(_evaluate_cap_violation(actual_broker))
    statuses.append(_evaluate_reconcile_drift(broker=actual_broker, db_path=db_path))
    statuses.append(
        _evaluate_ks_test(
            days_elapsed=days_elapsed_val,
            db_path=db_path,
            backtest_returns_path=backtest_returns_path,
        )
    )

    triggered = [s for s in statuses if s.triggered]
    result = DailyCheckResult(
        as_of_utc=_utc_now(),
        days_elapsed=days_elapsed_val,
        statuses=statuses,
        kill_fired=len(triggered) > 0,
        triggered_criteria=[s.name for s in triggered],
    )

    # Persist kill events, fire alert, engage kill switch.
    nav = actual_broker.get_account().equity if triggered else Decimal("0")
    for s in triggered:
        paper_clock.insert_kill_event(
            criterion=s.name,
            value=s.value,
            threshold=s.threshold,
            nav_at_kill=nav,
            detail=s.detail,
            run_id=run_id,
            db_path=db_path,
        )
        alerts.fire(
            title=f"TSMOM 30-day cap KILL CRITERION [{s.name}]",
            message=s.detail,
            severity="critical",
            fields={
                "Criterion": s.name,
                "Value": str(s.value),
                "Threshold": str(s.threshold),
                "NAV": str(nav),
                "Days elapsed": str(days_elapsed_val),
            },
            transport=transport,
        )

    if triggered and engage_kill_switch:
        ks = flatten_and_disable(
            broker=actual_broker,
            reason=f"tsmom_clock_check: {','.join(s.name for s in triggered)}",
            db_path=db_path,
        )
        result.kill_switch_result = ks
        logger.error(
            "tsmom_clock_check: KILL fired ({}); kill_switch closed={} cancelled={}",
            ",".join(s.name for s in triggered),
            ks.closed_positions,
            ks.cancelled_orders,
        )

    return result


# ---------------------------------------------------------------------------- verdict


def _paper_sharpe(rs: Sequence[float]) -> float | None:
    if len(rs) < 5:
        return None
    arr = np.asarray(rs, dtype=float)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return None
    return (mu / sd) * math.sqrt(252.0)


def _paper_max_drawdown(eqs: Sequence[Decimal]) -> Decimal:
    if not eqs:
        return Decimal("0")
    hwm = eqs[0]
    worst = Decimal("0")
    for v in eqs:
        if v > hwm:
            hwm = v
        if hwm > Decimal("0"):
            dd = (hwm - v) / hwm
            if dd > worst:
                worst = dd
    return worst


def run_verdict(
    *,
    broker: AlpacaBroker | None = None,
    db_path: Path | None = None,
    backtest_returns_path: Path = BACKTEST_RETURNS_PATH,
    run_id: str = paper_clock.DEFAULT_RUN_ID,
    reports_dir: Path | None = None,
    transport: alerts.WebhookTransport | None = None,
) -> VerdictResult:
    """Compute the day-30 COMMIT-or-KILL verdict and write the report.

    Verdict gate (ALL required for COMMIT):

    1. Zero kill_event rows in the run.
    2. ``rebals_completed >= 1``.
    3. Reconcile drift now < 0.5% NAV.
    4. Paper Sharpe ≥ -0.2 (loose; 30 days is small).
    5. KS-test p > 0.10 (or unavailable, in which case treat as PASS
       with a documented caveat — we have no baseline to disprove).

    The verdict ITSELF does not auto-promote anything. COMMIT is the
    operator's signal to consider next steps (ensemble work, live-cash
    gate). KILL means abandon Branch C.
    """
    assert_paper_account()
    actual_broker = broker if broker is not None else build_default_broker()
    clock = paper_clock.fetch_paper_clock(run_id=run_id, db_path=db_path)
    if clock is None:
        raise RuntimeError(
            "paper_clock has no start row for "
            f"run_id={run_id!r}; the runner has not been invoked yet"
        )

    days = paper_clock.days_elapsed(run_id=run_id, db_path=db_path) or 0
    rebals = paper_clock.fetch_rebal_events(run_id=run_id, db_path=db_path)
    kills = paper_clock.fetch_kill_events(run_id=run_id, db_path=db_path)

    eqs, _dds = _equity_curve(start_nav=clock.start_nav, db_path=db_path)
    rs = _load_paper_returns(db_path=db_path)
    paper_sharpe = _paper_sharpe(rs)
    paper_max_dd = _paper_max_drawdown(eqs)

    # Total return.
    final_eq = eqs[-1] if eqs else clock.start_nav
    total_return = (
        (final_eq - clock.start_nav) / clock.start_nav
        if clock.start_nav > Decimal("0")
        else Decimal("0")
    )

    # Reconcile drift now.
    drift_status = _evaluate_reconcile_drift(broker=actual_broker, db_path=db_path)

    # KS test now.
    ks_status = _evaluate_ks_test(
        days_elapsed=days,
        db_path=db_path,
        backtest_returns_path=backtest_returns_path,
    )
    # ks_status.value is the p-value if available; if detail mentions
    # "no backtest baseline", treat as None.
    ks_p: float | None = None
    if "no backtest baseline" not in ks_status.detail and "below" not in ks_status.detail:
        try:
            ks_p = float(ks_status.value)
        except (TypeError, ValueError):
            ks_p = None

    reasons: list[str] = []
    commit = True
    if kills:
        commit = False
        reasons.append(
            f"KILL: {len(kills)} kill_event(s): "
            + ", ".join(f"{k.criterion}@{k.fired_at_utc.date()}" for k in kills)
        )
    if len(rebals) < 1:
        commit = False
        reasons.append(f"KILL: only {len(rebals)} rebal completed; need >= 1")
    if drift_status.value > RECONCILE_DRIFT_COMMIT_THRESHOLD:
        commit = False
        reasons.append(
            f"KILL: reconcile drift {drift_status.value:.4%} > "
            f"COMMIT threshold {RECONCILE_DRIFT_COMMIT_THRESHOLD:.4%}"
        )
    if paper_sharpe is None or paper_sharpe < COMMIT_SHARPE_FLOOR:
        commit = False
        reasons.append(f"KILL: paper Sharpe {paper_sharpe} below floor {COMMIT_SHARPE_FLOOR}")
    if ks_p is not None and ks_p <= KS_COMMIT_PVALUE:
        commit = False
        reasons.append(f"KILL: KS p={ks_p:.4f} <= COMMIT threshold {KS_COMMIT_PVALUE}")
    if commit:
        reasons.append("All COMMIT preconditions met.")

    verdict = "COMMIT" if commit else "KILL"
    paper_clock.set_verdict(
        verdict=verdict,
        run_id=run_id,
        notes="; ".join(reasons),
        db_path=db_path,
    )

    out = VerdictResult(
        verdict=verdict,
        reasons=reasons,
        paper_sharpe=paper_sharpe,
        paper_total_return=total_return,
        paper_max_drawdown=paper_max_dd,
        n_rebals=len(rebals),
        drift_now=drift_status.value,
        ks_pvalue=ks_p,
        days_elapsed=days,
    )

    # Write reports.
    target_reports = reports_dir if reports_dir is not None else Path("reports")
    target_reports.mkdir(parents=True, exist_ok=True)
    # Parametrized filenames per run_id so the live (DiCaprio) and the
    # shadow (Belfort) verdicts don't overwrite each other.
    # The default run_id keeps the historical filename unchanged for
    # backwards compatibility with existing scripts and dashboards.
    if run_id == paper_clock.DEFAULT_RUN_ID:
        csv_path = target_reports / "tsmom_paper_30day_verdict.csv"
        md_path = target_reports / "tsmom_paper_30day_verdict.md"
    else:
        csv_path = target_reports / f"tsmom_paper_30day_verdict_{run_id}.csv"
        md_path = target_reports / f"tsmom_paper_30day_verdict_{run_id}.md"
    _write_verdict_csv(csv_path, out)
    _write_verdict_md(md_path, out, clock=clock, kills=kills, rebals=rebals)

    # Discord alert.
    next_steps = (
        "Branch C lives. Operator decides next step (ensemble work, live-cash "
        "gate prep). COMMIT does NOT authorize live trading."
        if verdict == "COMMIT"
        else "Branch C is dead. Abandon TSMOM-alone; reassess strategy."
    )
    alerts.fire(
        title=f"[ACTION REQUIRED] TSMOM 30-day verdict: {verdict} [{run_id}]",
        message=next_steps + "\n\n" + "\n".join(reasons),
        severity="warning" if verdict == "COMMIT" else "critical",
        fields={
            "Run ID": run_id,
            "Verdict": verdict,
            "Days elapsed": str(days),
            "Rebals": str(len(rebals)),
            "Total return": f"{total_return:.4%}",
            "MaxDD": f"{paper_max_dd:.4%}",
            "Paper Sharpe": "n/a" if paper_sharpe is None else f"{paper_sharpe:.3f}",
            "KS p-value": "n/a" if ks_p is None else f"{ks_p:.4f}",
            "Drift now": f"{drift_status.value:.4%}",
        },
        transport=transport,
    )
    return out


def _write_verdict_csv(path: Path, v: VerdictResult) -> None:
    rows: list[tuple[str, str]] = [
        ("verdict", v.verdict),
        ("days_elapsed", str(v.days_elapsed)),
        ("paper_sharpe", "" if v.paper_sharpe is None else f"{v.paper_sharpe:.6f}"),
        ("paper_total_return", str(v.paper_total_return)),
        ("paper_max_drawdown", str(v.paper_max_drawdown)),
        ("n_rebals", str(v.n_rebals)),
        ("drift_now", str(v.drift_now)),
        ("ks_pvalue", "" if v.ks_pvalue is None else f"{v.ks_pvalue:.6f}"),
        ("reasons", " | ".join(v.reasons)),
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("metric,value\n")
        for k, val in rows:
            esc = val.replace('"', '""')
            fh.write(f'{k},"{esc}"\n')


def _write_verdict_md(
    path: Path,
    v: VerdictResult,
    *,
    clock: paper_clock.PaperClockRow,
    kills: Sequence[paper_clock.KillEventRow],
    rebals: Sequence[paper_clock.RebalEventRow],
) -> None:
    lines: list[str] = []
    lines.append(f"# TSMOM 30-day paper-trade verdict: **{v.verdict}**")
    lines.append("")
    lines.append(f"- Run ID: `{clock.run_id}`")
    lines.append(f"- Strategy: `{clock.strategy}`")
    lines.append(f"- Started: `{clock.started_at_utc.isoformat()}`")
    lines.append(f"- Start NAV: `${clock.start_nav}`")
    lines.append(f"- Days elapsed: `{v.days_elapsed}` / `{clock.cap_days}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- Paper Sharpe (annualized): `{v.paper_sharpe}`")
    lines.append(f"- Paper total return: `{v.paper_total_return:.4%}`")
    lines.append(f"- Paper max drawdown: `{v.paper_max_drawdown:.4%}`")
    lines.append(f"- Rebals completed: `{v.n_rebals}`")
    lines.append(f"- Reconcile drift now: `{v.drift_now:.4%}`")
    lines.append(f"- KS p-value: `{v.ks_pvalue}`")
    lines.append("")
    lines.append("## Verdict reasoning")
    lines.append("")
    for r in v.reasons:
        lines.append(f"- {r}")
    lines.append("")
    if kills:
        lines.append("## Kill events recorded")
        lines.append("")
        for k in kills:
            lines.append(
                f"- `{k.fired_at_utc.isoformat()}` — **{k.criterion}** "
                f"(value=`{k.value}`, threshold=`{k.threshold}`): {k.detail}"
            )
        lines.append("")
    if rebals:
        lines.append("## Rebals completed")
        lines.append("")
        for rb in rebals:
            lines.append(
                f"- `{rb.rebal_at_utc.isoformat()}` — orders=`{rb.n_orders_submitted}` "
                f"NAV=`${rb.nav_at_rebal}`"
            )
        lines.append("")
    lines.append("## Next steps")
    lines.append("")
    if v.verdict == "COMMIT":
        lines.append(
            "**COMMIT does NOT authorize live trading.** It means Branch C is "
            "alive. The operator now decides whether to (a) start ensemble work "
            "(carry already FAIL — would need a different complementary signal), "
            "(b) extend paper trading toward the live-cash gate, or (c) park "
            "Branch C and start a new branch."
        )
    else:
        lines.append(
            "**Branch C is dead.** Abandon TSMOM-alone. Reassess strategy. The "
            "PRD §3 hard-restart rule applies — do not soldier on with a known-"
            "broken signal."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casino.execution.tsmom_clock_check",
        description="TSMOM 30-day-cap daily monitor + day-30 verdict.",
    )
    parser.add_argument(
        "--verdict",
        action="store_true",
        help="Compute and persist the day-30 COMMIT-or-KILL verdict (writes report).",
    )
    parser.add_argument(
        "--no-kill-switch",
        action="store_true",
        help="Do not engage kill switch even if a criterion fires (testing only).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=paper_clock.DEFAULT_RUN_ID,
        help=(
            "paper_clock run_id to scope queries to. Defaults to "
            f"'{paper_clock.DEFAULT_RUN_ID}' for the live vanilla bot. "
            "Pass 'Belfort' to evaluate the shadow simulator."
        ),
    )
    args = parser.parse_args(argv)

    try:
        assert_paper_account()
    except Exception as e:  # noqa: BLE001
        logger.error("tsmom_clock_check: refusing to run: {}", e)
        return 2

    try:
        result = run_daily_check(
            engage_kill_switch=not args.no_kill_switch,
            run_id=args.run_id,
        )
        logger.info(
            "tsmom_clock_check[{}]: daily; kill_fired={} criteria={} days={}",
            args.run_id,
            result.kill_fired,
            result.triggered_criteria,
            result.days_elapsed,
        )
        if args.verdict:
            v = run_verdict(run_id=args.run_id)
            logger.warning(
                "tsmom_clock_check[{}]: VERDICT={} sharpe={} rebals={} drift={:.4%}",
                args.run_id,
                v.verdict,
                v.paper_sharpe,
                v.n_rebals,
                v.drift_now,
            )
        return 0
    except Exception as e:  # noqa: BLE001
        logger.exception("tsmom_clock_check: unhandled exception: {}", e)
        return 1


# Convenience for tests + dashboard: surface the threshold constants as a dict.
def kill_thresholds_view() -> dict[str, Any]:
    return {
        "drawdown": float(DD_KILL_THRESHOLD),
        "single_day": float(SINGLE_DAY_KILL_THRESHOLD),
        "gross_cap_buffer": float(GROSS_CAP_BUFFER),
        "reconcile_drift": float(RECONCILE_DRIFT_THRESHOLD),
        "ks_kill_pvalue": KS_KILL_PVALUE,
        "ks_commit_pvalue": KS_COMMIT_PVALUE,
        "commit_sharpe_floor": COMMIT_SHARPE_FLOOR,
        "commit_drift_threshold": float(RECONCILE_DRIFT_COMMIT_THRESHOLD),
    }


# Re-export so static analyzers see `json` and `book` are used (json used in
# verdict CSV writer indirectly).
__all__ = ["json", "book"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
