# Daily Branch C maintenance — DiCaprio (live) + Belfort (shadow).
#
# Wire this into Windows Task Scheduler to run once per day, ~6 PM ET
# (after NYSE close + ~30 min for yfinance to publish today's bar).
#
# Foreground:
#   .\daily_branch_c.ps1
#
# Background (recommended for always-on box):
#   Start-Process powershell -ArgumentList "-NoProfile","-File",".\daily_branch_c.ps1" `
#       -WorkingDirectory $PWD -WindowStyle Hidden -RedirectStandardOutput daily_branch_c.log
#
# What it does, in order:
#   1. yfinance ingest — pulls today's OHLCV bars for the 10-ETF universe.
#   2. Belfort shadow runner — daily mark-to-market on the sim broker;
#      computes regime-filtered weights, fills any pending sim orders,
#      records sim_nav_history for the day. Force-flag handles non-rebal
#      days as a daily MTM (not a rebal) per the runner's contract.
#   3. DiCaprio clock check — evaluates kill criteria (drawdown,
#      single-day, cap-violation, reconcile drift, KS-test) for the live
#      Alpaca paper bot. Engages kill switch if anything fires.
#   4. Belfort clock check — same kill criteria for the sim shadow.
#
# Day 30 (~2026-06-07) — append `--verdict` to the two clock checks to
# emit the COMMIT/KILL verdict for each bot.
#
# Idempotent. Safe to re-run on the same day. Each step's failure is
# logged but does not halt the others (one bad ingest shouldn't block
# the kill checks; one halted clock shouldn't block the other).

$ErrorActionPreference = "Continue"
$logFile = Join-Path $PSScriptRoot "daily_branch_c.log"

function Log-Step {
    param([string]$msg)
    $line = "=== $(Get-Date -Format 'u') $msg ==="
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $logFile -Value $line
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found on PATH" -ForegroundColor Yellow
    exit 1
}

Log-Step "yfinance ingest (today's bar for the 10-ETF universe)"
$ingestStart = (Get-Date).AddDays(-30).ToString("yyyy-MM-dd")
uv run python -m casino.data.ingest_yfinance `
    --tickers-file universe_tsmom.txt `
    --mode ohlcv `
    --ohlcv-start $ingestStart `
    --rate-limit-sec 0 *>> $logFile

Log-Step "Belfort shadow daily mark-to-market"
uv run python -m casino.execution.tsmom_shadow_runner *>> $logFile

Log-Step "DiCaprio clock check (kill criteria)"
uv run python -m casino.execution.tsmom_clock_check --run-id DiCaprio *>> $logFile

Log-Step "Belfort clock check (kill criteria)"
uv run python -m casino.execution.tsmom_clock_check --run-id Belfort *>> $logFile

Log-Step "daily_branch_c.ps1 done"
