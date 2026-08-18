# Branch C 30-day clock starter -- one-shot for 2026-05-29.
#
# Wraps the manual `casino.execution.tsmom_runner` invocation that PRD §6.3
# (amendment) reserves for the operator. Scheduling this is a deliberate
# operator decision made on 2026-05-07 to remove "did I forget" risk.
#
# Sequence:
#   1. Run casino.execution.tsmom_runner (no flags). The runner is a no-op
#      on non-rebal days, so this is safe to fire from a misconfigured
#      schedule -- it will simply log "not the last business day".
#   2. Verify paper_clock has been populated. If it has → info Discord
#      "Branch C clock STARTED". If it hasn't → critical Discord with the
#      log tail; the operator must investigate same evening before the
#      next monthly rebal opportunity (~30 days later).
#
# This wrapper does NOT add a heartbeat ping -- the regular 17:30 kill-check
# task will fire the heartbeat 20 min after this wrapper, and the heartbeat
# will reflect the new clock state automatically.

$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

. (Join-Path $PSScriptRoot "_common.ps1")

$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("branch_c_clock_start_" + (Get-Date -Format "yyyyMMdd") + ".log")
$envPath = Join-Path $projectRoot ".env"
$uv = "C:\Users\TNTMi\.local\bin\uv.exe"

Write-CasinoLog -LogFile $logFile -Message "=== branch_c_clock_start START ==="

# --- Step 1: invoke the runner ---
& $uv run python -m casino.execution.tsmom_runner *>> $logFile
$runnerExit = $LASTEXITCODE

if ($runnerExit -ne 0) {
    $tail = ""
    try { $tail = (Get-Content $logFile -Tail 60 -ErrorAction SilentlyContinue) -join "`n" } catch {}
    Send-CasinoDiscordAlert `
        -Title "casino: BRANCH C CLOCK START FAILED (runner exit=$runnerExit)" `
        -Body $tail `
        -EnvPath $envPath `
        -LogFile $logFile | Out-Null
    Write-CasinoLog -LogFile $logFile -Message "=== branch_c_clock_start END exit_code=$runnerExit (verify skipped) ==="
    exit $runnerExit
}

# --- Step 2: verify paper_clock has a row (clock actually started) ---
# Exit 0 if a row exists, 1 otherwise. A row missing here means the runner
# returned 0 but hit a no-op path (non-rebal day, stale OHLCV, etc.).
& $uv run python -c "from casino.execution.paper_clock import fetch_paper_clock; import sys; sys.exit(0 if fetch_paper_clock() else 1)" *>> $logFile
$verifyExit = $LASTEXITCODE
Write-CasinoLog -LogFile $logFile -Message "verify_clock_started exit=$verifyExit"

if ($verifyExit -eq 0) {
    Send-CasinoDiscordAlert `
        -Title "casino: Branch C clock STARTED" `
        -Body "tsmom_runner completed and paper_clock has a row. Day 1/30 has begun. The 17:30 kill-check task will pick up the new state automatically." `
        -EnvPath $envPath `
        -Color 3066993 `
        -LogFile $logFile | Out-Null
    Write-CasinoLog -LogFile $logFile -Message "=== branch_c_clock_start END exit_code=0 (CLOCK STARTED) ==="
    exit 0
} else {
    $tail = ""
    try { $tail = (Get-Content $logFile -Tail 60 -ErrorAction SilentlyContinue) -join "`n" } catch {}
    Send-CasinoDiscordAlert `
        -Title "casino: BRANCH C CLOCK DID NOT START" `
        -Body "tsmom_runner exited 0 but paper_clock has no row. The runner likely returned a no-op (non-rebal day, stale OHLCV, freshness gate, etc.). Investigate before the next month-end opportunity. Log tail:`n`n$tail" `
        -EnvPath $envPath `
        -LogFile $logFile | Out-Null
    Write-CasinoLog -LogFile $logFile -Message "=== branch_c_clock_start END exit_code=2 (NO-OP -- clock not started) ==="
    exit 2
}
