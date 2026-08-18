# Daily TSMOM kill-criteria check (Branch C 30-day cap monitor) + heartbeat.
#
# Triggered by Windows Task Scheduler task: Casino_Daily_TSMOM_KillCheck
# Schedule: weekdays 17:30 local (15 min after reconcile_eod).
#
# This wrapper does TWO things in order:
#   1. Run casino.execution.tsmom_clock_check — daily kill-criteria evaluation
#      (no-op pre-clock; active May 30+).
#   2. If step 1 succeeded, run casino.jobs.heartbeat — single Discord embed
#      summarizing today's state (equity, drift, OHLCV freshness, kill events,
#      Branch C day/30 post-clock).
#
# Heartbeat severity is info on a clean day, warning on drift/drawdown/stale
# data, critical if any kill_event fired today. The heartbeat dispatch itself
# is best-effort; a failed Discord post does not flip this wrapper's exit
# code (the kill_event row + the kill_check's own critical alert remain
# authoritative).
#
# On non-zero exit from step 1, posts a critical Discord alert with the
# log tail and skips step 2.

$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

. (Join-Path $PSScriptRoot "_common.ps1")

$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("daily_killcheck_" + (Get-Date -Format "yyyyMMdd") + ".log")
$envPath = Join-Path $projectRoot ".env"

Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_kill_check START ==="

$uv = "C:\Users\TNTMi\.local\bin\uv.exe"

# --- Step 1: daily kill-criteria check ---
& $uv run python -m casino.execution.tsmom_clock_check *>> $logFile
$exit = $LASTEXITCODE

if ($exit -ne 0) {
    $tail = ""
    try {
        $tail = (Get-Content $logFile -Tail 40 -ErrorAction SilentlyContinue) -join "`n"
    } catch {}
    Send-CasinoDiscordAlert `
        -Title "casino: daily_tsmom_kill_check FAILED (exit=$exit)" `
        -Body $tail `
        -EnvPath $envPath `
        -LogFile $logFile | Out-Null
    Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_kill_check END exit_code=$exit (heartbeat skipped) ==="
    Get-ChildItem -Path $logDir -Filter "daily_killcheck_*.log" |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
        Remove-Item -Force -ErrorAction SilentlyContinue
    exit $exit
}

# --- Step 2: heartbeat (best-effort; never flips exit code) ---
Write-CasinoLog -LogFile $logFile -Message "--- heartbeat ---"
& $uv run python -m casino.jobs.heartbeat *>> $logFile
$hbExit = $LASTEXITCODE
if ($hbExit -ne 0) {
    Write-CasinoLog -LogFile $logFile -Message "heartbeat exited $hbExit (non-fatal; kill-check exit code preserved)"
}

Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_kill_check END exit_code=$exit heartbeat_exit=$hbExit ==="

# Prune logs > 30 days.
Get-ChildItem -Path $logDir -Filter "daily_killcheck_*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $exit
