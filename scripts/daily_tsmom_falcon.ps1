# Daily Falcon runner — aggressive vanilla-TSMOM sibling of DiCaprio (sim).
#
# Triggered by Windows Task Scheduler task: Casino_Daily_TSMOM_Falcon
# Schedule: weekdays 17:45 local (after the live DiCaprio chain at
# 17:00/17:15/17:30, so the ingest task has finished WRITING DuckDB before
# this read-only sim step runs).
#
# Falcon (run_id="Falcon") runs entirely in the in-process SimBroker. This
# wrapper invokes the runner with no args, which is a daily mark-to-market
# off rebal days and a full rebal on the last business day of the month
# (the runner's own contract). It needs today's OHLCV bar, which the
# Casino_Daily_TSMOM_Ingest task (17:00) already pulled into the shared
# DuckDB - there is no separate Falcon ingest.
#
# Deliberately NOT scheduled for Falcon:
#   * reconcile / book-sync  -> live-broker-only, meaningless for a sim.
#   * tsmom_clock_check       -> the daily kill path uses the LIVE Alpaca
#     broker regardless of --run-id and would flatten DiCaprio. Falcon is
#     monitored only via the read-only day-30 verdict (run manually):
#       uv run python -m casino.execution.tsmom_clock_check --verdict --run-id Falcon
#
# Logs to logs\daily_falcon_<yyyymmdd>.log (LOCAL date in filename, UTC in
# timestamps). Logs older than 30 days auto-prune.
#
# On non-zero exit, posts a critical Discord alert with the log tail.

# See note in daily_tsmom_ingest.ps1: "Stop" + native stderr breaks in PS 5.1.
$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

. (Join-Path $PSScriptRoot "_common.ps1")

$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("daily_falcon_" + (Get-Date -Format "yyyyMMdd") + ".log")
$envPath = Join-Path $projectRoot ".env"

Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_falcon START ==="

$uv = "C:\Users\TNTMi\.local\bin\uv.exe"
& $uv run python -m casino.execution.tsmom_falcon_runner *>> $logFile
$exit = $LASTEXITCODE

Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_falcon END exit_code=$exit ==="

if ($exit -ne 0) {
    $tail = ""
    try {
        $tail = (Get-Content $logFile -Tail 40 -ErrorAction SilentlyContinue) -join "`n"
    } catch {}
    Send-CasinoDiscordAlert `
        -Title "casino: daily_tsmom_falcon FAILED (exit=$exit)" `
        -Body $tail `
        -EnvPath $envPath `
        -LogFile $logFile | Out-Null
}

# Prune logs > 30 days.
Get-ChildItem -Path $logDir -Filter "daily_falcon_*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $exit
