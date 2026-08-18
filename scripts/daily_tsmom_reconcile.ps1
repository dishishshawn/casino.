# Daily reconcile_eod — broker connection, book sync, daily P&L, drawdown.
#
# Triggered by Windows Task Scheduler task: Casino_Daily_TSMOM_Reconcile
# Schedule: weekdays 17:15 local (15 min after the ingest task).
#
# Logs to logs\daily_reconcile_<yyyymmdd>.log (LOCAL date in filename, UTC in
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
$logFile = Join-Path $logDir ("daily_reconcile_" + (Get-Date -Format "yyyyMMdd") + ".log")
$envPath = Join-Path $projectRoot ".env"

Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_reconcile START ==="

$uv = "C:\Users\TNTMi\.local\bin\uv.exe"
& $uv run python -m casino.jobs.reconcile_eod *>> $logFile
$exit = $LASTEXITCODE

Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_reconcile END exit_code=$exit ==="

if ($exit -ne 0) {
    $tail = ""
    try {
        $tail = (Get-Content $logFile -Tail 40 -ErrorAction SilentlyContinue) -join "`n"
    } catch {}
    Send-CasinoDiscordAlert `
        -Title "casino: daily_tsmom_reconcile FAILED (exit=$exit)" `
        -Body $tail `
        -EnvPath $envPath `
        -LogFile $logFile | Out-Null
}

# Prune logs > 30 days.
Get-ChildItem -Path $logDir -Filter "daily_reconcile_*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $exit
