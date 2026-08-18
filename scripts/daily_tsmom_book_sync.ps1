# Daily post-fill book sync — populate book.positions from the broker.
#
# Triggered by Windows Task Scheduler task: Casino_Daily_TSMOM_BookSync
# Schedule: weekdays 09:45 local (15 min after NYSE open at 08:30 CT).
#
# Closes the gap exposed by the 2026-05-15 incident: bracket-order fills
# landed at 09:30 ET but book.positions stayed empty until the 17:30
# reconcile, which then saw full-notional drift and auto-fired the kill.
#
# Logs to logs\daily_book_sync_<yyyymmdd>.log. Discord alert on failure.

$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

. (Join-Path $PSScriptRoot "_common.ps1")

$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("daily_book_sync_" + (Get-Date -Format "yyyyMMdd") + ".log")
$envPath = Join-Path $projectRoot ".env"

Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_book_sync START ==="

$uv = "C:\Users\TNTMi\.local\bin\uv.exe"
& $uv run python -m casino.jobs.sync_book *>> $logFile
$exit = $LASTEXITCODE

Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_book_sync END exit_code=$exit ==="

if ($exit -ne 0) {
    $tail = ""
    try {
        $tail = (Get-Content $logFile -Tail 40 -ErrorAction SilentlyContinue) -join "`n"
    } catch {}
    Send-CasinoDiscordAlert `
        -Title "casino: daily_tsmom_book_sync FAILED (exit=$exit)" `
        -Body $tail `
        -EnvPath $envPath `
        -LogFile $logFile | Out-Null
}

# Prune logs > 30 days.
Get-ChildItem -Path $logDir -Filter "daily_book_sync_*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $exit
