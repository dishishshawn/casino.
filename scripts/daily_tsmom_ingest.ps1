# Daily TSMOM OHLCV ingest — refreshes the 10-ETF universe in DuckDB.
#
# Triggered by Windows Task Scheduler task: Casino_Daily_TSMOM_Ingest
# Schedule: weekdays 17:00 local (after NYSE close + yfinance publish lag).
# CT close = 3 PM, ET close = 4 PM; running at 17:00 CT = 18:00 ET gives ~2hr buffer.
#
# Logs to logs\daily_ingest_<yyyymmdd>.log (file name uses LOCAL date for human
# readability; embedded timestamps are UTC). Logs older than 30 days auto-prune.
#
# On non-zero exit, posts a critical Discord alert with the log tail.

# Note: do NOT use $ErrorActionPreference = "Stop" here. uv/Python emit normal
# logs to stderr (loguru), and in PS 5.1 that surfaces as NativeCommandError.
# We rely on $LASTEXITCODE for failure detection instead.
$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

. (Join-Path $PSScriptRoot "_common.ps1")

$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("daily_ingest_" + (Get-Date -Format "yyyyMMdd") + ".log")
$envPath = Join-Path $projectRoot ".env"
Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_ingest START ==="

# Ingest the last ~30 days; yfinance dedupes against existing rows in DuckDB.
$startDate = (Get-Date).AddDays(-30).ToString("yyyy-MM-dd")
Write-CasinoLog -LogFile $logFile -Message "ohlcv-start=$startDate universe=universe_tsmom.txt"

$uv = "C:\Users\TNTMi\.local\bin\uv.exe"
& $uv run python -m casino.data.ingest_yfinance `
    --tickers-file universe_tsmom.txt --mode ohlcv `
    --ohlcv-start $startDate --rate-limit-sec 0 *>> $logFile
$exit = $LASTEXITCODE

Write-CasinoLog -LogFile $logFile -Message "=== daily_tsmom_ingest END exit_code=$exit ==="

if ($exit -ne 0) {
    $tail = ""
    try {
        $tail = (Get-Content $logFile -Tail 40 -ErrorAction SilentlyContinue) -join "`n"
    } catch {}
    Send-CasinoDiscordAlert `
        -Title "casino: daily_tsmom_ingest FAILED (exit=$exit)" `
        -Body $tail `
        -EnvPath $envPath `
        -LogFile $logFile | Out-Null
}

# Prune logs > 30 days.
Get-ChildItem -Path $logDir -Filter "daily_ingest_*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $exit
