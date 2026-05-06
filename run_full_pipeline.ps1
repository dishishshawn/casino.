# End-to-end Phase 1 Stage 1 text-PEAD kill-test pipeline (PowerShell).
# Run on a fresh always-on Windows box. Total ~5-8 hours, mostly CPU FinBERT.
#
# Foreground (window must stay open):
#   .\run_full_pipeline.ps1
#
# Detached (survives shell close — recommended for always-on PC):
#   Start-Process powershell -ArgumentList "-NoProfile","-File",".\run_full_pipeline.ps1" `
#       -WorkingDirectory $PWD -WindowStyle Hidden -RedirectStandardOutput pipeline.log
#

$ErrorActionPreference = "Stop"
$logFile = Join-Path $PSScriptRoot "pipeline.log"

function Log-Step {
    param([string]$msg)
    $line = "=== $(Get-Date -Format 'u') $msg ==="
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $logFile -Value $line
}

# 1. Ensure uv is on PATH (assumes already installed; install with the command in README).
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Install with:" -ForegroundColor Yellow
    Write-Host "  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
    exit 1
}

# 2. Sync deps (downloads torch ~400MB on first run).
Log-Step "uv sync"
uv sync

# 3. OHLCV ingest — 503 S&P 500 tickers, 2018-now. ~3 hrs at rate-limit 0.
Log-Step "ohlcv ingest"
uv run python -m casino.data.ingest_yfinance `
    --tickers-file universe_sp500.txt --mode ohlcv `
    --ohlcv-start 2018-01-01 --rate-limit-sec 0

# 4. Earnings ingest with 40-quarter backfill. ~15 min.
Log-Step "earnings ingest"
uv run python -m casino.data.ingest_yfinance `
    --tickers-file universe_sp500.txt --mode earnings --rate-limit-sec 0

# 5. Transcript ingest. Capped at 3000 to keep FinBERT runtime under ~3hrs.
Log-Step "transcript ingest"
uv run python -m casino.data.ingest_transcripts_hf --limit 3000

# 6. FinBERT scoring. ~3 hrs on CPU for 3000 transcripts.
Log-Step "FinBERT scoring"
uv run python -m casino.signals.finbert_score

# 7. Kill-test gate. <1 min.
Log-Step "kill-test gate"
uv run python -m casino.backtest.finbert_baseline --no-save *>&1 | Tee-Object -FilePath finbert_killtest_result.txt

Log-Step "DONE"
Write-Host "Verdict written to finbert_killtest_result.txt" -ForegroundColor Green
