#!/bin/bash
# End-to-end Phase 1 Stage 1 text-PEAD kill-test pipeline.
# Run on a fresh always-on Linux/Mac box. Total ~5-8 hours, mostly CPU FinBERT.
# Logs everything to pipeline.log and prints a final verdict.
#
# Usage:
#   ./run_full_pipeline.sh
#   nohup ./run_full_pipeline.sh > pipeline.log 2>&1 &      # detached
#
set -euo pipefail
exec > >(tee -a pipeline.log) 2>&1

echo "=== $(date -u) start ==="

# 1. Ensure uv is on PATH (idempotent — installs only if missing).
if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

# 2. Sync deps (downloads torch ~400MB on first run).
echo "=== $(date -u) uv sync ==="
uv sync

# 3. OHLCV ingest — 503 S&P 500 tickers, 2018-now. ~3 hrs at rate-limit 0.
echo "=== $(date -u) ohlcv ingest ==="
uv run python -m casino.data.ingest_yfinance \
    --tickers-file universe_sp500.txt --mode ohlcv \
    --ohlcv-start 2018-01-01 --rate-limit-sec 0

# 4. Earnings ingest with 40-quarter backfill. ~15 min.
echo "=== $(date -u) earnings ingest ==="
uv run python -m casino.data.ingest_yfinance \
    --tickers-file universe_sp500.txt --mode earnings --rate-limit-sec 0

# 5. Transcript ingest. Capped at 3000 to keep FinBERT runtime under ~3hrs.
echo "=== $(date -u) transcript ingest ==="
uv run python -m casino.data.ingest_transcripts_hf --limit 3000

# 6. FinBERT scoring. ~3 hrs on CPU for 3000 transcripts.
echo "=== $(date -u) FinBERT scoring ==="
uv run python -m casino.signals.finbert_score

# 7. Kill-test gate. <1 min.
echo "=== $(date -u) kill-test gate ==="
uv run python -m casino.backtest.finbert_baseline --no-save | tee finbert_killtest_result.txt

echo "=== $(date -u) DONE ==="
echo "Verdict written to finbert_killtest_result.txt"
