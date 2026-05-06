#!/bin/bash
# Phase 1 Stage 1 text-PEAD kill-test pipeline (Branch D).
# Run in Codespace or any Linux host with ~16GB RAM and ~5GB free disk.
# Total runtime: ~1-3 hours depending on transcript count and CPU.
set -euo pipefail

echo "=== 1/3 ingest free HF transcripts ==="
uv run python -m casino.data.ingest_transcripts_hf

echo "=== 2/3 score every transcript with ProsusAI/finbert (CPU) ==="
uv run python -m casino.signals.finbert_score

echo "=== 3/3 run kill-test gate ==="
uv run python -m casino.backtest.finbert_baseline --no-save | tee finbert_killtest_result.txt

echo ""
echo "Done. Verdict in finbert_killtest_result.txt"
