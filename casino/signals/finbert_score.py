"""FinBERT sentiment scoring for earnings transcripts.

Uses ProsusAI/finbert (HF), a 2018-vintage BERT model fine-tuned on Financial
PhraseBank. We pin to this model specifically because its parametric knowledge
cutoff is well before our 2018-2026 backtest window — eliminates the look-ahead
bias that frontier LLMs (Claude/GPT-4) carry on transcript backtests
(Sarkar-Vafa 2024, Glasserman-Lin 2024 *JFDS*, Benhenda 2026 *Look-Ahead-Bench*).

Per-transcript score: mean of (P_pos - P_neg) probabilities across non-overlapping
512-token windows. Stored in DuckDB `finbert_scores` (ticker, event_date) → score.

CLI:
    uv run python -m casino.signals.finbert_score                 # score all unscored
    uv run python -m casino.signals.finbert_score --limit 50      # smoke test
    uv run python -m casino.signals.finbert_score --rebuild       # re-score all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from casino.data import store

_MODEL_NAME = "ProsusAI/finbert"
_MAX_TOKENS = 512
_LABELS = ("positive", "negative", "neutral")  # ProsusAI/finbert label order


def _load_model() -> tuple[Any, Any, Any]:
    """Return (tokenizer, model, torch). Imports lazily so test-only paths skip the cost."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    logger.info("loading {} (first run downloads ~440 MB)", _MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    mdl.eval()
    return tok, mdl, torch


def score_text(text: str, *, tokenizer: Any, model: Any, torch: Any) -> dict[str, float]:
    """Score one transcript by averaging logit probabilities across 512-token chunks.

    Returns {pos, neu, neg, net, n_chunks} where net = pos - neg in [-1, 1].
    """
    enc = tokenizer(text, return_tensors=None, truncation=False)
    input_ids = enc["input_ids"]

    chunks: list[list[int]] = []
    for i in range(0, len(input_ids), _MAX_TOKENS):
        chunks.append(input_ids[i : i + _MAX_TOKENS])
    if not chunks:
        return {"pos": 0.0, "neu": 0.0, "neg": 0.0, "net": 0.0, "n_chunks": 0}

    pos_sum = neu_sum = neg_sum = 0.0
    with torch.no_grad():
        for chunk in chunks:
            ids = torch.tensor([chunk])
            attn = torch.ones_like(ids)
            out = model(input_ids=ids, attention_mask=attn)
            probs = torch.softmax(out.logits, dim=-1)[0].tolist()
            # ProsusAI/finbert label order: positive, negative, neutral.
            pos_sum += probs[0]
            neg_sum += probs[1]
            neu_sum += probs[2]
    n = len(chunks)
    pos = pos_sum / n
    neg = neg_sum / n
    neu = neu_sum / n
    return {"pos": pos, "neu": neu, "neg": neg, "net": pos - neg, "n_chunks": n}


def _select_unscored(
    *, db_path: Path | None, limit: int | None, rebuild: bool
) -> list[dict[str, Any]]:
    """Return transcript rows that need scoring (full table if rebuild=True)."""
    if rebuild:
        sql = """
            SELECT ticker, event_date, transcript_text
            FROM transcripts
            WHERE transcript_text IS NOT NULL AND length(transcript_text) > 0
        """
    else:
        sql = """
            SELECT t.ticker, t.event_date, t.transcript_text
            FROM transcripts t
            LEFT JOIN finbert_scores f
              ON t.ticker = f.ticker AND t.event_date = f.event_date
            WHERE f.ticker IS NULL
              AND t.transcript_text IS NOT NULL
              AND length(t.transcript_text) > 0
        """
    with store.get_duckdb_conn(db_path, read_only=True) as conn:
        rows = conn.execute(sql).fetchall()
    if limit is not None:
        rows = rows[:limit]
    return [{"ticker": r[0], "event_date": r[1], "text": r[2]} for r in rows]


def score_all(
    *,
    db_path: Path | None = None,
    limit: int | None = None,
    rebuild: bool = False,
    flush_every: int = 50,
) -> int:
    """Score every unscored transcript, upserting to `finbert_scores` in batches."""
    store.create_schema(db_path=db_path)
    todo = _select_unscored(db_path=db_path, limit=limit, rebuild=rebuild)
    if not todo:
        logger.info("nothing to score (all transcripts already have FinBERT scores)")
        return 0

    logger.info("scoring {} transcripts with {}", len(todo), _MODEL_NAME)
    tokenizer, model, torch = _load_model()

    pending: list[dict[str, object]] = []
    written = 0
    t0 = time.monotonic()
    for i, row in enumerate(todo, 1):
        text = str(row["text"])
        try:
            s = score_text(text, tokenizer=tokenizer, model=model, torch=torch)
        except Exception as exc:  # noqa: BLE001 — keep loop alive on one-row failure
            logger.warning("FinBERT failed on {} {}: {}", row["ticker"], row["event_date"], exc)
            continue
        pending.append(
            {
                "ticker": row["ticker"],
                "event_date": row["event_date"],
                "score_pos": s["pos"],
                "score_neu": s["neu"],
                "score_neg": s["neg"],
                "score_net": s["net"],
                "n_chunks": int(s["n_chunks"]),
                "model_name": _MODEL_NAME,
            }
        )
        if len(pending) >= flush_every:
            written += store.upsert_finbert_scores(pending, db_path=db_path)
            pending = []
            elapsed = time.monotonic() - t0
            logger.info(
                "scored {}/{} ({:.1f} chunks/s avg)",
                i,
                len(todo),
                (i * 5) / max(elapsed, 1e-6),
            )
    if pending:
        written += store.upsert_finbert_scores(pending, db_path=db_path)

    logger.info("FinBERT scoring done: {} rows written in {:.0f}s", written, time.monotonic() - t0)
    return written


# ============================================================================ CLI
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="casino.signals.finbert_score",
        description="Score every transcript in the DB with ProsusAI/finbert.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap rows (smoke test)")
    p.add_argument("--rebuild", action="store_true", help="Re-score every transcript")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    n = score_all(limit=args.limit, rebuild=args.rebuild)
    return 0 if n > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
