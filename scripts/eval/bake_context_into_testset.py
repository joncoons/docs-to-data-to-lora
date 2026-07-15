"""Pre-bake retrieved context into the Stage 3 test sets.

Mirrors the prior nim-sft-final/eval/collect_responses.py + retrieval.py pattern
of doing retrieval ONCE per question and sharing the same context across all
models being compared. Here we do it OFFLINE so the context becomes part of
the dataset itself — every Evaluator job against this dataset sees identical
input, making pairwise judging fair without any per-request retrieval at
eval time.

Per row, we:
  1. Take the original prompt as the query
  2. Embed via nemoretriever-embedding-ms (Llama-3.2-NV-EmbedQA-1B-v2, 2048-dim)
  3. kNN search in the corpus's ES index, retrieving vdb_top_k=25 candidates
  4. Rerank via nemoretriever-ranking-ms (Llama-3.2-NV-RerankQA-1B-v2)
  5. Apply sigmoid to the raw logits → relevance scores in [0,1]
  6. Drop chunks below score_threshold=0.5
  7. Take the top rerank_top_k=5 of what remains
  8. Format Context:..\n\nQuestion: <original prompt>
  9. Rewrite the row's prompt field

Output is a new JSONL alongside the original with `_with_context` suffix, and
a per-row JSON audit log with chunk counts + score ranges.

Usage:
  python3 scripts/eval/bake_context_into_testset.py \\
      --corpus nim_curated \\
      [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import httpx


# Service endpoints. Override these for your deployment.
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8002/v1/embeddings")
RANK_URL  = os.getenv("RANK_URL",  "http://localhost:8003/v1/ranking")
ES_URL    = os.getenv("ES_URL",    "http://localhost:9200")

EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
RANK_MODEL  = "nvidia/llama-3.2-nv-rerankqa-1b-v2"

_CORPUS_TO_INDEX = {
    "nim_curated":        "nim_curated",
    "nemo_usvcs_curated": "nemo_usvcs_curated",
}

log = logging.getLogger(__name__)


def _es_creds() -> tuple[str, str]:
    user = os.getenv("ES_USER", "elastic")
    pw = os.getenv("ES_PASSWORD")
    if not pw:
        sys.exit(
            "ES_PASSWORD must be set. Run:\n"
            "  export ES_PASSWORD=<your-elasticsearch-password>"
        )
    return user, pw


def embed_query(query: str) -> list[float]:
    r = httpx.post(
        EMBED_URL,
        json={"input": query, "model": EMBED_MODEL, "input_type": "query"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def knn_search(vector: list[float], index: str, top_k: int) -> list[dict]:
    user, pw = _es_creds()
    r = httpx.post(
        f"{ES_URL}/{index}/_search",
        auth=(user, pw),
        verify=False,
        json={
            "knn": {
                "field":         "vector",
                "query_vector":  vector,
                "k":             top_k,
                # standard ANN heuristic: explore ~20× the desired top-k
                "num_candidates": top_k * 20,
            },
            "_source": ["text", "metadata.content_metadata.content_url"],
            "size":    top_k,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["hits"]["hits"]


def rerank(query: str, hits: list[dict]) -> list[tuple[int, float, float]]:
    """Return list of (hit_index, raw_logit, sigmoid_score) sorted by score desc."""
    passages = [{"text": h["_source"]["text"]} for h in hits]
    r = httpx.post(
        RANK_URL,
        json={"model": RANK_MODEL, "query": {"text": query}, "passages": passages},
        timeout=60,
    )
    r.raise_for_status()
    rankings = r.json()["rankings"]
    scored = []
    for i in range(len(hits)):
        logit = rankings[i]["logit"]
        score = 1.0 / (1.0 + math.exp(-logit))
        scored.append((i, logit, score))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored


def select_chunks(
    query: str,
    index: str,
    vdb_top_k: int,
    rerank_top_k: int,
    score_threshold: float,
) -> tuple[list[dict], list[float]]:
    """Returns (selected_chunks, all_reranker_scores). all_scores is for audit."""
    vec = embed_query(query)
    hits = knn_search(vec, index, vdb_top_k)
    if not hits:
        return [], []
    ranked = rerank(query, hits)
    all_scores = [s for _, _, s in ranked]
    kept_idx = [(i, s) for i, _, s in ranked if s >= score_threshold][:rerank_top_k]
    chunks: list[dict] = []
    for i, score in kept_idx:
        src = hits[i]["_source"]
        chunks.append({
            "content": src.get("text", ""),
            "url":     (src.get("metadata", {}) or {})
                          .get("content_metadata", {})
                          .get("content_url", ""),
            "score":   score,
        })
    return chunks, all_scores


def format_context(chunks: list[dict]) -> str:
    """Format selected chunks as a numbered context block."""
    if not chunks:
        return "(no relevant context found)"
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}]"
        if c["url"]:
            header += f" {c['url']}"
        parts.append(f"{header}\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def bake_row(
    row: dict,
    index: str,
    vdb_top_k: int,
    rerank_top_k: int,
    score_threshold: float,
) -> tuple[dict, dict]:
    """Return (new_row, audit_entry)."""
    query = row.get("prompt", "")
    if not query:
        return row, {"error": "no prompt field"}
    chunks, all_scores = select_chunks(
        query, index, vdb_top_k, rerank_top_k, score_threshold
    )
    context = format_context(chunks)
    new_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    new_row = dict(row)
    new_row["prompt"] = new_prompt
    audit = {
        "chunks_kept":   len(chunks),
        "kept_scores":   [round(c["score"], 4) for c in chunks],
        "max_score":     round(max(all_scores), 4) if all_scores else None,
        "min_score":     round(min(all_scores), 4) if all_scores else None,
        "above_threshold_count": sum(1 for s in all_scores if s >= score_threshold),
        "candidates_scored":     len(all_scores),
    }
    return new_row, audit


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-bake retrieved context into a Stage 3 test set"
    )
    ap.add_argument("--corpus", required=True, choices=list(_CORPUS_TO_INDEX))
    ap.add_argument("--in-path", type=Path,
                    help="Default: <DATASET_ROOT>/<corpus>/test_set.jsonl")
    ap.add_argument("--out-path", type=Path,
                    help="Default: <in>.with_context.jsonl alongside the input")
    ap.add_argument("--audit-path", type=Path,
                    help="Default: <out>.audit.jsonl")
    ap.add_argument("--vdb-top-k", type=int, default=25)
    ap.add_argument("--rerank-top-k", type=int, default=5)
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N rows (0=all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write the output JSONL; only print audit summary")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index = _CORPUS_TO_INDEX[args.corpus]
    in_path = args.in_path or Path(f"<DATASET_ROOT>/{args.corpus}/test_set.jsonl")
    out_path = args.out_path or in_path.with_name("test_set_with_context.jsonl")
    audit_path = args.audit_path or out_path.with_suffix(".audit.jsonl")

    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")
    log.info("corpus=%s index=%s vdb=%d rerank=%d threshold=%.2f",
             args.corpus, index, args.vdb_top_k, args.rerank_top_k, args.score_threshold)
    log.info("input : %s", in_path)
    log.info("output: %s%s", out_path, " (DRY RUN)" if args.dry_run else "")

    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    log.info("processing %d rows", len(rows))

    out_f = None
    audit_f = None
    if not args.dry_run:
        out_f = out_path.open("w")
        audit_f = audit_path.open("w")

    # summary accumulators
    chunk_counts: list[int] = []
    zero_chunk_rows = 0
    max_seen_scores: list[float] = []

    for i, row in enumerate(rows):
        try:
            new_row, audit = bake_row(
                row, index,
                args.vdb_top_k, args.rerank_top_k, args.score_threshold,
            )
        except httpx.HTTPError as e:
            log.error("row %d: HTTP error: %r", i, e)
            audit = {"error": repr(e)}
            new_row = row
        chunk_counts.append(audit.get("chunks_kept", 0))
        if audit.get("chunks_kept", 0) == 0:
            zero_chunk_rows += 1
        if audit.get("max_score") is not None:
            max_seen_scores.append(audit["max_score"])
        if out_f is not None:
            out_f.write(json.dumps(new_row) + "\n")
        if audit_f is not None:
            audit_f.write(json.dumps({"row_index": i, **audit}) + "\n")
        if (i + 1) % 50 == 0:
            log.info("  ... %d/%d rows", i + 1, len(rows))

    if out_f is not None:
        out_f.close()
    if audit_f is not None:
        audit_f.close()

    # summary
    total = len(rows)
    avg_chunks = (sum(chunk_counts) / total) if total else 0
    log.info("=== summary ===")
    log.info("  rows processed     : %d", total)
    log.info("  avg chunks kept    : %.2f", avg_chunks)
    for k in (0, 1, 2, 3, 4, 5):
        cnt = sum(1 for c in chunk_counts if c == k)
        log.info("  rows with %d chunks: %4d  (%.1f%%)",
                 k, cnt, 100.0 * cnt / total if total else 0)
    if max_seen_scores:
        log.info("  max-reranker-score per row: median=%.3f  min=%.3f  max=%.3f",
                 sorted(max_seen_scores)[len(max_seen_scores) // 2],
                 min(max_seen_scores), max(max_seen_scores))
    log.info("  rows with 0 chunks : %d  (%.1f%%)  <-- below threshold across all 25 candidates",
             zero_chunk_rows, 100.0 * zero_chunk_rows / total if total else 0)
    if not args.dry_run:
        log.info("  wrote %s and audit %s", out_path, audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
