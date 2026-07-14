#!/usr/bin/env python3
"""Bake retrieved context into a test set while filtering allowed source URLs.

This is intended for ablation datasets such as nim_curated_html_only. The ES
index can remain the original corpus index; retrieved chunks are filtered after
retrieval/reranking to retain only URLs present in the filtered passages.jsonl.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import ssl
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests


EMBED_URL = os.getenv("EMBED_URL", "http://10.43.101.173:8000/v1/embeddings")
RANK_URL = os.getenv("RANK_URL", "http://10.43.126.32:8000/v1/ranking")
ES_URL = os.getenv("ES_URL", "https://10.43.233.46:9200")

EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
RANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"

log = logging.getLogger(__name__)


def get_es_password() -> str:
    if os.getenv("ES_PASSWORD"):
        return os.environ["ES_PASSWORD"]
    try:
        raw = subprocess.check_output(
            [
                "kubectl",
                "get",
                "secret",
                "rag-eck-elasticsearch-es-elastic-user",
                "-n",
                "runai-rag",
                "-o",
                "jsonpath={.data.elastic}",
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        sys.exit(f"failed to read ES password from Kubernetes secret: {exc}")
    return base64.b64decode(raw).decode().strip()


def canonical_url(value: str) -> str:
    value = (value or "").strip()
    return value.split("#", 1)[0]


def allowed_urls_from_passages(path: Path) -> set[str]:
    allowed: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        url = canonical_url(str(row.get("url", "")))
        if url:
            allowed.add(url)
    return allowed


def embed_query(query: str) -> list[float]:
    response = requests.post(
        EMBED_URL,
        json={"input": query, "model": EMBED_MODEL, "input_type": "query"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def knn_search(vector: list[float], *, index: str, top_k: int) -> list[dict[str, Any]]:
    response = requests.post(
        f"{ES_URL}/{index}/_search",
        auth=(os.getenv("ES_USER", "elastic"), get_es_password()),
        verify=False,
        json={
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "k": top_k,
                "num_candidates": top_k * 20,
            },
            "_source": [
                "text",
                "metadata.content_metadata.content_url",
                "metadata.doc_kind",
            ],
            "size": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["hits"]["hits"]


def rerank(query: str, hits: list[dict[str, Any]]) -> list[tuple[int, float, float]]:
    passages = [{"text": hit["_source"].get("text", "")} for hit in hits]
    response = requests.post(
        RANK_URL,
        json={"model": RANK_MODEL, "query": {"text": query}, "passages": passages},
        timeout=90,
    )
    response.raise_for_status()
    rankings = response.json()["rankings"]
    scored = []
    for i in range(len(hits)):
        logit = rankings[i]["logit"]
        score = 1.0 / (1.0 + math.exp(-logit))
        scored.append((i, logit, score))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored


def hit_url(hit: dict[str, Any]) -> str:
    metadata = hit.get("_source", {}).get("metadata", {}) or {}
    content_metadata = metadata.get("content_metadata", {}) or {}
    return canonical_url(str(content_metadata.get("content_url", "")))


def select_chunks(
    query: str,
    *,
    index: str,
    allowed_urls: set[str],
    vdb_top_k: int,
    rerank_top_k: int,
    score_threshold: float,
) -> tuple[list[dict[str, Any]], list[float], int]:
    vector = embed_query(query)
    hits = knn_search(vector, index=index, top_k=vdb_top_k)
    if not hits:
        return [], [], 0
    ranked = rerank(query, hits)
    all_scores = [score for _, _, score in ranked]
    chunks: list[dict[str, Any]] = []
    filtered_out = 0
    for hit_idx, _, score in ranked:
        hit = hits[hit_idx]
        url = hit_url(hit)
        if url not in allowed_urls:
            filtered_out += 1
            continue
        if score < score_threshold:
            continue
        chunks.append(
            {
                "content": hit.get("_source", {}).get("text", ""),
                "url": url,
                "score": score,
            }
        )
        if len(chunks) >= rerank_top_k:
            break
    return chunks, all_scores, filtered_out


def format_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(no relevant HTML context found)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}]"
        if chunk["url"]:
            header += f" {chunk['url']}"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def main() -> int:
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    ssl._create_default_https_context = ssl._create_unverified_context

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--index", default="nim_curated")
    ap.add_argument("--in-path", type=Path)
    ap.add_argument("--out-path", type=Path)
    ap.add_argument("--audit-path", type=Path)
    ap.add_argument("--vdb-top-k", type=int, default=75)
    ap.add_argument("--rerank-top-k", type=int, default=5)
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    in_path = args.in_path or args.dataset_dir / "test_set.jsonl"
    out_path = args.out_path or args.dataset_dir / "test_set_with_context.jsonl"
    audit_path = args.audit_path or args.dataset_dir / "test_set_with_context.audit.jsonl"
    allowed_urls = allowed_urls_from_passages(args.dataset_dir / "passages.jsonl")
    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    log.info("dataset=%s index=%s rows=%d allowed_urls=%d", args.dataset_dir, args.index, len(rows), len(allowed_urls))
    log.info("vdb_top_k=%d rerank_top_k=%d threshold=%.2f", args.vdb_top_k, args.rerank_top_k, args.score_threshold)

    chunk_counts: list[int] = []
    filtered_counts: list[int] = []
    max_scores: list[float] = []
    zero_chunk_rows = 0
    with out_path.open("w") as out_f, audit_path.open("w") as audit_f:
        for i, row in enumerate(rows):
            query = row.get("prompt", "")
            try:
                chunks, all_scores, filtered_out = select_chunks(
                    query,
                    index=args.index,
                    allowed_urls=allowed_urls,
                    vdb_top_k=args.vdb_top_k,
                    rerank_top_k=args.rerank_top_k,
                    score_threshold=args.score_threshold,
                )
                audit = {
                    "row_index": i,
                    "chunks_kept": len(chunks),
                    "filtered_out_non_allowed_url": filtered_out,
                    "kept_scores": [round(chunk["score"], 4) for chunk in chunks],
                    "max_score": round(max(all_scores), 4) if all_scores else None,
                    "min_score": round(min(all_scores), 4) if all_scores else None,
                    "candidates_scored": len(all_scores),
                }
            except requests.HTTPError as exc:
                chunks = []
                audit = {"row_index": i, "error": repr(exc)}
            chunk_counts.append(audit.get("chunks_kept", 0))
            filtered_counts.append(audit.get("filtered_out_non_allowed_url", 0))
            if audit.get("chunks_kept", 0) == 0:
                zero_chunk_rows += 1
            if audit.get("max_score") is not None:
                max_scores.append(audit["max_score"])

            new_row = dict(row)
            new_row["prompt"] = f"Context:\n{format_context(chunks)}\n\nQuestion: {query}"
            out_f.write(json.dumps(new_row) + "\n")
            audit_f.write(json.dumps(audit) + "\n")
            if (i + 1) % 50 == 0:
                log.info("processed %d/%d", i + 1, len(rows))

    total = len(rows)
    summary = {
        "rows": total,
        "avg_chunks_kept": round(sum(chunk_counts) / total, 4) if total else 0,
        "zero_chunk_rows": zero_chunk_rows,
        "zero_chunk_percent": round(100.0 * zero_chunk_rows / total, 2) if total else 0,
        "avg_filtered_non_allowed": round(sum(filtered_counts) / total, 4) if total else 0,
        "median_max_score": sorted(max_scores)[len(max_scores) // 2] if max_scores else None,
        "out_path": str(out_path),
        "audit_path": str(audit_path),
    }
    (args.dataset_dir / "test_set_with_context.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    log.info("summary: %s", json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
