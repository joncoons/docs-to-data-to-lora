"""Stage 1B: kNN neighborhood synthesis (BRIDGING + CONTRASTIVE pairs)."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import tiktoken
from tqdm import tqdm

from scripts.pipeline.es_client import knn_search
from scripts.pipeline.llm_client import LLMClient
from scripts.pipeline.models import KVPRow, Passage
from scripts.pipeline.provenance import (
    passage_modalities,
    passage_source_chunk_ids,
    passage_source_kinds,
    passage_source_revision_id,
    passage_source_systems,
)
from scripts.pipeline.prompts import SYNTHESIS_SYSTEM, SYNTHESIS_USER

Elasticsearch = Any

log = logging.getLogger(__name__)
_enc = tiktoken.get_encoding("cl100k_base")


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()


def build_neighborhood_context(
    seed_text: str, neighbors: list[dict], max_tokens: int
) -> str:
    """Stitch seed + neighbor texts together; truncate to max_tokens."""
    parts = ["Passage A (seed):", seed_text]
    for i, n in enumerate(neighbors, start=1):
        parts.append(f"Passage {chr(65 + i)}:")
        parts.append(n.get("text", "").strip())
    joined = "\n\n".join(parts)
    tokens = _enc.encode(joined)
    if len(tokens) <= max_tokens:
        return joined
    return _enc.decode(tokens[:max_tokens])


def parse_synthesis_response(raw: str) -> list[dict]:
    s = _strip_fences(raw)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    return obj.get("pairs", [])


def process_passage_1b(
    passage: Passage,
    seed_vector: list[float],
    es: Elasticsearch,
    index: str,
    domain: str,
    llm: LLMClient,
    k: int = 6,
    num_candidates: int = 50,
    top_neighbors: int = 3,
    max_context_tokens: int = 1200,
) -> list[KVPRow]:
    if not seed_vector:
        return []
    hits = knn_search(
        es, index=index, vector=seed_vector,
        exclude_url=passage.url, k=k, num_candidates=num_candidates,
    )
    neighbors = []
    neighbor_urls = []
    for h in hits[:top_neighbors]:
        src = h.get("_source", {})
        meta = src.get("metadata", {}).get("content_metadata", {})
        neighbors.append({"text": src.get("text", ""), "url": meta.get("content_url", "")})
        neighbor_urls.append(meta.get("content_url", ""))
    if not neighbors:
        return []

    context = build_neighborhood_context(passage.text, neighbors, max_tokens=max_context_tokens)
    raw = llm.call(SYNTHESIS_SYSTEM,
                   SYNTHESIS_USER.format(domain=domain, neighborhood_context=context),
                   max_tokens=4096)
    if not raw:
        return []

    pairs = parse_synthesis_response(raw)
    rows: list[KVPRow] = []
    source_revision_id = passage_source_revision_id(passage)
    source_chunk_ids = passage_source_chunk_ids(passage)
    source_systems = passage_source_systems(passage)
    source_kinds = passage_source_kinds(passage)
    modalities = passage_modalities(passage)
    for p in pairs:
        if not p.get("question") or not p.get("answer"):
            continue
        qa_type = p.get("type", "")
        if qa_type not in ("bridging", "contrastive"):
            continue
        rows.append(KVPRow(
            passage_id=passage.passage_id,
            source_url=passage.url,
            product_family=passage.product_family,
            stage="1b",
            qa_type=qa_type,
            question=p["question"].strip(),
            answer=p["answer"].strip(),
            context=context,
            source_revision_ids=[source_revision_id],
            source_chunk_ids=source_chunk_ids,
            source_systems=source_systems or None,
            source_kinds=source_kinds or None,
            modalities=modalities or None,
            neighbor_urls=neighbor_urls,
            refined=False,
        ))
    return rows


def run_stage1b(
    passages: list[Passage], seed_vectors: dict[str, list[float]],
    es: Elasticsearch, index: str, domain: str, llm: LLMClient,
    output_dir: Path, max_workers: int = 5,
    knn_k: int = 6, num_candidates: int = 50, top_neighbors: int = 3,
    max_context_tokens: int = 1200,
) -> list[KVPRow]:
    out_file = output_dir / "stage1b_synthesis.jsonl"
    all_rows: list[KVPRow] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(process_passage_1b, p, seed_vectors.get(p.passage_id, []),
                        es, index, domain, llm, knn_k, num_candidates, top_neighbors,
                        max_context_tokens)
            for p in passages
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage 1B"):
            all_rows.extend(fut.result())

    with out_file.open("w") as f:
        for r in all_rows:
            f.write(r.model_dump_json() + "\n")
    log.info("Stage 1B: %d KVPs → %s", len(all_rows), out_file)
    return all_rows
