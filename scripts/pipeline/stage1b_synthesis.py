"""Stage 1B: kNN neighborhood synthesis (BRIDGING + CONTRASTIVE pairs)."""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def process_passage_1b_result(
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
) -> tuple[str, list[KVPRow], str | None]:
    if not seed_vector:
        return "no_seed_vector", [], None
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
        return "no_neighbors", [], None

    context = build_neighborhood_context(passage.text, neighbors, max_tokens=max_context_tokens)
    raw = llm.call(SYNTHESIS_SYSTEM,
                   SYNTHESIS_USER.format(domain=domain, neighborhood_context=context),
                   max_tokens=4096)
    if not raw:
        return "llm_empty", [], None

    pairs = parse_synthesis_response(raw)
    if not pairs:
        return "no_pairs", [], None

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
    if not rows:
        return "no_valid_pairs", [], None
    return "complete", rows, None


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
    _, rows, _ = process_passage_1b_result(
        passage, seed_vector, es, index, domain, llm,
        k=k, num_candidates=num_candidates, top_neighbors=top_neighbors,
        max_context_tokens=max_context_tokens,
    )
    return rows


def read_stage1b_rows(path: Path) -> list[KVPRow]:
    if not path.exists():
        return []
    return [KVPRow.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def latest_stage1b_statuses(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[str(row["passage_id"])] = row
    return latest


def append_stage1b_rows(stream: Any, rows: list[KVPRow]) -> None:
    for row in rows:
        stream.write(row.model_dump_json() + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def append_stage1b_status(
    stream: Any,
    passage: Passage,
    *,
    status: str,
    row_count: int,
    error: str | None = None,
) -> None:
    payload = {
        "finished_at": utc_now(),
        "passage_id": passage.passage_id,
        "source_url": passage.url,
        "status": status,
        "row_count": row_count,
        "error": error,
    }
    stream.write(json.dumps(payload, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run_stage1b(
    passages: list[Passage], seed_vectors: dict[str, list[float]],
    es: Elasticsearch, index: str, domain: str, llm: LLMClient,
    output_dir: Path, max_workers: int = 5,
    knn_k: int = 6, num_candidates: int = 50, top_neighbors: int = 3,
    max_context_tokens: int = 1200, resume: bool = False,
) -> list[KVPRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "stage1b_synthesis.jsonl"
    status_file = output_dir / "stage1b_passage_results.jsonl"
    if not resume:
        out_file.unlink(missing_ok=True)
        status_file.unlink(missing_ok=True)

    existing_rows = read_stage1b_rows(out_file)
    rows_by_passage = {row.passage_id for row in existing_rows}
    latest_status = latest_stage1b_statuses(status_file)
    terminal_no_row_statuses = {"no_seed_vector", "no_neighbors"}
    pending: list[Passage] = []
    skipped = 0
    for passage in passages:
        status = str(latest_status.get(passage.passage_id, {}).get("status") or "")
        if passage.passage_id in rows_by_passage or status in terminal_no_row_statuses:
            skipped += 1
            continue
        pending.append(passage)

    if skipped:
        log.info("Stage 1B: resuming with %d skipped passages and %d pending", skipped, len(pending))
    all_rows: list[KVPRow] = list(existing_rows)
    if not pending:
        log.info("Stage 1B: %d KVPs already available at %s", len(all_rows), out_file)
        return all_rows

    with out_file.open("a", encoding="utf-8") as row_stream, status_file.open("a", encoding="utf-8") as status_stream:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    process_passage_1b_result,
                    p,
                    seed_vectors.get(p.passage_id, []),
                    es,
                    index,
                    domain,
                    llm,
                    knn_k,
                    num_candidates,
                    top_neighbors,
                    max_context_tokens,
                ): p
                for p in pending
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage 1B"):
                passage = futures[fut]
                try:
                    status, rows, error = fut.result()
                except Exception as exc:  # noqa: BLE001 - recorded for durable retry.
                    status, rows, error = "exception", [], str(exc)
                    log.warning("Stage 1B passage failed: %s: %s", passage.passage_id, exc)
                if rows:
                    append_stage1b_rows(row_stream, rows)
                    all_rows.extend(rows)
                append_stage1b_status(
                    status_stream,
                    passage,
                    status=status,
                    row_count=len(rows),
                    error=error,
                )

    log.info("Stage 1B: %d KVPs -> %s", len(all_rows), out_file)
    return all_rows
