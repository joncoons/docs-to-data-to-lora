"""Stage 0: ES scroll → grouped passages → noise filter → passages.jsonl."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from elasticsearch import Elasticsearch
from tqdm import tqdm

from scripts.pipeline.es_client import scroll_all_chunks
from scripts.pipeline.models import Passage
from scripts.pipeline.noise_filter import count_tokens, is_noise

log = logging.getLogger(__name__)

_BINARY_EXTENSIONS = (".pdf", ".docx", ".pptx", ".doc", ".ppt")


def classify_doc_kind(url: str, doc_type: str) -> Literal["html", "pdf"]:
    """URL extension is authoritative; doc_type is informational."""
    url_lower = url.lower().split("?")[0].split("#")[0]
    if any(url_lower.endswith(ext) for ext in _BINARY_EXTENSIONS):
        return "pdf"
    return "html"


def extract_chunk_dict(hit: dict) -> dict | None:
    """Flatten an ES hit into our internal chunk dict; None if essential fields missing."""
    src = hit.get("_source", {})
    meta = src.get("metadata", {})
    cm = meta.get("content_metadata", {})

    url = cm.get("content_url") or meta.get("source", {}).get("source_name", "")
    text = (src.get("text") or "").strip()
    if not url or not text:
        return None

    return {
        "_id": hit["_id"],
        "url": url,
        "chunk_index": cm.get("chunk_index", 0),
        "text": text,
        "vector": src.get("vector"),
        "doc_type": cm.get("document_type", "text"),
        "product_family": meta.get("product_family") or cm.get("product_family") or "unknown",
        "product_name": meta.get("product_name") or cm.get("product_name") or "unknown",
    }


def group_html_chunks_by_url(chunks: list[dict]) -> dict[str, list[dict]]:
    """For each URL, concatenate all chunks' text into a single passage.
    Returns {url: [passage_dict]} (always one entry per URL for HTML)."""
    by_url: dict[str, list[dict]] = {}
    for c in chunks:
        url = c["url"]
        by_url.setdefault(url, []).append(c)

    grouped: dict[str, list[dict]] = {}
    for url, cks in by_url.items():
        cks.sort(key=lambda x: x["chunk_index"])
        text = " ".join(c["text"] for c in cks).strip()
        grouped[url] = [{
            "url": url,
            "text": text,
            "chunk_ids": [c["_id"] for c in cks],
            "seed_vector": cks[0].get("vector"),
            "product_family": cks[0].get("product_family", "unknown"),
            "product_name": cks[0].get("product_name", "unknown"),
            "doc_kind": "html",
        }]
    return grouped


def build_passages(chunks: list[dict], min_passage_tokens: int = 60) -> list[Passage]:
    """Group HTML chunks by URL (concat); emit PDF chunks as-is. Apply noise filter."""
    # Normalize ES hits to internal dicts if they look like hits
    normalized: list[dict] = []
    for c in chunks:
        if "_source" in c:
            d = extract_chunk_dict(c)
            if d is not None:
                normalized.append(d)
        else:
            normalized.append(c)

    # Classify each chunk
    for c in normalized:
        c["doc_kind"] = classify_doc_kind(c["url"], c.get("doc_type", "text"))

    html_chunks = [c for c in normalized if c["doc_kind"] == "html"]
    pdf_chunks  = [c for c in normalized if c["doc_kind"] == "pdf"]

    passages: list[Passage] = []
    # HTML: group by URL, concat, one passage per URL
    grouped = group_html_chunks_by_url(html_chunks)
    for url, plist in grouped.items():
        for p_idx, p in enumerate(plist):
            text = p["text"]
            if is_noise(text, url=url, min_tokens=min_passage_tokens):
                continue
            passages.append(Passage(
                passage_id=f"{url}#p{p_idx}",
                url=url,
                text=text,
                token_count=count_tokens(text),
                chunk_ids=p["chunk_ids"],
                product_family=p["product_family"],
                product_name=p["product_name"],
                doc_kind="html",
            ))

    # PDF: one passage per chunk
    for i, c in enumerate(pdf_chunks):
        text = c["text"]
        if is_noise(text, url=c["url"], min_tokens=min_passage_tokens):
            continue
        passages.append(Passage(
            passage_id=f"{c['url']}#c{c['chunk_index']}",
            url=c["url"],
            text=text,
            token_count=count_tokens(text),
            chunk_ids=[c["_id"]],
            product_family=c["product_family"],
            product_name=c["product_name"],
            doc_kind="pdf",
        ))

    return passages


def run_stage0(es: Elasticsearch, index: str, output_dir: Path,
               min_passage_tokens: int = 60) -> tuple[list[Passage], dict[str, list[float]]]:
    """Scroll the index, build passages, write JSONL.

    Returns (passages, seed_vectors_by_passage_id).
    Seed vectors are kept in memory only — they're large and only needed at runtime.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "passages.jsonl"

    log.info("Stage 0: scrolling ES index '%s'...", index)
    raw_hits = list(tqdm(scroll_all_chunks(es, index=index), desc="ES scroll"))
    log.info("Stage 0: %d hits retrieved", len(raw_hits))

    chunks: list[dict] = []
    for h in raw_hits:
        d = extract_chunk_dict(h)
        if d is not None:
            chunks.append(d)
    log.info("Stage 0: %d chunks after extraction", len(chunks))

    passages = build_passages(chunks, min_passage_tokens=min_passage_tokens)
    log.info("Stage 0: %d passages after grouping + noise filter", len(passages))

    seed_vectors: dict[str, list[float]] = {}
    for p in passages:
        # Find any chunk with a vector
        for c in chunks:
            if c["_id"] in p.chunk_ids and c.get("vector"):
                seed_vectors[p.passage_id] = c["vector"]
                break

    with out_file.open("w") as f:
        for p in passages:
            f.write(p.model_dump_json() + "\n")
    log.info("Stage 0: written %s", out_file)

    return passages, seed_vectors
