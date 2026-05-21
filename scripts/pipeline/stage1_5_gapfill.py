"""Stage 1.5: bias analysis + Data Designer RAG-grounded gap-fill."""
from __future__ import annotations

import json
import logging
import random
import re
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from elasticsearch import Elasticsearch
from tqdm import tqdm

from scripts.pipeline.es_client import knn_search
from scripts.pipeline.llm_client import LLMClient
from scripts.pipeline.models import KVPRow, Passage
from scripts.pipeline.prompts import GAPFILL_RECIPE_USER

log = logging.getLogger(__name__)


def compute_bias_report(passages: list[Passage], kvps: list[KVPRow],
                        threshold_factor: float = 0.5,
                        min_chunks_for_inclusion: int = 10) -> dict:
    """Compute KVP density per product_family, flag under-represented products."""
    chunks_per_family: dict[str, int] = {}
    for p in passages:
        chunks_per_family[p.product_family] = chunks_per_family.get(p.product_family, 0) + 1

    kvps_per_family: dict[str, int] = {}
    for k in kvps:
        kvps_per_family[k.product_family] = kvps_per_family.get(k.product_family, 0) + 1

    densities: dict[str, float] = {}
    for fam, n_chunks in chunks_per_family.items():
        if n_chunks < min_chunks_for_inclusion:
            continue
        n_kvps = kvps_per_family.get(fam, 0)
        densities[fam] = n_kvps / n_chunks

    median_d = statistics.median(densities.values()) if densities else 0.0
    threshold = median_d * threshold_factor

    products = []
    for fam, dens in sorted(densities.items()):
        products.append({
            "product_family": fam,
            "chunk_count": chunks_per_family[fam],
            "kvp_count": kvps_per_family.get(fam, 0),
            "density": round(dens, 4),
            "underrepresented": dens < threshold,
        })

    return {
        "median_density": round(median_d, 4),
        "threshold": round(threshold, 4),
        "threshold_factor": threshold_factor,
        "products": products,
    }


def build_seed_styles(kvps: list[KVPRow], product_family: str, n: int = 3) -> list[str]:
    """Pick n example questions for the given product_family to seed style variation."""
    matching = [k for k in kvps if k.product_family == product_family]
    if not matching:
        return []
    sample = random.sample(matching, min(n, len(matching)))
    return [f"- {k.question}" for k in sample]


def render_recipe_prompt(retrieved_chunks: str, product_family: str,
                         seed_styles: list[str], pairs_count: int = 5) -> str:
    seed_block = "\n".join(seed_styles) if seed_styles else "- (no seed styles available)"
    return GAPFILL_RECIPE_USER.format(
        retrieved_chunks=retrieved_chunks,
        product_family=product_family,
        seed_styles=seed_block,
        pairs_count=pairs_count,
    )


def retrieve_chunks_for_product(
    es: Elasticsearch, index: str, product_family: str,
    target_vector: list[float] | None, top_n: int = 20,
    max_tokens: int = 1500,
) -> list[dict]:
    """Hybrid retrieval: product_family filter + kNN (if vector available)."""
    if target_vector:
        body = {
            "knn": {
                "field": "vector",
                "query_vector": target_vector,
                "k": top_n,
                "num_candidates": 100,
            },
            "query": {"bool": {
                "filter": [{"term": {"metadata.product_family": product_family}}],
            }},
            "_source": ["text", "metadata"],
        }
    else:
        body = {
            "query": {"bool": {
                "filter": [{"term": {"metadata.product_family": product_family}}],
            }},
            "size": top_n,
            "_source": ["text", "metadata"],
        }
    res = es.search(index=index, body=body)
    return res["hits"]["hits"]


def _parse_pairs(raw: str) -> list[dict]:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
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


def gapfill_one_call(
    es: Elasticsearch, index: str, product_family: str,
    seed_styles: list[str], llm: LLMClient,
    top_n_chunks: int = 20, pairs_count: int = 5,
) -> list[KVPRow]:
    hits = retrieve_chunks_for_product(es, index, product_family,
                                        target_vector=None, top_n=top_n_chunks)
    if not hits:
        return []
    retrieved_chunks = "\n\n---\n\n".join(
        h.get("_source", {}).get("text", "") for h in hits
    )
    retrieved_urls = [
        h.get("_source", {}).get("metadata", {}).get("content_metadata", {}).get("content_url", "")
        for h in hits
    ]
    prompt = render_recipe_prompt(retrieved_chunks, product_family, seed_styles,
                                   pairs_count=pairs_count)
    raw = llm.call(
        "You are a precise NVIDIA technical assistant. Generate Q+A pairs grounded ONLY in the provided documentation chunks. Return JSON only.",
        prompt,
        max_tokens=2048,
    )
    if not raw:
        return []

    rows: list[KVPRow] = []
    for p in _parse_pairs(raw):
        if not p.get("question") or not p.get("answer"):
            continue
        rows.append(KVPRow(
            passage_id=f"gapfill#{product_family}",
            source_url="<multi-chunk retrieval>",
            product_family=product_family,
            stage="1.5",
            target_product_family=product_family,
            question=p["question"].strip(),
            answer=p["answer"].strip(),
            context=retrieved_chunks[:5000],
            retrieved_urls=retrieved_urls,
            refined=False,
        ))
    return rows


def run_stage1_5(
    passages: list[Passage], existing_kvps: list[KVPRow],
    es: Elasticsearch, index: str, llm: LLMClient, output_dir: Path,
    threshold_factor: float = 0.5, target_factor: float = 0.8,
    top_n_chunks: int = 20, pairs_per_call: int = 5,
    max_attempt_factor: int = 3,
) -> list[KVPRow]:
    report = compute_bias_report(passages, existing_kvps, threshold_factor=threshold_factor)
    bias_file = output_dir / "bias_report.json"
    with bias_file.open("w") as f:
        json.dump(report, f, indent=2)
    log.info("Stage 1.5: bias report → %s", bias_file)

    target_density = report["median_density"] * target_factor
    underrep = [p for p in report["products"] if p["underrepresented"]]
    log.info("Stage 1.5: %d under-represented products", len(underrep))

    all_rows: list[KVPRow] = []
    out_file = output_dir / "stage1_5_gapfill.jsonl"

    for prod in tqdm(underrep, desc="Stage 1.5 gap-fill"):
        fam = prod["product_family"]
        pairs_needed = max(0, int(target_density * prod["chunk_count"]) - prod["kvp_count"])
        if pairs_needed <= 0:
            continue
        max_attempts = pairs_needed * max_attempt_factor
        attempts = 0
        accepted = 0
        while accepted < pairs_needed and attempts < max_attempts:
            seed_styles = build_seed_styles(existing_kvps + all_rows, fam, n=3)
            new_rows = gapfill_one_call(
                es, index, fam, seed_styles, llm,
                top_n_chunks=top_n_chunks, pairs_count=pairs_per_call,
            )
            if not new_rows:
                attempts += pairs_per_call
                continue
            all_rows.extend(new_rows)
            accepted += len(new_rows)
            attempts += pairs_per_call

    with out_file.open("w") as f:
        for r in all_rows:
            f.write(r.model_dump_json() + "\n")
    log.info("Stage 1.5: %d gap-fill pairs → %s", len(all_rows), out_file)
    return all_rows
