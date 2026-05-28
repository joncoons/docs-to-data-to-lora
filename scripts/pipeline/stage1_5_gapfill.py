"""Stage 1.5: coverage analysis and Data Designer gap-fill planning."""
from __future__ import annotations

import json
import logging
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any

from tqdm import tqdm

from scripts.pipeline.llm_client import LLMClient
from scripts.pipeline.models import KVPRow, Passage
from scripts.pipeline.provenance import SCHEMA_VERSION, sha256_json, stable_id, utc_now
from scripts.pipeline.prompts import GAPFILL_RECIPE_USER

Elasticsearch = Any

log = logging.getLogger(__name__)

ANALYZER_NAME = "stage1_5_product_density_gap_analysis"
ANALYZER_VERSION = "2026-05-28"
DEFAULT_MAX_SEED_IDS = 12


def compute_bias_report(
    passages: list[Passage],
    kvps: list[KVPRow],
    threshold_factor: float = 0.5,
    min_chunks_for_inclusion: int = 10,
) -> dict[str, Any]:
    """Compute KVP density per product_family and flag underrepresented products."""
    chunks_per_family: dict[str, int] = {}
    for passage in passages:
        chunks_per_family[passage.product_family] = (
            chunks_per_family.get(passage.product_family, 0) + 1
        )

    kvps_per_family: dict[str, int] = {}
    for row in kvps:
        kvps_per_family[row.product_family] = kvps_per_family.get(row.product_family, 0) + 1

    densities: dict[str, float] = {}
    for family, chunk_count in chunks_per_family.items():
        if chunk_count < min_chunks_for_inclusion:
            continue
        kvp_count = kvps_per_family.get(family, 0)
        densities[family] = kvp_count / chunk_count

    median_density = statistics.median(densities.values()) if densities else 0.0
    threshold = median_density * threshold_factor

    products = []
    for family, density in sorted(densities.items()):
        products.append({
            "product_family": family,
            "chunk_count": chunks_per_family[family],
            "kvp_count": kvps_per_family.get(family, 0),
            "density": round(density, 4),
            "underrepresented": density < threshold,
        })

    return {
        "median_density": round(median_density, 4),
        "threshold": round(threshold, 4),
        "threshold_factor": threshold_factor,
        "products": products,
    }


def build_seed_styles(kvps: list[KVPRow], product_family: str, n: int = 3) -> list[str]:
    """Pick example questions for a product_family to seed style variation."""
    matching = [row for row in kvps if row.product_family == product_family]
    if not matching:
        return []
    sample = random.sample(matching, min(n, len(matching)))
    return [f"- {row.question}" for row in sample]


def render_recipe_prompt(
    retrieved_chunks: str,
    product_family: str,
    seed_styles: list[str],
    pairs_count: int = 5,
) -> str:
    seed_block = "\n".join(seed_styles) if seed_styles else "- (no seed styles available)"
    return GAPFILL_RECIPE_USER.format(
        retrieved_chunks=retrieved_chunks,
        product_family=product_family,
        seed_styles=seed_block,
        pairs_count=pairs_count,
    )


def retrieve_chunks_for_product(
    es: Elasticsearch,
    index: str,
    product_family: str,
    target_vector: list[float] | None,
    top_n: int = 20,
    max_tokens: int = 1500,
) -> list[dict[str, Any]]:
    """Hybrid retrieval: product_family filter plus kNN when a target vector exists."""
    _ = max_tokens
    if target_vector:
        body = {
            "knn": {
                "field": "vector",
                "query_vector": target_vector,
                "k": top_n,
                "num_candidates": 100,
            },
            "query": {
                "bool": {
                    "filter": [{"term": {"metadata.product_family": product_family}}],
                }
            },
            "_source": ["text", "metadata"],
        }
    else:
        body = {
            "query": {
                "bool": {
                    "filter": [{"term": {"metadata.product_family": product_family}}],
                }
            },
            "size": top_n,
            "_source": ["text", "metadata"],
        }
    res = es.search(index=index, body=body)
    return res["hits"]["hits"]


def _parse_pairs(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return obj.get("pairs", [])


def _analysis_config(
    *,
    collection: str,
    threshold_factor: float,
    target_factor: float,
    top_n_chunks: int,
    pairs_per_call: int,
    max_attempt_factor: int,
) -> dict[str, Any]:
    return {
        "collection": collection,
        "threshold_factor": threshold_factor,
        "target_factor": target_factor,
        "top_n_chunks": top_n_chunks,
        "pairs_per_call": pairs_per_call,
        "max_attempt_factor": max_attempt_factor,
    }


def _source_chunk_ids_for_family(
    passages: list[Passage],
    product_family: str,
    limit: int = DEFAULT_MAX_SEED_IDS,
) -> list[str]:
    seen: list[str] = []
    for passage in passages:
        if passage.product_family != product_family:
            continue
        chunk_ids = passage.source_chunk_ids or passage.chunk_ids
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.append(chunk_id)
            if len(seen) >= limit:
                return seen
    return seen


def _entailment_ids_for_family(
    kvps: list[KVPRow],
    product_family: str,
    limit: int = DEFAULT_MAX_SEED_IDS,
) -> list[str]:
    seen: list[str] = []
    for row in kvps:
        if row.product_family != product_family or not row.entailment_id:
            continue
        if row.entailment_id not in seen:
            seen.append(row.entailment_id)
        if len(seen) >= limit:
            break
    return seen


def _severity(observed_count: int, target_count: int) -> str:
    if target_count <= 0:
        return "low"
    missing_ratio = max(0, target_count - observed_count) / target_count
    if observed_count == 0 and target_count >= 10:
        return "critical"
    if missing_ratio >= 0.75:
        return "high"
    if missing_ratio >= 0.5:
        return "medium"
    return "low"


def _default_dataset_version_id(
    collection: str,
    report: dict[str, Any],
    existing_kvps: list[KVPRow],
) -> str:
    return stable_id(
        "dsv",
        "source_grounded_pre_gapfill",
        collection,
        report,
        [row.sample_id or row.question for row in existing_kvps[:100]],
        len(existing_kvps),
    )


def build_gap_manifest(
    passages: list[Passage],
    existing_kvps: list[KVPRow],
    report: dict[str, Any],
    *,
    collection: str,
    dataset_version_id: str | None = None,
    threshold_factor: float = 0.5,
    target_factor: float = 0.8,
    top_n_chunks: int = 20,
    pairs_per_call: int = 5,
    max_attempt_factor: int = 3,
) -> dict[str, Any]:
    """Build the provenance gap_manifest consumed by NeMo Data Designer."""
    analysis_config = _analysis_config(
        collection=collection,
        threshold_factor=threshold_factor,
        target_factor=target_factor,
        top_n_chunks=top_n_chunks,
        pairs_per_call=pairs_per_call,
        max_attempt_factor=max_attempt_factor,
    )
    dataset_version_id = dataset_version_id or _default_dataset_version_id(
        collection,
        report,
        existing_kvps,
    )
    target_density = float(report.get("median_density", 0.0)) * target_factor
    gaps: list[dict[str, Any]] = []

    for product in report.get("products", []):
        if not product.get("underrepresented"):
            continue
        product_family = str(product["product_family"])
        observed_count = int(product.get("kvp_count", 0))
        target_count = max(
            observed_count,
            math.ceil(target_density * int(product.get("chunk_count", 0))),
        )
        if target_count <= observed_count:
            continue
        gap_id = stable_id("gap", product_family)
        missing = target_count - observed_count
        gaps.append({
            "gap_id": gap_id,
            "dimension": {
                "collection": collection,
                "product": product_family,
                "microservice": product_family,
            },
            "severity": _severity(observed_count, target_count),
            "observed_count": observed_count,
            "target_count": target_count,
            "seed_entailment_ids": _entailment_ids_for_family(existing_kvps, product_family),
            "seed_chunk_ids": _source_chunk_ids_for_family(passages, product_family),
            "recommendation": "generate_synthetic",
            "generation_brief": (
                f"Generate approximately {missing} source-grounded QA pairs for "
                f"{product_family}. Use retrieved documentation chunks only, preserve "
                "question style diversity from seed examples, and emit JSON pairs."
            ),
        })

    gap_manifest_id = stable_id(
        "gapmanifest",
        dataset_version_id,
        sha256_json(analysis_config),
        [gap["gap_id"] for gap in gaps],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "gap_manifest_id": gap_manifest_id,
        "dataset_version_id": dataset_version_id,
        "created_at": utc_now(),
        "analysis": {
            "analyzer_name": ANALYZER_NAME,
            "analyzer_version": ANALYZER_VERSION,
            "config_hash": sha256_json(analysis_config),
        },
        "gaps": gaps,
    }


def data_designer_recipe_name(collection: str) -> str:
    if collection == "nemo_usvcs_curated":
        return "nemo-usvcs-gapfill"
    return "nim-gapfill"


def _hit_text(hit: dict[str, Any]) -> str:
    return str(hit.get("_source", {}).get("text", ""))


def _hit_url(hit: dict[str, Any]) -> str:
    metadata = hit.get("_source", {}).get("metadata", {})
    content_metadata = metadata.get("content_metadata", {}) if isinstance(metadata, dict) else {}
    return str(content_metadata.get("content_url") or metadata.get("url") or "")


def _join_retrieved_chunks(hits: list[dict[str, Any]], max_chars: int) -> str:
    chunks: list[str] = []
    total = 0
    for hit in hits:
        text = _hit_text(hit).strip()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        chunk = text[:remaining]
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(chunks)


def build_data_designer_requests(
    gap_manifest: dict[str, Any],
    existing_kvps: list[KVPRow],
    es: Elasticsearch,
    index: str,
    *,
    recipe_name: str | None = None,
    top_n_chunks: int = 20,
    pairs_per_call: int = 5,
    max_context_chars: int = 6000,
) -> list[dict[str, Any]]:
    """Materialize seed records that a native Data Designer job can consume."""
    recipe_name = recipe_name or data_designer_recipe_name(index)
    requests: list[dict[str, Any]] = []
    for gap in gap_manifest.get("gaps", []):
        if gap.get("recommendation") != "generate_synthetic":
            continue
        product_family = str(gap.get("dimension", {}).get("product") or "")
        if not product_family:
            continue
        hits = retrieve_chunks_for_product(
            es,
            index,
            product_family,
            target_vector=None,
            top_n=top_n_chunks,
        )
        retrieved_chunks = _join_retrieved_chunks(hits, max_chars=max_context_chars)
        pairs_needed = max(0, int(gap["target_count"]) - int(gap["observed_count"]))
        records_needed = max(1, math.ceil(pairs_needed / pairs_per_call))
        seed_styles = build_seed_styles(existing_kvps, product_family, n=3)
        requests.append({
            "gap_id": gap["gap_id"],
            "gap_manifest_id": gap_manifest["gap_manifest_id"],
            "dataset_version_id": gap_manifest["dataset_version_id"],
            "recipe_name": recipe_name,
            "num_records": records_needed,
            "pairs_needed": pairs_needed,
            "pairs_per_record": pairs_per_call,
            "input": {
                "gap_id": gap["gap_id"],
                "product_family": product_family,
                "pairs_count": pairs_per_call,
                "seed_styles": "\n".join(seed_styles),
                "retrieved_chunks": retrieved_chunks,
                "generation_brief": gap.get("generation_brief"),
            },
            "retrieved_urls": [url for url in (_hit_url(hit) for hit in hits) if url],
            "seed_entailment_ids": gap.get("seed_entailment_ids", []),
            "seed_chunk_ids": gap.get("seed_chunk_ids", []),
        })
    return requests


def write_gap_analysis_artifacts(
    output_dir: Path,
    report: dict[str, Any],
    gap_manifest: dict[str, Any],
    data_designer_requests: list[dict[str, Any]],
) -> None:
    provenance_dir = output_dir / "provenance"
    data_designer_dir = output_dir / "data_designer"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    data_designer_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "bias_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (provenance_dir / "gap_manifest.json").write_text(
        json.dumps(gap_manifest, indent=2, sort_keys=True) + "\n"
    )
    requests_path = data_designer_dir / "gapfill_requests.jsonl"
    with requests_path.open("w") as f:
        for request in data_designer_requests:
            f.write(json.dumps(request, sort_keys=True) + "\n")

    request_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "gap_manifest_id": gap_manifest["gap_manifest_id"],
        "dataset_version_id": gap_manifest["dataset_version_id"],
        "requests_uri": "data_designer/gapfill_requests.jsonl",
        "request_count": len(data_designer_requests),
        "records_requested": sum(int(item["num_records"]) for item in data_designer_requests),
        "pairs_requested": sum(int(item["pairs_needed"]) for item in data_designer_requests),
        "status": "planned",
        "native_service": "NeMo Data Designer",
    }
    (data_designer_dir / "request_manifest.json").write_text(
        json.dumps(request_manifest, indent=2, sort_keys=True) + "\n"
    )


def gapfill_one_call(
    es: Elasticsearch,
    index: str,
    product_family: str,
    seed_styles: list[str],
    llm: LLMClient,
    top_n_chunks: int = 20,
    pairs_count: int = 5,
) -> list[KVPRow]:
    """Legacy direct LLM fallback for gap-fill generation."""
    hits = retrieve_chunks_for_product(
        es,
        index,
        product_family,
        target_vector=None,
        top_n=top_n_chunks,
    )
    if not hits:
        return []
    retrieved_chunks = "\n\n---\n\n".join(_hit_text(hit) for hit in hits)
    retrieved_urls = [_hit_url(hit) for hit in hits]
    prompt = render_recipe_prompt(
        retrieved_chunks,
        product_family,
        seed_styles,
        pairs_count=pairs_count,
    )
    raw = llm.call(
        "You are a precise NVIDIA technical assistant. Generate Q+A pairs grounded "
        "ONLY in the provided documentation chunks. Return JSON only.",
        prompt,
        max_tokens=4096,
    )
    if not raw:
        return []

    rows: list[KVPRow] = []
    for pair in _parse_pairs(raw):
        if not pair.get("question") or not pair.get("answer"):
            continue
        rows.append(KVPRow(
            passage_id=f"gapfill#{product_family}",
            source_url="<multi-chunk retrieval>",
            product_family=product_family,
            stage="1.5",
            target_product_family=product_family,
            sample_id=stable_id(
                "sample",
                "1.5",
                product_family,
                pair["question"],
                pair["answer"],
            ),
            question=pair["question"].strip(),
            answer=pair["answer"].strip(),
            context=retrieved_chunks[:5000],
            retrieved_urls=retrieved_urls,
            refined=False,
            source_systems=["legacy_direct_llm_gapfill"],
            source_kinds=["synthetic_gapfill"],
            modalities=["text"],
        ))
    return rows


def _run_legacy_direct_gapfill(
    report: dict[str, Any],
    existing_kvps: list[KVPRow],
    es: Elasticsearch,
    index: str,
    llm: LLMClient,
    output_dir: Path,
    *,
    target_factor: float,
    top_n_chunks: int,
    pairs_per_call: int,
    max_attempt_factor: int,
) -> list[KVPRow]:
    target_density = float(report["median_density"]) * target_factor
    underrepresented = [p for p in report["products"] if p["underrepresented"]]
    log.info("Stage 1.5 legacy direct gap-fill: %d products", len(underrepresented))

    all_rows: list[KVPRow] = []
    out_file = output_dir / "stage1_5_gapfill.jsonl"

    for product in tqdm(underrepresented, desc="Stage 1.5 legacy gap-fill"):
        family = product["product_family"]
        pairs_needed = max(
            0,
            math.ceil(target_density * product["chunk_count"]) - product["kvp_count"],
        )
        if pairs_needed <= 0:
            continue
        max_attempts = pairs_needed * max_attempt_factor
        attempts = 0
        accepted = 0
        while accepted < pairs_needed and attempts < max_attempts:
            seed_styles = build_seed_styles(existing_kvps + all_rows, family, n=3)
            new_rows = gapfill_one_call(
                es,
                index,
                family,
                seed_styles,
                llm,
                top_n_chunks=top_n_chunks,
                pairs_count=pairs_per_call,
            )
            attempts += pairs_per_call
            if not new_rows:
                continue
            all_rows.extend(new_rows)
            accepted += len(new_rows)

    with out_file.open("w") as f:
        for row in all_rows:
            f.write(row.model_dump_json() + "\n")
    log.info("Stage 1.5 legacy direct gap-fill: %d pairs -> %s", len(all_rows), out_file)
    return all_rows


def run_stage1_5_analysis(
    passages: list[Passage],
    existing_kvps: list[KVPRow],
    es: Elasticsearch,
    index: str,
    output_dir: Path,
    *,
    threshold_factor: float = 0.5,
    target_factor: float = 0.8,
    top_n_chunks: int = 20,
    pairs_per_call: int = 5,
    max_attempt_factor: int = 3,
    dataset_version_id: str | None = None,
    recipe_name: str | None = None,
) -> dict[str, Any]:
    report = compute_bias_report(passages, existing_kvps, threshold_factor=threshold_factor)
    gap_manifest = build_gap_manifest(
        passages,
        existing_kvps,
        report,
        collection=index,
        dataset_version_id=dataset_version_id,
        threshold_factor=threshold_factor,
        target_factor=target_factor,
        top_n_chunks=top_n_chunks,
        pairs_per_call=pairs_per_call,
        max_attempt_factor=max_attempt_factor,
    )
    data_designer_requests = build_data_designer_requests(
        gap_manifest,
        existing_kvps,
        es,
        index,
        recipe_name=recipe_name,
        top_n_chunks=top_n_chunks,
        pairs_per_call=pairs_per_call,
    )
    write_gap_analysis_artifacts(output_dir, report, gap_manifest, data_designer_requests)
    log.info(
        "Stage 1.5: %d gaps -> %s",
        len(gap_manifest["gaps"]),
        output_dir / "provenance" / "gap_manifest.json",
    )
    log.info(
        "Stage 1.5: %d Data Designer requests -> %s",
        len(data_designer_requests),
        output_dir / "data_designer" / "gapfill_requests.jsonl",
    )
    return {
        "bias_report": report,
        "gap_manifest": gap_manifest,
        "data_designer_requests": data_designer_requests,
    }


def run_stage1_5(
    passages: list[Passage],
    existing_kvps: list[KVPRow],
    es: Elasticsearch,
    index: str,
    llm: LLMClient,
    output_dir: Path,
    threshold_factor: float = 0.5,
    target_factor: float = 0.8,
    top_n_chunks: int = 20,
    pairs_per_call: int = 5,
    max_attempt_factor: int = 3,
    legacy_direct: bool = False,
) -> list[KVPRow]:
    analysis = run_stage1_5_analysis(
        passages,
        existing_kvps,
        es,
        index,
        output_dir,
        threshold_factor=threshold_factor,
        target_factor=target_factor,
        top_n_chunks=top_n_chunks,
        pairs_per_call=pairs_per_call,
        max_attempt_factor=max_attempt_factor,
    )
    if not legacy_direct:
        out_file = output_dir / "stage1_5_gapfill.jsonl"
        out_file.write_text("")
        log.info("Stage 1.5: generation deferred to NeMo Data Designer -> %s", out_file)
        return []

    return _run_legacy_direct_gapfill(
        analysis["bias_report"],
        existing_kvps,
        es,
        index,
        llm,
        output_dir,
        target_factor=target_factor,
        top_n_chunks=top_n_chunks,
        pairs_per_call=pairs_per_call,
        max_attempt_factor=max_attempt_factor,
    )
