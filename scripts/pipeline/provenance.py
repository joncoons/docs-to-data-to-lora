"""Provenance models and deterministic IDs for the dataset pipeline."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scripts.pipeline.models import KVPRow, Passage

SCHEMA_VERSION = "provenance.v1"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def source_revision_id_for_text(url: str, normalized_text: str) -> str:
    return stable_id("srcrev", url, sha256_text(normalized_text))


def source_chunk_id_for_text(source_revision_id: str, passage_id: str, text: str) -> str:
    return stable_id("chunk", source_revision_id, passage_id, sha256_text(text))


def passage_source_revision_id(passage: Passage) -> str:
    if passage.source_revision_id:
        return passage.source_revision_id
    return source_revision_id_for_text(passage.url, passage.text)


def passage_source_chunk_ids(passage: Passage) -> list[str]:
    if passage.source_chunk_ids:
        return passage.source_chunk_ids
    revision_id = passage_source_revision_id(passage)
    return [source_chunk_id_for_text(revision_id, passage.passage_id, passage.text)]


def entailment_id_for_passage(
    passage: Passage,
    entailment_index: int,
    claim: str,
    premises: list[str],
) -> str:
    return stable_id(
        "ent",
        passage_source_revision_id(passage),
        passage.passage_id,
        entailment_index,
        claim,
        premises,
    )


def sample_id_for_row(row: KVPRow) -> str:
    return stable_id(
        "sample",
        row.stage,
        row.source_url,
        row.question,
        row.answer,
        row.entailment_id,
        row.qa_type,
        row.instr_type,
        row.target_product_family,
    )


class ProvenanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CrawlRun(ProvenanceModel):
    schema_version: Literal["provenance.v1"] = SCHEMA_VERSION
    crawl_run_id: str
    previous_crawl_run_id: str | None = None
    started_at: str
    completed_at: str | None = None
    status: Literal["running", "completed", "failed", "partial"] = "completed"
    recrawl_reason: Literal["initial", "scheduled", "manual", "hotfix", "backfill", "test"] = "initial"
    crawler: dict[str, Any]
    scope: dict[str, Any]
    outputs: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)


class SourceRevision(ProvenanceModel):
    schema_version: Literal["provenance.v1"] = SCHEMA_VERSION
    source_revision_id: str
    previous_source_revision_id: str | None = None
    crawl_run_id: str
    canonical_url: str
    final_url: str | None = None
    retrieved_at: str
    status: Literal["active", "unchanged", "changed", "deleted", "redirected", "failed", "excluded"] = "active"
    http: dict[str, Any] = Field(default_factory=dict)
    hashes: dict[str, str | None]
    content: dict[str, Any] = Field(default_factory=dict)
    classification: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


class SourceChunk(ProvenanceModel):
    schema_version: Literal["provenance.v1"] = SCHEMA_VERSION
    chunk_id: str
    source_revision_id: str
    previous_chunk_id: str | None = None
    chunker: dict[str, Any]
    anchors: dict[str, Any] = Field(default_factory=dict)
    spans: dict[str, int | None] = Field(default_factory=dict)
    text_sha256: str
    text: str
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entailment(ProvenanceModel):
    schema_version: Literal["provenance.v1"] = SCHEMA_VERSION
    entailment_id: str
    supersedes_entailment_id: str | None = None
    claim: str
    premises: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]]
    status: Literal["active", "superseded", "retracted", "rejected", "needs_review"] = "active"
    status_reason: str | None = None
    extractor: dict[str, Any]
    validator: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetSample(ProvenanceModel):
    schema_version: Literal["provenance.v1"] = SCHEMA_VERSION
    sample_id: str
    origin: Literal["source_entailed", "synthetic_gapfill", "human_reviewed", "imported_baseline"]
    task_type: Literal["qa", "summary", "listicle", "procedural", "bridging", "contrastive", "rag_eval", "other"]
    prompt: str
    completion: str
    system: str | None = None
    lineage: dict[str, Any]
    quality: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_crawl_run(
    *,
    index: str,
    started_at: str,
    completed_at: str,
    chunk_count: int,
    passage_count: int,
    min_passage_tokens: int,
) -> CrawlRun:
    config = {
        "index": index,
        "min_passage_tokens": min_passage_tokens,
        "chunker": "stage0-url-grouping",
    }
    crawl_run_id = stable_id("crawlrun", index, started_at)
    return CrawlRun(
        crawl_run_id=crawl_run_id,
        started_at=started_at,
        completed_at=completed_at,
        crawler={
            "name": "docs-to-data-to-lora-stage0",
            "version": "v1",
            "git_commit": "unknown",
            "config_hash": sha256_json(config),
        },
        scope={
            "seed_url_set_hash": sha256_json({"index": index}),
            "allowed_url_prefixes": [],
            "excluded_url_patterns": [],
            "binary_host_allowlist": [],
        },
        outputs={
            "source_revisions_uri": "provenance/source_revisions.jsonl",
            "source_chunks_uri": "provenance/source_chunks.jsonl",
            "delta_manifest_uri": None,
        },
        metrics={
            "raw_chunk_count": chunk_count,
            "passage_count": passage_count,
        },
    )


def attach_source_provenance(
    passages: list[Passage],
    *,
    crawl_run_id: str,
    retrieved_at: str,
    chunker_config: dict[str, Any] | None = None,
) -> tuple[list[Passage], list[SourceRevision], list[SourceChunk]]:
    """Return passage copies with source IDs plus source revision/chunk ledgers."""
    chunker_config = chunker_config or {"name": "stage0-url-grouping", "version": "v1"}
    chunker = {
        "name": chunker_config.get("name", "stage0-url-grouping"),
        "version": chunker_config.get("version", "v1"),
        "config_hash": sha256_json(chunker_config),
        "extraction_method": chunker_config.get("extraction_method", "es-scroll-url-grouping"),
    }

    text_by_url: dict[str, list[str]] = {}
    first_by_url: dict[str, Passage] = {}
    for passage in passages:
        text_by_url.setdefault(passage.url, []).append(passage.text)
        first_by_url.setdefault(passage.url, passage)

    revisions: list[SourceRevision] = []
    revision_id_by_url: dict[str, str] = {}
    for url in sorted(text_by_url):
        normalized_text = "\n\n".join(text_by_url[url])
        revision_id = source_revision_id_for_text(url, normalized_text)
        revision_id_by_url[url] = revision_id
        first = first_by_url[url]
        revisions.append(SourceRevision(
            source_revision_id=revision_id,
            crawl_run_id=crawl_run_id,
            canonical_url=url,
            final_url=url,
            retrieved_at=retrieved_at,
            hashes={
                "raw_sha256": None,
                "normalized_sha256": sha256_text(normalized_text),
            },
            content={
                "raw_uri": None,
                "normalized_uri": None,
                "language": None,
                "title": None,
            },
            classification={
                "product_family": first.product_family,
                "product_name": first.product_name,
                "version": None,
                "doc_kind": first.doc_kind,
            },
        ))

    updated: list[Passage] = []
    source_chunks: list[SourceChunk] = []
    for passage in passages:
        revision_id = revision_id_by_url[passage.url]
        chunk_id = source_chunk_id_for_text(revision_id, passage.passage_id, passage.text)
        updated.append(passage.model_copy(update={
            "source_revision_id": revision_id,
            "source_chunk_ids": [chunk_id],
        }))
        source_chunks.append(SourceChunk(
            chunk_id=chunk_id,
            source_revision_id=revision_id,
            chunker=chunker,
            anchors={
                "heading_path": [],
                "dom_path": None,
                "markdown_anchor": None,
                "page_number": None,
                "parser_element_id": None,
            },
            spans={
                "byte_start": None,
                "byte_end": None,
                "char_start": 0,
                "char_end": len(passage.text),
            },
            text_sha256=sha256_text(passage.text),
            text=passage.text,
            token_count=passage.token_count,
            metadata={
                "passage_id": passage.passage_id,
                "source_url": passage.url,
                "external_chunk_ids": passage.chunk_ids,
                "product_family": passage.product_family,
                "product_name": passage.product_name,
                "doc_kind": passage.doc_kind,
            },
        ))

    return updated, revisions, source_chunks


def entailments_from_kvp_rows(rows: list[KVPRow]) -> list[Entailment]:
    seen: dict[str, Entailment] = {}
    created_at = utc_now()
    for row in rows:
        if not row.entailment_id:
            continue
        revision_ids = row.source_revision_ids or [
            source_revision_id_for_text(row.source_url, row.context)
        ]
        chunk_ids = row.source_chunk_ids or [
            source_chunk_id_for_text(revision_ids[0], row.passage_id, row.context)
        ]
        evidence = []
        for chunk_id in chunk_ids:
            source_revision_id = revision_ids[0] if revision_ids else ""
            evidence.append({
                "chunk_id": chunk_id,
                "source_revision_id": source_revision_id,
                "quote": None,
                "char_start": None,
                "char_end": None,
                "support_role": "primary",
            })
        seen.setdefault(row.entailment_id, Entailment(
            entailment_id=row.entailment_id,
            claim=row.entailment_claim or row.answer,
            premises=row.entailment_premises or [],
            evidence=evidence,
            extractor={
                "model": row.extractor_model or "unknown",
                "prompt_hash": row.extractor_prompt_hash or sha256_text("unknown"),
                "temperature": row.extractor_temperature,
                "created_at": created_at,
            },
            metadata={
                "stage": row.stage,
                "passage_id": row.passage_id,
                "source_url": row.source_url,
                "product_family": row.product_family,
            },
        ))
    return list(seen.values())


def dataset_sample_from_kvp_row(row: KVPRow, system_prompt: str | None = None) -> DatasetSample:
    origin = "synthetic_gapfill" if row.stage == "1.5" else "source_entailed"
    if row.qa_type in {"bridging", "contrastive"}:
        task_type = row.qa_type
    elif row.instr_type in {"summary", "listicle", "procedural"}:
        task_type = row.instr_type
    else:
        task_type = "qa"

    lineage = {
        "entailment_ids": [row.entailment_id] if row.entailment_id else [],
        "source_revision_ids": row.source_revision_ids or [],
        "source_chunk_ids": row.source_chunk_ids or [],
        "gap_id": stable_id("gap", row.target_product_family or row.product_family)
        if row.stage == "1.5" else None,
        "data_designer_job_id": "legacy_direct_llm_gapfill" if row.stage == "1.5" else None,
        "seed_sample_ids": [],
    }
    return DatasetSample(
        sample_id=row.sample_id or sample_id_for_row(row),
        origin=origin,
        task_type=task_type,  # type: ignore[arg-type]
        prompt=row.question,
        completion=row.answer,
        system=system_prompt,
        lineage=lineage,
        quality={
            "curator_job_id": None,
            "judge_model": None,
            "grounding_score": None,
        },
        metadata={
            "stage": row.stage,
            "source_url": row.source_url,
            "product_family": row.product_family,
            "refined": row.refined,
            "qa_type": row.qa_type,
            "instr_type": row.instr_type,
        },
    )


def dataset_samples_from_kvp_rows(
    rows: list[KVPRow],
    *,
    system_prompt: str | None = None,
) -> list[DatasetSample]:
    return [dataset_sample_from_kvp_row(row, system_prompt=system_prompt) for row in rows]
