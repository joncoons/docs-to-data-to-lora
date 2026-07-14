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


def normalize_sha256_hash(value: Any) -> str | None:
    """Normalize crawler hash variants to the schema's sha256:<hex> form."""
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if raw.startswith("sha256:"):
        digest = raw[len("sha256:"):]
    else:
        digest = raw
    if len(digest) != 64:
        return None
    if not all(ch in "0123456789abcdefABCDEF" for ch in digest):
        return None
    return f"sha256:{digest.lower()}"


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


def passage_source_systems(passage: Passage) -> list[str]:
    return list(passage.source_systems or [])


def passage_source_kinds(passage: Passage) -> list[str]:
    return list(passage.source_kinds or [])


def passage_modalities(passage: Passage) -> list[str]:
    return list(passage.modalities or [])


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
    recrawl_reason: Literal[
        "initial", "scheduled", "manual", "hotfix", "backfill", "test"
    ] = "initial"
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
    status: Literal[
        "active", "unchanged", "changed", "deleted", "redirected", "failed", "excluded"
    ] = "active"
    http: dict[str, Any] = Field(default_factory=dict)
    hashes: dict[str, str | None]
    content: dict[str, Any] = Field(default_factory=dict)
    classification: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    origin: Literal[
        "source_entailed", "synthetic_gapfill", "human_reviewed", "imported_baseline"
    ]
    task_type: Literal[
        "qa", "summary", "listicle", "procedural", "bridging",
        "contrastive", "rag_eval", "other"
    ]
    prompt: str
    completion: str
    system: str | None = None
    lineage: dict[str, Any]
    quality: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if v is not None}


def _registry_link_count(registry: dict[str, Any]) -> int | None:
    linked_hrefs = registry.get("linked_hrefs")
    if isinstance(linked_hrefs, list):
        return len(linked_hrefs)
    return None


def _registry_metadata(registry: dict[str, Any]) -> dict[str, Any]:
    if not registry:
        return {}
    return _compact_dict({
        "registry_url": registry.get("registry_url"),
        "collection": registry.get("collection"),
        "last_seen": registry.get("last_seen"),
        "last_ingested": registry.get("last_ingested"),
        "content_hash": registry.get("content_hash"),
        "content_hash_kind": (
            "crawler_source_content_hash" if registry.get("content_hash") else None
        ),
        "linked_href_count": _registry_link_count(registry),
        "redirect_to": registry.get("redirect_to") or registry.get("final_url"),
    })


def _source_revision_metadata(source_meta: dict[str, Any]) -> dict[str, Any]:
    if not source_meta:
        return {}
    metadata: dict[str, Any] = {}
    registry = _registry_metadata(source_meta.get("url_registry") or {})
    if registry:
        metadata["url_registry"] = registry
    es_meta = source_meta.get("es") or {}
    if es_meta:
        metadata["elasticsearch"] = es_meta
    es_chunks = source_meta.get("es_chunks") or []
    upstream_revision_ids = sorted({
        revision_id
        for chunk in es_chunks
        for revision_id in [_upstream_source_revision_id(chunk)]
        if revision_id
    })
    upstream_chunk_ids = sorted({
        chunk_id
        for chunk in es_chunks
        for chunk_id in [_upstream_source_chunk_id(chunk)]
        if chunk_id
    })
    if upstream_revision_ids or upstream_chunk_ids:
        metadata["upstream"] = _compact_dict({
            "source_revision_ids": upstream_revision_ids,
            "source_chunk_ids": upstream_chunk_ids,
            "source_provenance_kind": "es_metadata_provenance",
        })
    return metadata


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _upstream_source_revision_id(chunk: dict[str, Any]) -> str | None:
    content_metadata = _as_dict(chunk.get("content_metadata"))
    source_metadata = _as_dict(chunk.get("source"))
    provenance = _as_dict(chunk.get("provenance"))
    candidate = _first_nonempty(
        content_metadata.get("source_revision_id"),
        source_metadata.get("source_revision_id"),
        provenance.get("source_revision_id"),
    )
    return candidate if isinstance(candidate, str) and candidate.startswith("srcrev_") else None


def _upstream_source_chunk_id(chunk: dict[str, Any]) -> str | None:
    content_metadata = _as_dict(chunk.get("content_metadata"))
    source_metadata = _as_dict(chunk.get("source"))
    provenance = _as_dict(chunk.get("provenance"))
    candidate = _first_nonempty(
        content_metadata.get("source_chunk_id"),
        source_metadata.get("source_chunk_id"),
        provenance.get("source_chunk_id"),
    )
    return candidate if isinstance(candidate, str) and candidate.startswith("chunk_") else None


def _source_dimension_from_chunk(chunk: dict[str, Any], key: str) -> str | None:
    content_metadata = _as_dict(chunk.get("content_metadata"))
    source_metadata = _as_dict(chunk.get("source"))
    provenance = _as_dict(chunk.get("provenance"))
    value = _first_nonempty(
        content_metadata.get(key),
        source_metadata.get(key),
        provenance.get(key),
    )
    return str(value) if value is not None and value != "" else None


def _source_dimension_values(chunks: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({
        value
        for chunk in chunks
        for value in [_source_dimension_from_chunk(chunk, key)]
        if value
    })


def _single_upstream_revision_id(source_meta: dict[str, Any]) -> str | None:
    revision_ids = {
        revision_id
        for chunk in source_meta.get("es_chunks", [])
        for revision_id in [_upstream_source_revision_id(chunk)]
        if revision_id
    }
    if len(revision_ids) == 1:
        return next(iter(revision_ids))
    return None


def _chunks_for_passage(source_meta: dict[str, Any], passage: Passage) -> list[dict[str, Any]]:
    chunks_by_external_id = {
        chunk.get("external_chunk_id"): chunk
        for chunk in source_meta.get("es_chunks", [])
        if chunk.get("external_chunk_id")
    }
    return [chunks_by_external_id[cid] for cid in passage.chunk_ids if cid in chunks_by_external_id]


def _single_upstream_chunk_id(chunks: list[dict[str, Any]], passage: Passage) -> str | None:
    if len(chunks) != 1 or len(passage.chunk_ids) != 1:
        return None
    return _upstream_source_chunk_id(chunks[0])


def _provenance_http(
    registry: dict[str, Any],
    content_metadata: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    http = _as_dict(provenance.get("http"))
    return {
        "status_code": _first_nonempty(
            registry.get("status_code"),
            http.get("status_code"),
            content_metadata.get("http_status_code"),
        ),
        "etag": _first_nonempty(
            registry.get("etag"),
            http.get("etag"),
            content_metadata.get("http_etag"),
        ),
        "last_modified": _first_nonempty(
            registry.get("last_modified"),
            http.get("last_modified"),
            content_metadata.get("http_last_modified"),
        ),
        "content_type": _first_nonempty(
            registry.get("content_type"),
            http.get("content_type"),
            content_metadata.get("content_type"),
        ),
        "cache_control": registry.get("cache_control"),
    }


def _raw_sha256(
    registry: dict[str, Any],
    content_metadata: dict[str, Any],
    provenance: dict[str, Any],
) -> str | None:
    hashes = _as_dict(provenance.get("hashes"))
    return normalize_sha256_hash(_first_nonempty(
        registry.get("content_hash"),
        hashes.get("raw_sha256"),
        content_metadata.get("raw_sha256"),
        content_metadata.get("source_content_hash"),
    ))


def _heading_path(content_metadata: dict[str, Any]) -> list[str]:
    section_path = content_metadata.get("section_path")
    if isinstance(section_path, list):
        return [str(part).strip() for part in section_path if str(part).strip()]
    if isinstance(section_path, str) and section_path.strip():
        if ">" in section_path:
            return [part.strip() for part in section_path.split(">") if part.strip()]
        return [section_path.strip()]
    heading = content_metadata.get("heading") or content_metadata.get("section_h1")
    if isinstance(heading, str) and heading.strip():
        return [heading.strip()]
    return []


def build_crawl_run(
    *,
    index: str,
    started_at: str,
    completed_at: str,
    chunk_count: int,
    passage_count: int,
    min_passage_tokens: int,
    url_registry_path: str | None = None,
    url_registry_record_count: int = 0,
    url_registry_matched_url_count: int = 0,
) -> CrawlRun:
    config = {
        "index": index,
        "min_passage_tokens": min_passage_tokens,
        "chunker": "stage0-url-grouping",
        "url_registry_path": url_registry_path,
    }
    crawl_run_id = stable_id("crawlrun", index, started_at)
    scope: dict[str, Any] = {
        "seed_url_set_hash": sha256_json({"index": index}),
        "allowed_url_prefixes": [],
        "excluded_url_patterns": [],
        "binary_host_allowlist": [],
    }
    if url_registry_path:
        scope["url_registry_uri"] = url_registry_path
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
        scope=scope,
        outputs={
            "source_revisions_uri": "provenance/source_revisions.jsonl",
            "source_chunks_uri": "provenance/source_chunks.jsonl",
            "delta_manifest_uri": None,
        },
        metrics={
            "raw_chunk_count": chunk_count,
            "passage_count": passage_count,
            "url_registry_record_count": url_registry_record_count,
            "url_registry_matched_url_count": url_registry_matched_url_count,
        },
    )


def attach_source_provenance(
    passages: list[Passage],
    *,
    crawl_run_id: str,
    retrieved_at: str,
    chunker_config: dict[str, Any] | None = None,
    source_metadata_by_url: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[Passage], list[SourceRevision], list[SourceChunk]]:
    """Return passage copies with source IDs plus source revision/chunk ledgers."""
    chunker_config = chunker_config or {"name": "stage0-url-grouping", "version": "v1"}
    source_metadata_by_url = source_metadata_by_url or {}
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
        source_meta = source_metadata_by_url.get(url, {})
        registry = source_meta.get("url_registry") or {}
        es_meta = source_meta.get("es") or {}
        content_metadata = _as_dict(es_meta.get("content_metadata"))
        source_metadata = _as_dict(es_meta.get("source"))
        upstream_provenance = _as_dict(es_meta.get("provenance"))
        revision_id = _single_upstream_revision_id(source_meta) or source_revision_id_for_text(
            url, normalized_text
        )
        revision_id_by_url[url] = revision_id
        first = first_by_url[url]
        final_url = _first_nonempty(
            registry.get("redirect_to"),
            registry.get("final_url"),
            content_metadata.get("final_uri"),
            upstream_provenance.get("final_uri"),
            url,
        )
        is_redirect = isinstance(final_url, str) and final_url.rstrip("/") != url.rstrip("/")
        retrieved_value = _first_nonempty(
            registry.get("last_ingested"),
            registry.get("last_seen"),
            content_metadata.get("retrieved_at"),
            content_metadata.get("captured_at"),
            content_metadata.get("processed_at"),
            upstream_provenance.get("observed_at"),
            retrieved_at,
        )
        revisions.append(SourceRevision(
            source_revision_id=revision_id,
            crawl_run_id=crawl_run_id,
            canonical_url=str(_first_nonempty(
                content_metadata.get("canonical_uri"),
                upstream_provenance.get("canonical_uri"),
                url,
            )),
            final_url=str(final_url) if final_url else None,
            retrieved_at=str(retrieved_value),
            status="redirected" if is_redirect else "active",
            http=_provenance_http(registry, content_metadata, upstream_provenance),
            hashes={
                "raw_sha256": _raw_sha256(registry, content_metadata, upstream_provenance),
                "normalized_sha256": sha256_text(normalized_text),
            },
            content={
                "raw_uri": content_metadata.get("raw_uri"),
                "normalized_uri": content_metadata.get("normalized_uri"),
                "language": content_metadata.get("language"),
                "title": content_metadata.get("page_title") or content_metadata.get("heading"),
            },
            classification={
                "domain_area": first.product_family,
                "domain_slice": first.product_name,
                "version": content_metadata.get("version"),
                "doc_kind": first.doc_kind,
                "document_type": content_metadata.get("document_type"),
                "section_h1": content_metadata.get("section_h1"),
                "section_path": content_metadata.get("section_path"),
                "heading": content_metadata.get("heading"),
                "source_system": _first_nonempty(
                    content_metadata.get("source_system"),
                    source_metadata.get("source_system"),
                    upstream_provenance.get("source_system"),
                ),
                "source_kind": _first_nonempty(
                    content_metadata.get("source_kind"),
                    source_metadata.get("source_kind"),
                    upstream_provenance.get("source_kind"),
                ),
                "modality": _first_nonempty(
                    content_metadata.get("modality"),
                    source_metadata.get("modality"),
                    upstream_provenance.get("modality"),
                ),
                "parser_version": content_metadata.get("parser_version"),
                "chunker_version": content_metadata.get("chunker_version"),
                "source_id": source_metadata.get("source_id"),
                "source_type": source_metadata.get("source_type"),
            },
            metadata=_source_revision_metadata(source_meta),
        ))

    updated: list[Passage] = []
    source_chunks: list[SourceChunk] = []
    for passage in passages:
        revision_id = revision_id_by_url[passage.url]
        source_meta = source_metadata_by_url.get(passage.url, {})
        registry = source_meta.get("url_registry") or {}
        matching_chunks = _chunks_for_passage(source_meta, passage)
        first_chunk = matching_chunks[0] if matching_chunks else {}
        content_metadata = _as_dict(first_chunk.get("content_metadata"))
        source_metadata = _as_dict(first_chunk.get("source"))
        upstream_provenance = _as_dict(first_chunk.get("provenance"))
        chunk_id = _single_upstream_chunk_id(matching_chunks, passage) or source_chunk_id_for_text(
            revision_id, passage.passage_id, passage.text
        )
        upstream_revision_ids = sorted({
            revision_id_value
            for chunk in matching_chunks
            for revision_id_value in [_upstream_source_revision_id(chunk)]
            if revision_id_value
        })
        upstream_chunk_ids = sorted({
            chunk_id_value
            for chunk in matching_chunks
            for chunk_id_value in [_upstream_source_chunk_id(chunk)]
            if chunk_id_value
        })
        upstream_provenance_records = [
            chunk.get("provenance")
            for chunk in matching_chunks
            if chunk.get("provenance")
        ]
        source_systems = _source_dimension_values(matching_chunks, "source_system")
        source_kinds = _source_dimension_values(matching_chunks, "source_kind")
        modalities = _source_dimension_values(matching_chunks, "modality")
        updated.append(passage.model_copy(update={
            "source_revision_id": revision_id,
            "source_chunk_ids": [chunk_id],
            "source_systems": source_systems or None,
            "source_kinds": source_kinds or None,
            "modalities": modalities or None,
        }))
        source_chunks.append(SourceChunk(
            chunk_id=chunk_id,
            source_revision_id=revision_id,
            chunker=chunker,
            anchors={
                "heading_path": _heading_path(content_metadata),
                "dom_path": content_metadata.get("dom_path"),
                "markdown_anchor": content_metadata.get("markdown_anchor"),
                "page_number": content_metadata.get("page_number"),
                "parser_element_id": content_metadata.get("parser_element_id"),
            },
            spans={
                "byte_start": content_metadata.get("byte_start"),
                "byte_end": content_metadata.get("byte_end"),
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
                "domain_area": passage.product_family,
                "domain_slice": passage.product_name,
                "doc_kind": passage.doc_kind,
                "content_metadata": content_metadata,
                "source": source_metadata,
                "url_registry": _registry_metadata(registry),
                "upstream_source_revision_ids": upstream_revision_ids,
                "upstream_source_chunk_ids": upstream_chunk_ids,
                "upstream_provenance": upstream_provenance_records,
                "source_systems": source_systems,
                "source_kinds": source_kinds,
                "modalities": modalities,
                "source_system": _first_nonempty(
                    content_metadata.get("source_system"),
                    source_metadata.get("source_system"),
                    upstream_provenance.get("source_system"),
                ),
                "source_kind": _first_nonempty(
                    content_metadata.get("source_kind"),
                    source_metadata.get("source_kind"),
                    upstream_provenance.get("source_kind"),
                ),
                "modality": _first_nonempty(
                    content_metadata.get("modality"),
                    source_metadata.get("modality"),
                    upstream_provenance.get("modality"),
                ),
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
                "domain_slice": row.product_family,
                "source_systems": row.source_systems or [],
                "source_kinds": row.source_kinds or [],
                "modalities": row.modalities or [],
            },
        ))
    return list(seen.values())


def _bool_score(value: bool | None) -> float | None:
    if value is None:
        return None
    return 1.0 if value else 0.0


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
        "source_systems": row.source_systems or [],
        "source_kinds": row.source_kinds or [],
        "modalities": row.modalities or [],
        "gap_id": stable_id("gap", row.target_product_family or row.product_family)
        if row.stage == "1.5" else None,
        "data_designer_job_id": "nemo_data_designer_gapfill" if row.stage == "1.5" else None,
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
            "judge_model": row.qa_judge_model,
            "judge_endpoints": row.qa_judge_endpoints or [],
            "grounding_score": _bool_score(row.qa_grounded),
            "qa_work_id": row.qa_work_id,
            "qa_status": row.qa_status,
            "qa_admitted": row.qa_admitted,
            "qa_execution_surface": row.qa_execution_surface,
            "grounded": row.qa_grounded,
            "answer_fidelity": row.qa_answer_fidelity,
            "no_hallucination": row.qa_no_hallucination,
            "repairable": row.qa_repairable,
            "reason": row.qa_reason,
        },
        metadata={
            "stage": row.stage,
            "source_url": row.source_url,
            "domain_slice": row.product_family,
            "source_systems": row.source_systems or [],
            "source_kinds": row.source_kinds or [],
            "modalities": row.modalities or [],
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
