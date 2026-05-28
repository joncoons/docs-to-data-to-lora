"""Source-agnostic provenance helpers for unstructured ingestion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA_VERSION = "unstructured-source-provenance.v1"
REGISTRY_SCHEMA_VERSION = "url-registry.v2"


_HASH_PREFIX = "sha256:"


def compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    """Drop only empty values that do not carry useful provenance."""
    return {k: v for k, v in value.items() if v is not None and v != ""}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalize_sha256_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    digest = raw[len(_HASH_PREFIX):] if raw.startswith(_HASH_PREFIX) else raw
    if len(digest) != 64:
        return None
    if not all(ch in "0123456789abcdefABCDEF" for ch in digest):
        return None
    return f"{_HASH_PREFIX}{digest.lower()}"


def sha256_bytes(data: bytes) -> str:
    return f"{_HASH_PREFIX}{hashlib.sha256(data).hexdigest()}"


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hashes_equal(left: Any, right: Any) -> bool:
    left_hash = normalize_sha256_hash(left)
    right_hash = normalize_sha256_hash(right)
    if left_hash and right_hash:
        return left_hash == right_hash
    if isinstance(left, str) and isinstance(right, str):
        return bool(left.strip() and left.strip() == right.strip())
    return False


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def source_revision_id_for(
    canonical_uri: str,
    raw_sha256: str | None,
    observed_at: str | None = None,
) -> str:
    return stable_id("srcrev", canonical_uri, raw_sha256 or "", observed_at or "")


def source_chunk_id_for(source_revision_id: str, chunk_index: int, text_sha256: str) -> str:
    return stable_id("chunk", source_revision_id, chunk_index, text_sha256)


def infer_modality(source_uri: str, metadata: dict[str, Any]) -> str:
    explicit = metadata.get("modality") or metadata.get("media_type")
    if explicit:
        return str(explicit)
    suffix = Path(source_uri.split("?", 1)[0].split("#", 1)[0]).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"}:
        return "image"
    if suffix in {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m4v"}:
        return "video"
    if suffix in {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus"}:
        return "audio"
    if suffix in {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}:
        return "document"
    return "text"


def build_es_provenance(
    *,
    text: str,
    source_uri: str,
    chunk_index: int,
    metadata: dict[str, Any],
    date_created: str,
) -> dict[str, Any]:
    """Build common provenance fields for a text-bearing ES chunk."""
    raw_sha256 = normalize_sha256_hash(
        metadata.get("source_content_hash")
        or metadata.get("raw_sha256")
        or metadata.get("content_hash")
    )
    text_hash = sha256_text(text)
    observed_at = (
        metadata.get("retrieved_at")
        or metadata.get("captured_at")
        or metadata.get("processed_at")
        or metadata.get("downloaded_at")
        or date_created
    )
    final_uri = metadata.get("final_uri") or metadata.get("final_url") or source_uri
    source_system = metadata.get("source_system") or "unknown"
    source_kind = metadata.get("source_kind") or metadata.get("document_type") or "text"
    modality = infer_modality(source_uri, metadata)
    source_revision_id = metadata.get("source_revision_id") or source_revision_id_for(
        source_uri, raw_sha256, observed_at
    )
    source_chunk_id = metadata.get("source_chunk_id") or source_chunk_id_for(
        source_revision_id, chunk_index, text_hash
    )
    ingestion_run_id = metadata.get("ingestion_run_id") or metadata.get("crawl_run_id")

    http = compact_dict({
        "status_code": metadata.get("http_status_code") or metadata.get("status_code"),
        "etag": metadata.get("http_etag") or metadata.get("etag"),
        "last_modified": metadata.get("http_last_modified") or metadata.get("last_modified"),
        "content_type": metadata.get("content_type"),
    })
    anchors = compact_dict({
        "heading_path": metadata.get("section_path"),
        "heading": metadata.get("heading"),
        "section_h1": metadata.get("section_h1"),
        "page_number": metadata.get("page_number"),
        "bbox": metadata.get("bbox"),
        "time_start_seconds": metadata.get("time_start_seconds"),
        "time_end_seconds": metadata.get("time_end_seconds"),
        "frame_start": metadata.get("frame_start"),
        "frame_end": metadata.get("frame_end"),
    })
    transforms = metadata.get("transforms") or []

    content_metadata = compact_dict({
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "ingestion_run_id": ingestion_run_id,
        "crawl_run_id": metadata.get("crawl_run_id") or ingestion_run_id,
        "source_revision_id": source_revision_id,
        "source_chunk_id": source_chunk_id,
        "source_system": source_system,
        "source_kind": source_kind,
        "modality": modality,
        "canonical_uri": source_uri,
        "final_uri": final_uri,
        "retrieved_at": observed_at,
        "raw_sha256": raw_sha256,
        "source_content_hash": raw_sha256,
        "text_sha256": text_hash,
        "parser_version": metadata.get("parser_version"),
        "chunker_version": metadata.get("chunker_version"),
        "http_status_code": http.get("status_code"),
        "http_etag": http.get("etag"),
        "http_last_modified": http.get("last_modified"),
        "content_type": http.get("content_type"),
    })

    provenance = compact_dict({
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "ingestion_run_id": ingestion_run_id,
        "source_revision_id": source_revision_id,
        "source_chunk_id": source_chunk_id,
        "source_system": source_system,
        "source_kind": source_kind,
        "modality": modality,
        "canonical_uri": source_uri,
        "final_uri": final_uri,
        "observed_at": observed_at,
        "hashes": compact_dict({
            "raw_sha256": raw_sha256,
            "text_sha256": text_hash,
        }),
        "http": http,
        "anchors": anchors,
        "transforms": transforms,
    })

    return {
        "source": compact_dict({
            "source_revision_id": source_revision_id,
            "source_chunk_id": source_chunk_id,
            "ingestion_run_id": ingestion_run_id,
            "source_system": source_system,
            "source_kind": source_kind,
            "modality": modality,
        }),
        "content_metadata": content_metadata,
        "provenance": provenance,
    }


def registry_provenance_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return registry-safe fields from source metadata without chunk text details."""
    return compact_dict({
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "ingestion_run_id": metadata.get("ingestion_run_id") or metadata.get("crawl_run_id"),
        "crawl_run_id": metadata.get("crawl_run_id") or metadata.get("ingestion_run_id"),
        "source_revision_id": metadata.get("source_revision_id"),
        "source_system": metadata.get("source_system"),
        "source_kind": metadata.get("source_kind"),
        "modality": metadata.get("modality"),
        "final_uri": metadata.get("final_uri") or metadata.get("final_url"),
        "raw_sha256": normalize_sha256_hash(
            metadata.get("raw_sha256")
            or metadata.get("source_content_hash")
            or metadata.get("content_hash")
        ),
        "source_content_hash": normalize_sha256_hash(
            metadata.get("source_content_hash")
            or metadata.get("raw_sha256")
            or metadata.get("content_hash")
        ),
        "retrieved_at": metadata.get("retrieved_at"),
        "content_type": metadata.get("content_type"),
        "parser_version": metadata.get("parser_version"),
        "chunker_version": metadata.get("chunker_version"),
    })


def normalize_registry_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy registry hash fields in place and return the entry."""
    for key in ("content_hash", "last_ingested_hash", "raw_sha256", "source_content_hash"):
        normalized = normalize_sha256_hash(entry.get(key))
        if normalized:
            entry[key] = normalized
    return entry
