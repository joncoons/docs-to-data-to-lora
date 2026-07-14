"""Stage 0: ES scroll to grouped passages, provenance, and observability."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.es_client import make_es_client, scroll_all_chunks  # noqa: E402
from scripts.pipeline.models import Passage  # noqa: E402
from scripts.pipeline.noise_filter import count_tokens, is_noise  # noqa: E402
from scripts.pipeline.provenance import (  # noqa: E402
    attach_source_provenance,
    build_crawl_run,
    normalize_sha256_hash,
    utc_now,
)
from scripts.pipeline.provenance_io import write_json, write_jsonl  # noqa: E402

Elasticsearch = Any

log = logging.getLogger(__name__)

_BINARY_EXTENSIONS = (".pdf", ".docx", ".pptx", ".doc", ".ppt")
DEFAULT_ES_HOST = os.getenv(
    "PIPELINE_ES_HOST",
    "https://rag-eck-elasticsearch-es-http.runai-rag:9200",
)
DEFAULT_ES_USER = os.getenv("PIPELINE_ES_USER", "elastic")
DEFAULT_ES_PASSWORD = os.getenv("PIPELINE_ES_PASSWORD") or os.getenv("ES_PASSWORD")
DEFAULT_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/outputs"))
DEFAULT_OBSERVABILITY_DIR = Path(
    os.getenv("OBSERVABILITY_DIR", "/outputs/observability/stage0-corpus-prep")
)
_DEFAULT_URL_REGISTRY_RAW = (
    os.getenv("URL_REGISTRY_PATH")
    or os.getenv("PIPELINE_URL_REGISTRY_PATH")
    or os.getenv("PIPELINE_URL_REGISTRY")
)
DEFAULT_URL_REGISTRY_PATH = Path(_DEFAULT_URL_REGISTRY_RAW) if _DEFAULT_URL_REGISTRY_RAW else None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


DEFAULT_MIN_PASSAGE_TOKENS = _env_int("MIN_PASSAGE_TOKENS", 60)


def classify_doc_kind(
    url: str,
    doc_type: str,
    *,
    source_system: str | None = None,
    modality: str | None = None,
    source_kind: str | None = None,
) -> Literal["html", "pdf"]:
    """Return Stage 0 grouping mode while preserving current html/pdf labels.

    Non-web source text, such as OCR, image captions, and video summaries, is
    kept per ES chunk. The Passage model currently has html/pdf labels, so
    `pdf` means per-chunk source text here, not strictly a PDF file.
    """
    source_system_l = (source_system or "").lower()
    modality_l = (modality or "").lower()
    source_kind_l = (source_kind or "").lower()
    if modality_l in {"document", "image", "audio", "video"}:
        return "pdf"
    if source_system_l in {
        "document_capture",
        "image_dense_caption",
        "audio_transcript",
        "video_summary",
    }:
        return "pdf"
    if source_kind_l in {"downloaded_asset", "dense_caption", "video_summary_text"}:
        return "pdf"

    url_lower = url.lower().split("?")[0].split("#")[0]
    if any(url_lower.endswith(ext) for ext in _BINARY_EXTENSIONS):
        return "pdf"
    return "html"


def extract_chunk_dict(hit: dict) -> dict | None:
    """Flatten an ES hit into our internal chunk dict; None if essential fields missing."""
    src = hit.get("_source", {})
    meta = src.get("metadata", {})
    cm = meta.get("content_metadata", {})
    source_meta = meta.get("source", {})
    provenance = meta.get("provenance") or cm.get("provenance") or {}

    url = (
        cm.get("content_url")
        or cm.get("canonical_uri")
        or provenance.get("canonical_uri")
        or source_meta.get("source_name")
        or ""
    )
    text = (src.get("text") or "").strip()
    if not url or not text:
        return None

    source_system = (
        cm.get("source_system")
        or source_meta.get("source_system")
        or provenance.get("source_system")
    )
    source_kind = (
        cm.get("source_kind")
        or source_meta.get("source_kind")
        or provenance.get("source_kind")
    )
    modality = cm.get("modality") or source_meta.get("modality") or provenance.get("modality")

    return {
        "_id": hit["_id"],
        "url": url,
        "chunk_index": cm.get("chunk_index", 0),
        "text": text,
        "vector": src.get("vector"),
        "doc_type": cm.get("document_type", "text"),
        "product_family": meta.get("product_family") or cm.get("product_family") or "unknown",
        "product_name": meta.get("product_name") or cm.get("product_name") or "unknown",
        "content_metadata": cm,
        "source_metadata": source_meta,
        "provenance": provenance,
        "source_system": source_system,
        "source_kind": source_kind,
        "modality": modality,
        "date_created": meta.get("date_created") or source_meta.get("date_created"),
    }


def group_html_chunks_by_url(chunks: list[dict]) -> dict[str, list[dict]]:
    """Concatenate HTML chunks into one passage dict per URL."""
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
    """Group HTML chunks by URL, emit PDF chunks as-is, and apply noise filters."""
    normalized: list[dict] = []
    for c in chunks:
        if "_source" in c:
            d = extract_chunk_dict(c)
            if d is not None:
                normalized.append(d)
        else:
            normalized.append(c)

    for c in normalized:
        c["doc_kind"] = classify_doc_kind(
            c["url"],
            c.get("doc_type", "text"),
            source_system=c.get("source_system"),
            modality=c.get("modality"),
            source_kind=c.get("source_kind"),
        )

    html_chunks = [c for c in normalized if c["doc_kind"] == "html"]
    pdf_chunks = [c for c in normalized if c["doc_kind"] == "pdf"]

    passages: list[Passage] = []
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

    for c in pdf_chunks:
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


def _url_registry_lookup_keys(url: Any) -> list[str]:
    if not isinstance(url, str) or not url.strip():
        return []
    raw = url.strip()
    keys: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in keys:
            keys.append(candidate)

    add(raw)
    try:
        parts = urlsplit(raw)
    except ValueError:
        return keys
    path = parts.path
    normalized = urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))
    add(normalized)
    if path and path != "/" and path.endswith("/"):
        add(urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), parts.query, "")))
    return keys


def load_url_registry(path: Path | None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        return {}, []
    with path.open() as f:
        payload = json.load(f)

    items: list[tuple[str | None, dict[str, Any]]] = []
    if isinstance(payload, dict):
        items = [(str(url), record) for url, record in payload.items() if isinstance(record, dict)]
    elif isinstance(payload, list):
        items = [(record.get("url"), record) for record in payload if isinstance(record, dict)]
    else:
        raise ValueError(f"URL registry must be a dict or list, got {type(payload).__name__}")

    lookup: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for url, record in items:
        entry = dict(record)
        registry_url = entry.get("registry_url") or entry.get("url") or url
        if not registry_url:
            continue
        entry["registry_url"] = registry_url
        records.append(entry)
        candidates = {
            registry_url,
            url,
            entry.get("url"),
            entry.get("canonical_url"),
            entry.get("final_url"),
            entry.get("redirect_to"),
        }
        for candidate in candidates:
            for key in _url_registry_lookup_keys(candidate):
                lookup.setdefault(key, entry)
    return lookup, records


def lookup_url_registry(
    url_registry_lookup: dict[str, dict[str, Any]],
    url: str,
) -> dict[str, Any] | None:
    for key in _url_registry_lookup_keys(url):
        if key in url_registry_lookup:
            return url_registry_lookup[key]
    return None


def build_source_metadata_by_url(
    chunks: list[dict],
    url_registry_lookup: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    metadata_by_url: dict[str, dict[str, Any]] = {}
    for chunk in sorted(chunks, key=lambda c: (c.get("url", ""), c.get("chunk_index") or 0)):
        url = chunk.get("url")
        if not url:
            continue
        entry = metadata_by_url.setdefault(url, {"es_chunks": []})
        content_metadata = chunk.get("content_metadata") or {}
        source_metadata = chunk.get("source_metadata") or {}
        provenance = chunk.get("provenance") or {}
        chunk_meta = {
            "external_chunk_id": chunk.get("_id"),
            "chunk_index": chunk.get("chunk_index"),
            "content_metadata": content_metadata,
            "source": source_metadata,
            "provenance": provenance,
            "date_created": chunk.get("date_created"),
        }
        entry["es_chunks"].append(chunk_meta)
        has_source_metadata = (
            content_metadata
            or source_metadata
            or provenance
            or chunk.get("date_created")
        )
        if "es" not in entry and has_source_metadata:
            entry["es"] = {
                "content_metadata": content_metadata,
                "source": source_metadata,
                "provenance": provenance,
                "date_created": chunk.get("date_created"),
            }
        registry = lookup_url_registry(url_registry_lookup, url)
        if registry:
            entry["url_registry"] = registry
    return {url: entry for url, entry in metadata_by_url.items() if entry}


def _source_revision_id_from_chunk(chunk: dict[str, Any]) -> str | None:
    content_metadata = chunk.get("content_metadata") or {}
    source_metadata = chunk.get("source") or {}
    provenance = chunk.get("provenance") or {}
    candidate = (
        content_metadata.get("source_revision_id")
        or source_metadata.get("source_revision_id")
        or provenance.get("source_revision_id")
    )
    return candidate if isinstance(candidate, str) and candidate.startswith("srcrev_") else None


def _source_chunk_id_from_chunk(chunk: dict[str, Any]) -> str | None:
    content_metadata = chunk.get("content_metadata") or {}
    source_metadata = chunk.get("source") or {}
    provenance = chunk.get("provenance") or {}
    candidate = (
        content_metadata.get("source_chunk_id")
        or source_metadata.get("source_chunk_id")
        or provenance.get("source_chunk_id")
    )
    return candidate if isinstance(candidate, str) and candidate.startswith("chunk_") else None


def build_es_provenance_metrics(chunks: list[dict]) -> dict[str, int]:
    chunk_count = 0
    revision_ids: set[str] = set()
    chunk_ids: set[str] = set()
    for chunk in chunks:
        chunk_meta = {
            "content_metadata": chunk.get("content_metadata") or {},
            "source": chunk.get("source_metadata") or {},
            "provenance": chunk.get("provenance") or {},
        }
        revision_id = _source_revision_id_from_chunk(chunk_meta)
        chunk_id = _source_chunk_id_from_chunk(chunk_meta)
        if chunk_meta["provenance"] or revision_id or chunk_id:
            chunk_count += 1
        if revision_id:
            revision_ids.add(revision_id)
        if chunk_id:
            chunk_ids.add(chunk_id)

    return {
        "chunk_count": chunk_count,
        "source_revision_count": len(revision_ids),
        "source_chunk_count": len(chunk_ids),
    }


def build_url_registry_metrics(
    records: list[dict[str, Any]],
    matched_source_urls: set[str],
    source_urls: set[str],
) -> dict[str, int]:
    if not records:
        unmatched_source_urls = 0
    else:
        unmatched_source_urls = len(source_urls - matched_source_urls)
    return {
        "record_count": len(records),
        "matched_url_count": len(matched_source_urls),
        "unmatched_url_count": unmatched_source_urls,
        "content_hash_count": sum(
            1 for record in records if normalize_sha256_hash(record.get("content_hash"))
        ),
        "etag_count": sum(1 for record in records if record.get("etag")),
        "last_modified_count": sum(1 for record in records if record.get("last_modified")),
        "redirect_count": sum(
            1 for record in records if record.get("redirect_to") or record.get("final_url")
        ),
    }


def safe_metric_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in cleaned.split("_") if part) or "unknown"


def count_jsonl_rows(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def file_manifest(path: Path, artifact_path: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "path": str(path),
        "artifact_path": artifact_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        manifest["rows"] = count_jsonl_rows(path)
    return manifest


def _artifact_path(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return path.name


def build_stage0_observability_documents(
    *,
    index: str,
    es_host: str | None,
    output_dir: Path,
    min_passage_tokens: int,
    raw_hit_count: int,
    extracted_chunk_count: int,
    passages: list[Passage],
    source_revisions: list[Any],
    source_chunks: list[Any],
    artifact_paths: list[Path],
    crawl_run_id: str,
    url_registry_path: Path | None = None,
    url_registry_metrics: dict[str, int] | None = None,
    es_provenance_metrics: dict[str, int] | None = None,
    pipeline_run_id: str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_parent_run_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build MLflow-ready observability files without requiring the MLflow client."""
    token_counts = [p.token_count for p in passages]
    doc_kind_counts = Counter(p.doc_kind for p in passages)
    product_family_counts = Counter(p.product_family or "unknown" for p in passages)
    registry_metrics = url_registry_metrics or {}
    es_metrics = es_provenance_metrics or {}
    metrics: dict[str, int | float] = {
        "stage0.raw_hits.count": raw_hit_count,
        "stage0.chunks.extracted": extracted_chunk_count,
        "stage0.passages.count": len(passages),
        "stage0.source_revisions.count": len(source_revisions),
        "stage0.source_chunks.count": len(source_chunks),
        "stage0.urls.count": len({p.url for p in passages}),
        "stage0.tokens.total": sum(token_counts),
        "stage0.tokens.mean": (
            round(sum(token_counts) / len(token_counts), 2) if token_counts else 0
        ),
        "stage0.doc_kind.html": doc_kind_counts.get("html", 0),
        "stage0.doc_kind.pdf": doc_kind_counts.get("pdf", 0),
        "stage0.es_provenance.chunks.count": es_metrics.get("chunk_count", 0),
        "stage0.es_provenance.source_revisions.count": es_metrics.get("source_revision_count", 0),
        "stage0.es_provenance.source_chunks.count": es_metrics.get("source_chunk_count", 0),
        "stage0.url_registry.records.count": registry_metrics.get("record_count", 0),
        "stage0.url_registry.matched_urls.count": registry_metrics.get("matched_url_count", 0),
        "stage0.url_registry.unmatched_urls.count": registry_metrics.get("unmatched_url_count", 0),
        "stage0.url_registry.content_hash.count": registry_metrics.get("content_hash_count", 0),
        "stage0.url_registry.etag.count": registry_metrics.get("etag_count", 0),
        "stage0.url_registry.last_modified.count": registry_metrics.get("last_modified_count", 0),
        "stage0.url_registry.redirects.count": registry_metrics.get("redirect_count", 0),
    }
    for product_family, count in sorted(product_family_counts.items()):
        metrics[f"stage0.product_family.{safe_metric_name(product_family)}.passages"] = count

    artifacts = [
        file_manifest(path, artifact_path=_artifact_path(path, output_dir))
        for path in artifact_paths
    ]

    run_context: dict[str, Any] = {
        "schema_version": "observability.v1",
        "pipeline_stage": "stage0-corpus-prep",
        "created_at": utc_now(),
        "pipeline_run_id": pipeline_run_id,
        "collection": index,
        "input_index": index,
        "output_dir": str(output_dir),
        "parameters": {
            "min_passage_tokens": min_passage_tokens,
        },
        "inputs": {
            "url_registry_path": str(url_registry_path) if url_registry_path else None,
        },
        "mlflow": {
            "tracking_uri": mlflow_tracking_uri,
            "experiment_name": mlflow_experiment_name,
            "parent_run_id": mlflow_parent_run_id,
        },
    }
    service_refs: dict[str, Any] = {
        "schema_version": "observability.v1",
        "services": {
            "elasticsearch": {
                "host": es_host,
                "index": index,
            },
            "url_registry": {
                "path": str(url_registry_path) if url_registry_path else None,
            },
        },
        "provenance": {
            "crawl_run_id": crawl_run_id,
            "source_revisions_uri": "provenance/source_revisions.jsonl",
            "source_chunks_uri": "provenance/source_chunks.jsonl",
        },
    }
    artifacts_manifest: dict[str, Any] = {
        "schema_version": "observability.v1",
        "artifacts": artifacts,
    }
    return {
        "run_context.json": run_context,
        "metrics.json": metrics,
        "artifacts_manifest.json": artifacts_manifest,
        "service_refs.json": service_refs,
    }


def write_observability_documents(out_dir: Path, documents: dict[str, dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in documents.items():
        (out_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_stage0(
    es: Elasticsearch,
    index: str,
    output_dir: Path,
    min_passage_tokens: int = 60,
    *,
    es_host: str | None = None,
    observability_dir: Path | None = None,
    url_registry_path: Path | None = None,
    pipeline_run_id: str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_parent_run_id: str | None = None,
) -> tuple[list[Passage], dict[str, list[float]]]:
    """Scroll the index, build passages, write JSONL, and optionally emit observability.

    Returns (passages, seed_vectors_by_passage_id).
    Seed vectors are kept in memory only because they are large and only needed at runtime.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "passages.jsonl"
    crawl_run_file = output_dir / "manifests" / "crawl_run.json"
    source_revisions_file = output_dir / "provenance" / "source_revisions.jsonl"
    source_chunks_file = output_dir / "provenance" / "source_chunks.jsonl"

    started_at = utc_now()
    log.info("Stage 0: scrolling ES index '%s'...", index)
    raw_hits = list(tqdm(scroll_all_chunks(es, index=index), desc="ES scroll"))
    log.info("Stage 0: %d hits retrieved", len(raw_hits))

    chunks: list[dict] = []
    for h in raw_hits:
        d = extract_chunk_dict(h)
        if d is not None:
            chunks.append(d)
    log.info("Stage 0: %d chunks after extraction", len(chunks))

    url_registry_lookup, url_registry_records = load_url_registry(url_registry_path)
    if url_registry_path is not None:
        log.info(
            "Stage 0: loaded %d URL registry records from %s",
            len(url_registry_records),
            url_registry_path,
        )
    source_metadata_by_url = build_source_metadata_by_url(chunks, url_registry_lookup)
    es_provenance_metrics = build_es_provenance_metrics(chunks)

    passages = build_passages(chunks, min_passage_tokens=min_passage_tokens)
    log.info("Stage 0: %d passages after grouping + noise filter", len(passages))

    source_urls = {p.url for p in passages}
    matched_source_urls = {
        url
        for url in source_urls
        if (source_metadata_by_url.get(url) or {}).get("url_registry")
    }
    url_registry_metrics = build_url_registry_metrics(
        url_registry_records,
        matched_source_urls,
        source_urls,
    )

    completed_at = utc_now()
    crawl_run = build_crawl_run(
        index=index,
        started_at=started_at,
        completed_at=completed_at,
        chunk_count=len(chunks),
        passage_count=len(passages),
        min_passage_tokens=min_passage_tokens,
        url_registry_path=str(url_registry_path) if url_registry_path else None,
        url_registry_record_count=url_registry_metrics["record_count"],
        url_registry_matched_url_count=url_registry_metrics["matched_url_count"],
    )
    passages, source_revisions, source_chunks = attach_source_provenance(
        passages,
        crawl_run_id=crawl_run.crawl_run_id,
        retrieved_at=started_at,
        chunker_config={
            "name": "stage0-url-grouping",
            "version": "v1",
            "min_passage_tokens": min_passage_tokens,
            "extraction_method": "es-scroll-url-grouping",
        },
        source_metadata_by_url=source_metadata_by_url,
    )

    seed_vectors: dict[str, list[float]] = {}
    for p in passages:
        for c in chunks:
            if c["_id"] in p.chunk_ids and c.get("vector"):
                seed_vectors[p.passage_id] = c["vector"]
                break

    with out_file.open("w") as f:
        for p in passages:
            f.write(p.model_dump_json() + "\n")
    log.info("Stage 0: written %s", out_file)

    write_json(crawl_run_file, crawl_run)
    write_jsonl(source_revisions_file, source_revisions)
    write_jsonl(source_chunks_file, source_chunks)
    log.info("Stage 0: written provenance sidecars under %s", output_dir / "provenance")

    artifact_paths = [out_file, crawl_run_file, source_revisions_file, source_chunks_file]
    if url_registry_path is not None and url_registry_path.exists():
        artifact_paths.append(url_registry_path)

    if observability_dir is not None:
        docs = build_stage0_observability_documents(
            index=index,
            es_host=es_host,
            output_dir=output_dir,
            min_passage_tokens=min_passage_tokens,
            raw_hit_count=len(raw_hits),
            extracted_chunk_count=len(chunks),
            passages=passages,
            source_revisions=source_revisions,
            source_chunks=source_chunks,
            artifact_paths=artifact_paths,
            crawl_run_id=crawl_run.crawl_run_id,
            url_registry_path=url_registry_path,
            url_registry_metrics=url_registry_metrics,
            es_provenance_metrics=es_provenance_metrics,
            pipeline_run_id=pipeline_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_parent_run_id=mlflow_parent_run_id,
        )
        write_observability_documents(observability_dir, docs)
        log.info("Stage 0: written observability files under %s", observability_dir)

    return passages, seed_vectors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Stage 0 passages and provenance from ES.")
    ap.add_argument(
        "--index",
        "--collection",
        dest="index",
        default=os.getenv("PIPELINE_ES_INDEX") or os.getenv("PIPELINE_ES_COLLECTION"),
        help="Elasticsearch index/collection to scroll.",
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--min-passage-tokens", type=int, default=DEFAULT_MIN_PASSAGE_TOKENS)
    ap.add_argument("--es-host", default=DEFAULT_ES_HOST)
    ap.add_argument("--es-user", default=DEFAULT_ES_USER)
    ap.add_argument("--es-password", default=DEFAULT_ES_PASSWORD)
    ap.add_argument("--observability-dir", type=Path, default=DEFAULT_OBSERVABILITY_DIR)
    ap.add_argument("--url-registry", type=Path, default=DEFAULT_URL_REGISTRY_PATH)
    ap.add_argument("--pipeline-run-id", default=os.getenv("PIPELINE_RUN_ID"))
    ap.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    ap.add_argument("--mlflow-experiment-name", default=os.getenv("MLFLOW_EXPERIMENT_NAME"))
    ap.add_argument("--mlflow-parent-run-id", default=os.getenv("MLFLOW_PARENT_RUN_ID"))
    args = ap.parse_args(argv)
    if not args.index:
        ap.error("--index is required unless PIPELINE_ES_INDEX is set")
    if args.es_password is None:
        ap.error("--es-password is required unless PIPELINE_ES_PASSWORD or ES_PASSWORD is set")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args(argv)
    es = make_es_client(args.es_host, args.es_password, user=args.es_user)
    passages, _ = run_stage0(
        es,
        args.index,
        args.output_dir,
        min_passage_tokens=args.min_passage_tokens,
        es_host=args.es_host,
        observability_dir=args.observability_dir,
        url_registry_path=args.url_registry,
        pipeline_run_id=args.pipeline_run_id,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_parent_run_id=args.mlflow_parent_run_id,
    )
    log.info("Stage 0 complete: %d passages", len(passages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
