#!/usr/bin/env python3
"""Build deterministic URL-level HTML documents from an Elasticsearch index.

The source index is read-only. All outputs are constrained to the
``curator_dataset`` experiment directory so this utility cannot overwrite a
canonical dataset.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from elasticsearch import Elasticsearch
except ModuleNotFoundError:  # pragma: no cover - reported clearly by main
    Elasticsearch = None  # type: ignore[assignment]


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ES_HOST = os.getenv(
    "PIPELINE_ES_HOST",
    "https://rag-eck-elasticsearch-es-http.runai-rag:9200",
)
DEFAULT_ES_USER = os.getenv("PIPELINE_ES_USER", "elastic")
DEFAULT_PASSWORD_SECRET = "rag-eck-elasticsearch-es-elastic-user"
DEFAULT_PASSWORD_NAMESPACE = "runai-rag"
DEFAULT_PASSWORD_KEY = "elastic"
_BINARY_EXTENSIONS = (".pdf", ".docx", ".pptx", ".doc", ".ppt")
_NON_HTML_SOURCE_SYSTEMS = {
    "document_capture",
    "image_dense_caption",
    "audio_transcript",
    "video_summary",
}
_NON_HTML_SOURCE_KINDS = {"downloaded_asset", "dense_caption", "video_summary_text"}
_NON_HTML_MODALITIES = {"document", "image", "audio", "video"}


@dataclass(frozen=True)
class HtmlChunk:
    es_id: str
    url: str
    text: str
    chunk_index: int
    product_family: str
    product_name: str
    source_revision_id: str | None
    source_chunk_id: str | None
    source_system: str | None
    source_kind: str | None
    modality: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_experiment_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"output must be beneath the experiment directory {EXPERIMENT_ROOT}: {resolved}"
        ) from exc
    return resolved


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_html_source(
    url: str,
    *,
    source_system: str | None,
    source_kind: str | None,
    modality: str | None,
) -> bool:
    source_system_l = (source_system or "").lower()
    source_kind_l = (source_kind or "").lower()
    modality_l = (modality or "").lower()
    if source_system_l in _NON_HTML_SOURCE_SYSTEMS:
        return False
    if source_kind_l in _NON_HTML_SOURCE_KINDS:
        return False
    if modality_l in _NON_HTML_MODALITIES:
        return False
    path_without_query = url.lower().split("?", 1)[0].split("#", 1)[0]
    return not path_without_query.endswith(_BINARY_EXTENSIONS)


def parse_hit(hit: dict[str, Any]) -> tuple[HtmlChunk | None, str | None]:
    source = hit.get("_source") if isinstance(hit.get("_source"), dict) else {}
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    content = (
        metadata.get("content_metadata")
        if isinstance(metadata.get("content_metadata"), dict)
        else {}
    )
    source_meta = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        provenance = content.get("provenance") if isinstance(content.get("provenance"), dict) else {}

    url = _first_string(
        content.get("content_url"),
        content.get("canonical_uri"),
        provenance.get("canonical_uri"),
        source_meta.get("source_name"),
    )
    text = _first_string(source.get("text"))
    if not url:
        return None, "missing_url"
    if not text:
        return None, "missing_text"

    source_system = _first_string(
        content.get("source_system"), source_meta.get("source_system"), provenance.get("source_system")
    )
    source_kind = _first_string(
        content.get("source_kind"), source_meta.get("source_kind"), provenance.get("source_kind")
    )
    modality = _first_string(
        content.get("modality"), source_meta.get("modality"), provenance.get("modality")
    )
    if not is_html_source(
        url,
        source_system=source_system,
        source_kind=source_kind,
        modality=modality,
    ):
        return None, "non_html"

    es_id = str(hit.get("_id") or "").strip()
    if not es_id:
        return None, "missing_es_id"

    source_revision_id = _first_string(
        content.get("source_revision_id"),
        source_meta.get("source_revision_id"),
        provenance.get("source_revision_id"),
    )
    source_chunk_id = _first_string(
        content.get("source_chunk_id"),
        source_meta.get("source_chunk_id"),
        provenance.get("source_chunk_id"),
    )
    return (
        HtmlChunk(
            es_id=es_id,
            url=url,
            text=text,
            chunk_index=_integer(content.get("chunk_index")),
            product_family=_first_string(
                metadata.get("product_family"), content.get("product_family")
            )
            or "unknown",
            product_name=_first_string(metadata.get("product_name"), content.get("product_name"))
            or "unknown",
            source_revision_id=source_revision_id,
            source_chunk_id=source_chunk_id,
            source_system=source_system,
            source_kind=source_kind,
            modality=modality,
        ),
        None,
    )


def stable_unique(values: Iterable[str | None]) -> list[str]:
    return sorted({value for value in values if value})


def build_documents(chunks: Iterable[HtmlChunk], collection: str) -> list[dict[str, Any]]:
    by_url: dict[str, list[HtmlChunk]] = defaultdict(list)
    for chunk in chunks:
        by_url[chunk.url].append(chunk)

    documents: list[dict[str, Any]] = []
    for url in sorted(by_url):
        ordered = sorted(by_url[url], key=lambda chunk: (chunk.chunk_index, chunk.es_id))
        text = "\n\n".join(chunk.text for chunk in ordered).strip()
        document_id = f"url_doc_{sha256_text(collection + chr(10) + url)[:24]}"
        text_hashes = [sha256_text(chunk.text) for chunk in ordered]
        duplicate_text_count = len(text_hashes) - len(set(text_hashes))
        product_families = stable_unique(chunk.product_family for chunk in ordered)
        product_names = stable_unique(chunk.product_name for chunk in ordered)
        documents.append(
            {
                "schema_version": "curator_dataset.url_document.v1",
                "id": document_id,
                "document_id": document_id,
                "collection": collection,
                "url": url,
                "text": text,
                "text_sha256": sha256_text(text),
                "character_count": len(text),
                "word_count": len(text.split()),
                "chunk_count": len(ordered),
                "exact_duplicate_chunk_text_count": duplicate_text_count,
                "es_chunk_ids": [chunk.es_id for chunk in ordered],
                "chunk_indices": [chunk.chunk_index for chunk in ordered],
                "source_chunk_ids": stable_unique(chunk.source_chunk_id for chunk in ordered),
                "source_revision_ids": stable_unique(
                    chunk.source_revision_id for chunk in ordered
                ),
                "source_systems": stable_unique(chunk.source_system for chunk in ordered),
                "source_kinds": stable_unique(chunk.source_kind for chunk in ordered),
                "modalities": stable_unique(chunk.modality for chunk in ordered),
                "product_family": product_families[0] if len(product_families) == 1 else "mixed",
                "product_name": product_names[0] if len(product_names) == 1 else "mixed",
                "product_families": product_families,
                "product_names": product_names,
                "doc_kind": "html",
            }
        )
    return documents


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def read_k8s_secret(name: str, namespace: str, key: str) -> str:
    result = subprocess.run(
        ["kubectl", "get", "secret", name, "-n", namespace, "-o", f"jsonpath={{.data.{key}}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return base64.b64decode(result.stdout.strip()).decode().strip()


def resolve_password(args: argparse.Namespace) -> str:
    value = os.getenv(args.password_env)
    if value:
        return value
    return read_k8s_secret(args.password_secret, args.password_namespace, args.password_key)


def scroll_hits(client: Any, index: str, page_size: int) -> Iterable[dict[str, Any]]:
    page = client.search(
        index=index,
        query={"match_all": {}},
        size=page_size,
        source=["text", "metadata"],
        scroll="5m",
    )
    scroll_id = page.get("_scroll_id")
    try:
        while True:
            hits = page.get("hits", {}).get("hits", [])
            if not hits:
                break
            yield from hits
            page = client.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = page.get("_scroll_id", scroll_id)
    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception:
                pass


def index_uuid(client: Any, index: str) -> str | None:
    try:
        settings = client.indices.get_settings(index=index)
    except Exception:
        return None
    index_settings = settings.get(index, {}).get("settings", {}).get("index", {})
    value = index_settings.get("uuid")
    return str(value) if value else None


def build_manifest(
    *,
    collection: str,
    es_host: str,
    es_index_uuid: str | None,
    output: Path,
    rejection_counts: Counter[str],
    raw_hit_count: int,
    html_chunks: list[HtmlChunk],
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    chunk_counts = [int(document["chunk_count"]) for document in documents]
    word_counts = [int(document["word_count"]) for document in documents]
    duplicate_documents = sum(
        1 for document in documents if document["exact_duplicate_chunk_text_count"] > 0
    )
    return {
        "schema_version": "curator_dataset.url_document_manifest.v1",
        "created_at": utc_now(),
        "collection": collection,
        "source": {
            "type": "elasticsearch",
            "host": es_host,
            "index": collection,
            "index_uuid": es_index_uuid,
            "query": {"match_all": {}},
            "source_fields": ["text", "metadata"],
            "scroll_consistency": "Elasticsearch scroll context",
        },
        "policy": {
            "include": "HTML-like text records only",
            "group_by": "resolved content_url/canonical_uri",
            "order_by": ["chunk_index", "_id"],
            "separator": "two newlines",
            "exact_text_deduplication": False,
            "note": "Exact duplicate chunk text is reported but retained.",
        },
        "counts": {
            "raw_es_hits": raw_hit_count,
            "accepted_html_chunks": len(html_chunks),
            "url_documents": len(documents),
            "rejected_hits": sum(rejection_counts.values()),
            "rejections": dict(sorted(rejection_counts.items())),
            "documents_with_exact_duplicate_chunk_text": duplicate_documents,
            "exact_duplicate_chunk_text_instances": sum(
                int(document["exact_duplicate_chunk_text_count"]) for document in documents
            ),
        },
        "distributions": {
            "chunks_per_document": {
                "min": min(chunk_counts, default=0),
                "max": max(chunk_counts, default=0),
                "mean": round(sum(chunk_counts) / len(chunk_counts), 3) if chunk_counts else 0,
            },
            "words_per_document": {
                "min": min(word_counts, default=0),
                "max": max(word_counts, default=0),
                "mean": round(sum(word_counts) / len(word_counts), 3) if word_counts else 0,
                "over_1000": sum(count > 1000 for count in word_counts),
                "over_4000": sum(count > 4000 for count in word_counts),
            },
        },
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "rows": len(documents),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="nim_curated")
    parser.add_argument("--es-host", default=DEFAULT_ES_HOST)
    parser.add_argument("--es-user", default=DEFAULT_ES_USER)
    parser.add_argument("--password-env", default="PIPELINE_ES_PASSWORD")
    parser.add_argument("--password-secret", default=DEFAULT_PASSWORD_SECRET)
    parser.add_argument("--password-namespace", default=DEFAULT_PASSWORD_NAMESPACE)
    parser.add_argument("--password-key", default=DEFAULT_PASSWORD_KEY)
    parser.add_argument("--verify-certs", action="store_true")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "data" / "nim_curated" / "url_documents.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT_ROOT / "data" / "nim_curated" / "url_documents.manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if Elasticsearch is None:
        raise RuntimeError("install the project Elasticsearch dependency before running")
    output = require_experiment_path(args.output)
    manifest_path = require_experiment_path(args.manifest)
    password = resolve_password(args)
    client = Elasticsearch(
        args.es_host,
        basic_auth=(args.es_user, password),
        verify_certs=args.verify_certs,
        ssl_show_warn=False,
        request_timeout=60,
    )

    rejection_counts: Counter[str] = Counter()
    html_chunks: list[HtmlChunk] = []
    raw_hit_count = 0
    for hit in scroll_hits(client, args.collection, args.page_size):
        raw_hit_count += 1
        chunk, reason = parse_hit(hit)
        if chunk is not None:
            html_chunks.append(chunk)
        elif reason:
            rejection_counts[reason] += 1

    documents = build_documents(html_chunks, args.collection)
    if not documents:
        raise RuntimeError(f"no HTML URL documents produced from index {args.collection!r}")
    write_jsonl_atomic(output, documents)
    manifest = build_manifest(
        collection=args.collection,
        es_host=args.es_host,
        es_index_uuid=index_uuid(client, args.collection),
        output=output,
        rejection_counts=rejection_counts,
        raw_hit_count=raw_hit_count,
        html_chunks=html_chunks,
        documents=documents,
    )
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
