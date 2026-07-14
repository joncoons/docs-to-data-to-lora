#!/usr/bin/env python3
"""Build acknowledgement-free, exact-deduplicated HTML URL documents."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_url_documents import (
    DEFAULT_ES_HOST,
    DEFAULT_ES_USER,
    DEFAULT_PASSWORD_KEY,
    DEFAULT_PASSWORD_NAMESPACE,
    DEFAULT_PASSWORD_SECRET,
    EXPERIMENT_ROOT,
    Elasticsearch,
    HtmlChunk,
    index_uuid,
    parse_hit,
    require_experiment_path,
    resolve_password,
    sha256_file,
    sha256_text,
    stable_unique,
    scroll_hits,
    utc_now,
    write_json_atomic,
    write_jsonl_atomic,
)


def is_acknowledgement_url(url: str) -> bool:
    path = url.lower().split("?", 1)[0].split("#", 1)[0]
    return "acknowledgement" in path or "acknowledgment" in path


def build_curator_ready_documents(
    chunks: Iterable[HtmlChunk], collection: str
) -> list[dict[str, Any]]:
    by_url: dict[str, list[HtmlChunk]] = defaultdict(list)
    for chunk in chunks:
        if not is_acknowledgement_url(chunk.url):
            by_url[chunk.url].append(chunk)

    documents: list[dict[str, Any]] = []
    for url in sorted(by_url):
        source_chunks = sorted(by_url[url], key=lambda chunk: (chunk.chunk_index, chunk.es_id))
        retained: list[HtmlChunk] = []
        seen_text_hashes: set[str] = set()
        dropped_duplicate_ids: list[str] = []
        for chunk in source_chunks:
            text_hash = sha256_text(chunk.text)
            if text_hash in seen_text_hashes:
                dropped_duplicate_ids.append(chunk.es_id)
                continue
            seen_text_hashes.add(text_hash)
            retained.append(chunk)

        text = "\n\n".join(chunk.text for chunk in retained).strip()
        document_id = f"url_doc_{sha256_text(collection + chr(10) + url)[:24]}"
        product_families = stable_unique(chunk.product_family for chunk in retained)
        product_names = stable_unique(chunk.product_name for chunk in retained)
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
                "source_chunk_count": len(source_chunks),
                "chunk_count": len(retained),
                "dropped_exact_duplicate_chunk_count": len(dropped_duplicate_ids),
                "dropped_exact_duplicate_es_chunk_ids": dropped_duplicate_ids,
                "es_chunk_ids": [chunk.es_id for chunk in retained],
                "chunk_indices": [chunk.chunk_index for chunk in retained],
                "source_chunk_ids": stable_unique(chunk.source_chunk_id for chunk in retained),
                "source_revision_ids": stable_unique(
                    chunk.source_revision_id for chunk in retained
                ),
                "source_systems": stable_unique(chunk.source_system for chunk in retained),
                "source_kinds": stable_unique(chunk.source_kind for chunk in retained),
                "modalities": stable_unique(chunk.modality for chunk in retained),
                "product_family": product_families[0] if len(product_families) == 1 else "mixed",
                "product_name": product_names[0] if len(product_names) == 1 else "mixed",
                "product_families": product_families,
                "product_names": product_names,
                "doc_kind": "html",
            }
        )
    return documents


def build_ready_manifest(
    *,
    collection: str,
    es_host: str,
    es_index_uuid: str | None,
    output: Path,
    raw_hit_count: int,
    html_chunk_count: int,
    excluded_counts: Counter[str],
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    chunk_counts = [int(document["chunk_count"]) for document in documents]
    word_counts = [int(document["word_count"]) for document in documents]
    dropped_duplicates = sum(
        int(document["dropped_exact_duplicate_chunk_count"]) for document in documents
    )
    return {
        "schema_version": "curator_dataset.curator_ready_manifest.v1",
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
            "exclude_url_substrings": ["acknowledgement", "acknowledgment"],
            "group_by": "resolved content_url/canonical_uri",
            "order_by": ["chunk_index", "_id"],
            "separator": "two newlines",
            "exact_text_deduplication": {
                "scope": "within URL",
                "key": "SHA-256 of stripped chunk text",
                "retention": "first in deterministic order",
            },
        },
        "counts": {
            "raw_es_hits": raw_hit_count,
            "html_chunks_before_acknowledgement_filter": html_chunk_count,
            "excluded_hits": dict(sorted(excluded_counts.items())),
            "url_documents": len(documents),
            "retained_chunks": sum(chunk_counts),
            "dropped_exact_duplicate_chunks": dropped_duplicates,
            "documents_with_dropped_exact_duplicate_chunks": sum(
                document["dropped_exact_duplicate_chunk_count"] > 0
                for document in documents
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
        default=(
            EXPERIMENT_ROOT / "data" / "nim_curated" / "url_documents.curator_ready.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            EXPERIMENT_ROOT
            / "data"
            / "nim_curated"
            / "url_documents.curator_ready.manifest.json"
        ),
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

    excluded_counts: Counter[str] = Counter()
    html_chunks: list[HtmlChunk] = []
    raw_hit_count = 0
    for hit in scroll_hits(client, args.collection, args.page_size):
        raw_hit_count += 1
        chunk, reason = parse_hit(hit)
        if chunk is None:
            excluded_counts[reason or "unknown"] += 1
        elif is_acknowledgement_url(chunk.url):
            excluded_counts["acknowledgement_page"] += 1
        else:
            html_chunks.append(chunk)

    documents = build_curator_ready_documents(html_chunks, args.collection)
    if not documents:
        raise RuntimeError(f"no Curator-ready documents produced from {args.collection!r}")
    write_jsonl_atomic(output, documents)
    manifest = build_ready_manifest(
        collection=args.collection,
        es_host=args.es_host,
        es_index_uuid=index_uuid(client, args.collection),
        output=output,
        raw_hit_count=raw_hit_count,
        html_chunk_count=len(html_chunks) + excluded_counts["acknowledgement_page"],
        excluded_counts=excluded_counts,
        documents=documents,
    )
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
