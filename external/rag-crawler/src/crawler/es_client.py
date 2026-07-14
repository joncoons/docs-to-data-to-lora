"""
es_client.py — direct Elasticsearch operations for rag-crawler.

Provides:
  - ensure_index(name, session, ssl_ctx)  — create index if it does not exist
  - bulk_write(index, chunks, session, ssl_ctx) -> int  — embed + bulk index
  - delete_by_source_uris(index, uris, session, ssl_ctx) -> int

Document schema matches nvidia_rag.storage.embed_store._build_doc exactly so
chunks written here are returned by the existing RAG server query pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import ssl
from datetime import UTC, datetime
from typing import Any

import aiohttp

from . import config
from .provenance import build_es_provenance

logger = logging.getLogger(__name__)

# Dense-vector index mapping — mirrors the langchain_elasticsearch VectorStore
# schema (text_field="text", vector_field="vector", DenseVectorStrategy).
_INDEX_MAPPING: dict[str, Any] = {
    "mappings": {
        "dynamic": "true",
        "properties": {
            "text": {"type": "text"},
            "vector": {
                "type": "dense_vector",
                "dims": config.EMBED_DIMENSIONS,
                "index": True,
                "similarity": "cosine",
            },
            "metadata": {"type": "object", "dynamic": True},
        },
    }
}


def _auth(session_kwargs: dict) -> tuple[str, str] | None:
    """Return (user, pass) tuple if credentials are configured."""
    if config.ES_USER and config.ES_PASS:
        return (config.ES_USER, config.ES_PASS)
    return None


async def ensure_index(
    index: str,
    session: aiohttp.ClientSession,
    ssl_ctx: ssl.SSLContext | None,
) -> None:
    """Create ES index with the standard vector mapping if it does not exist."""
    url = f"{config.ES_URL.rstrip('/')}/{index}"
    auth = aiohttp.BasicAuth(config.ES_USER, config.ES_PASS) if config.ES_USER else None
    ssl_param: ssl.SSLContext | bool = ssl_ctx if ssl_ctx is not None else False

    try:
        # HEAD request — fast existence check
        async with session.head(url, auth=auth, ssl=ssl_param) as resp:
            if resp.status == 200:
                return  # already exists
    except Exception:
        pass

    # Create the index
    try:
        async with session.put(
            url,
            json=_INDEX_MAPPING,
            auth=auth,
            ssl=ssl_param,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status in (200, 201):
                logger.info("es_client: created index '%s'", index)
            elif resp.status == 400:
                body = await resp.json()
                # resource_already_exists_exception is fine
                if "already exists" in str(body):
                    return
                logger.warning("es_client: PUT /%s returned 400: %s", index, body)
            else:
                resp.raise_for_status()
    except Exception as exc:
        logger.warning("es_client: ensure_index('%s') failed: %r", index, exc)


def _build_doc(
    text: str,
    vector: list[float],
    source_uri: str,
    chunk_index: int,
    metadata: dict,
) -> dict:
    """
    Build an ES document matching nvidia_rag.storage.embed_store._build_doc.

    Fields:
      text          — chunk text (used by BM25 full-text search)
      vector        — dense embedding (used by kNN search)
      metadata      — nested object with source + content_metadata sub-objects
    """
    now = datetime.now(UTC).isoformat()
    doc_id = hashlib.sha256(
        f"{source_uri}:{chunk_index}".encode()
    ).hexdigest()[:16]

    provenance = build_es_provenance(
        text=text,
        source_uri=source_uri,
        chunk_index=chunk_index,
        metadata=metadata,
        date_created=now,
    )

    # Merge caller-supplied metadata into content_metadata
    content_meta: dict = {
        "type": "text",
        "content_url": source_uri,
        "chunk_index": chunk_index,
        "document_type": metadata.get("document_type", "text"),
        "filename": metadata.get("filename", source_uri),
    }
    # Promote known flat fields into content_metadata. These fields cover web
    # crawl, document capture/OCR, dense image captioning, and video/audio
    # summarization text without making webcrawl the only provenance shape.
    for field in (
        "page_title", "heading", "section_path", "crawl_depth",
        "meta_description", "section_h1", "source_system",
        "source_kind", "modality", "product_family", "product_name",
        "referring_page_url", "retrieved_at", "captured_at", "processed_at",
        "parser_version", "chunker_version", "source_revision_id",
        "source_chunk_id", "ingestion_run_id", "crawl_run_id",
        "source_content_hash", "raw_sha256", "text_sha256",
        "final_uri", "canonical_uri", "http_status_code", "http_etag",
        "http_last_modified", "content_type", "page_number", "bbox",
        "time_start_seconds", "time_end_seconds", "frame_start", "frame_end",
    ):
        if field in metadata:
            content_meta[field] = metadata[field]
    content_meta.update(provenance["content_metadata"])

    source_meta = {
        "source_id": doc_id,
        "source_name": metadata.get("source_uri", source_uri),
        "source_type": metadata.get("source_type", "web"),
        "date_created": now,
    }
    source_meta.update(provenance["source"])

    return {
        "text": text,
        "vector": vector,
        "metadata": {
            "source": source_meta,
            "content_metadata": content_meta,
            "provenance": provenance["provenance"],
        },
    }


async def bulk_write(
    index: str,
    chunks: list[tuple[str, dict]],  # (chunk_text, metadata)
    vectors: list[list[float] | None],
    session: aiohttp.ClientSession,
    ssl_ctx: ssl.SSLContext | None,
) -> int:
    """
    Bulk-index (chunk_text, metadata) pairs into *index*.

    *vectors* must be aligned with *chunks*.  Items with a None vector are
    skipped.  Returns count of successfully indexed documents.
    """
    if not chunks:
        return 0

    auth = aiohttp.BasicAuth(config.ES_USER, config.ES_PASS) if config.ES_USER else None
    ssl_param: ssl.SSLContext | bool = ssl_ctx if ssl_ctx is not None else False
    bulk_url = f"{config.ES_URL.rstrip('/')}/{index}/_bulk"
    batch_size = config.ES_BULK_BATCH_SIZE

    docs = []
    for i, ((text, meta), vec) in enumerate(zip(chunks, vectors)):
        if vec is None:
            logger.warning("es_client: skipping chunk %d (no vector)", i)
            continue
        docs.append(_build_doc(
            text=text,
            vector=vec,
            source_uri=meta.get("source_uri", ""),
            chunk_index=i,
            metadata=meta,
        ))

    total_indexed = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        lines: list[str] = []
        for doc in batch:
            lines.append(json.dumps({"index": {"_index": index}}))
            lines.append(json.dumps(doc))
        body = "\n".join(lines) + "\n"

        try:
            async with session.post(
                bulk_url,
                data=body,
                headers={"Content-Type": "application/x-ndjson"},
                auth=auth,
                ssl=ssl_param,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()

            if result.get("errors"):
                failed = sum(
                    1
                    for item in result.get("items", [])
                    if "error" in item.get("index", {})
                )
                logger.warning(
                    "es_client: %d/%d docs failed in batch at offset %d",
                    failed, len(batch), i,
                )
                total_indexed += len(batch) - failed
            else:
                total_indexed += len(batch)

        except Exception as exc:
            logger.warning("es_client: bulk batch at offset %d failed: %r", i, exc)

    return total_indexed


async def delete_by_source_uris(
    index: str,
    source_uris: list[str],
    session: aiohttp.ClientSession,
    ssl_ctx: ssl.SSLContext | None,
) -> int:
    """
    Delete all documents in *index* whose content_url matches any of *source_uris*.

    Returns count of deleted documents.
    """
    if not source_uris:
        return 0

    auth = aiohttp.BasicAuth(config.ES_USER, config.ES_PASS) if config.ES_USER else None
    ssl_param: ssl.SSLContext | bool = ssl_ctx if ssl_ctx is not None else False
    url = f"{config.ES_URL.rstrip('/')}/{index}/_delete_by_query"

    query = {
        "query": {
            "terms": {
                "metadata.content_metadata.content_url.keyword": source_uris
            }
        }
    }

    try:
        async with session.post(
            url,
            json=query,
            auth=auth,
            ssl=ssl_param,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()
            deleted = result.get("deleted", 0)
            logger.info(
                "es_client: deleted %d doc(s) from '%s' for %d URI(s)",
                deleted, index, len(source_uris),
            )
            return deleted
    except Exception as exc:
        logger.warning(
            "es_client: delete_by_source_uris in '%s' failed: %r", index, exc
        )
        return 0
