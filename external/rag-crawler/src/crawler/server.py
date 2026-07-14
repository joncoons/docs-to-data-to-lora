"""
server.py — FastAPI HTTP server for rag-crawler.

Endpoints:
  POST /crawl            — start a new crawl (returns task_id immediately)
  GET  /status           — poll task state + progress
  POST /cancel           — request cancellation of a running crawl
  POST /ingest-binaries  — trigger Phase 3: POST manifest files to ingestor
  GET  /health           — liveness probe
  GET  /collections      — list known ES collection names from the product URL map
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, task_handler
from . import scheduler as _scheduler
from . import domain_config as _domain_cfg
from .crawl import (
    SimpleWebCrawler,
    load_binary_manifest,
    load_url_registry,
    save_url_registry,
)
from .product_url_map import CRAWLER_PRODUCT_URL_MAP

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _domain_cfg.seed_default_configs()
    _scheduler.start()
    yield
    _scheduler.stop()


app = FastAPI(
    title="rag-crawler",
    description="Standalone async web crawler — HTML→embed→ES, binaries→NFS→Phase 3",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / response models ─────────────────────────────────────────────────

class CrawlRequest(BaseModel):
    start_url: str = Field(..., description="Seed URL to begin crawling from")
    collection_name: str | None = Field(
        None,
        description=(
            "ES index to write into.  If omitted the product URL map is used "
            "to route each page to the matching product collection."
        ),
    )
    max_pages: int | None = Field(None, description="Page limit (null = unlimited)")
    max_depth: int | None = Field(None, description="BFS depth limit (null = unlimited)")
    batch_ingest_size: int = Field(
        50, description="HTML pages to buffer before flushing to ES"
    )
    extract_linked_files: bool = Field(
        True, description="Download PDFs/DOCX/etc. to NFS repo dirs during crawl"
    )
    use_sitemap: bool = Field(
        False,
        description="Seed BFS queue from robots.txt sitemaps before crawling",
    )
    allowed_url_prefixes: list[str] | None = Field(
        None,
        description=(
            "Restrict crawl to URLs with these prefixes. "
            "null = auto-derive from domain config or start URL path."
        ),
    )
    use_product_url_map: bool | None = Field(
        None,
        description=(
            "Override domain config: True forces product URL map routing, "
            "False forces fixed collection_name. null = use domain config default."
        ),
    )
    blocked_url_patterns: list[str] | None = Field(
        None,
        description=(
            "Additional URL substrings to block (appended to built-in defaults). "
            "Use unblock_url_patterns to remove items from the defaults instead."
        ),
    )
    unblock_url_patterns: list[str] | None = Field(
        None,
        description=(
            "URL substrings to REMOVE from the default block list. "
            "Example: ['github.com'] to allow GitHub link following. "
            "Patterns are matched exactly against the default-blocked list."
        ),
    )
    binary_host_allowlist: list[str] | None = Field(
        None,
        description=(
            "Cross-host substrings that bypass the same-host check for binary "
            "and inline-text downloads (e.g., 'raw.githubusercontent.com/NVIDIA/'). "
            "HTML link following remains same-host even with this set."
        ),
    )
    seed_urls: list[str] | None = Field(
        None,
        description=(
            "Additional URLs to enqueue at BFS depth 0 alongside start_url. "
            "Use for docs sites whose sidebar navigation is JS-rendered and "
            "not exposed in the initial HTML — caller seeds known section-index "
            "pages from a sitemap or manual enumeration. Each seed must still "
            "pass allowed_url_prefixes / blocked / same-host gates."
        ),
    )


class CrawlResponse(BaseModel):
    task_id: str
    message: str = "Crawl started"


class StatusResponse(BaseModel):
    task_id: str
    state: str
    result: dict[str, Any]
    progress: dict[str, Any] | None


class CancelRequest(BaseModel):
    task_id: str


class ExtractionMethod(str, Enum):
    auto = "auto"           # PDF → nemotron_parse (forced), all others → nv_ingest
    nemotron_parse = "nemotron_parse"
    nv_ingest = "nv_ingest"
    both = "both"


class IngestBinariesRequest(BaseModel):
    collection_name: str | None = Field(
        None,
        description=(
            "Manifest slug to process (e.g. 'docs_nvidia_com').  "
            "If null, all binary manifests in the registry are processed."
        ),
    )
    method: ExtractionMethod = Field(
        ExtractionMethod.auto,
        description=(
            "Extraction pipeline: 'auto' (PDF→nemotron_parse forced, others→nv_ingest), "
            "'nemotron_parse', 'nv_ingest', or 'both' (A/B comparison)."
        ),
    )
    max_files: int | None = Field(
        None, description="Limit number of files to process (null = all pending)"
    )
    force_reingest: bool = Field(
        False,
        description="Re-ingest files even if last_ingested_hash matches content_hash",
    )
    url_filter: str | None = Field(
        None,
        description=(
            "Only ingest files whose source_uri or referring_page_url contains this substring. "
            "Example: '/en-us/' to restrict to US-English PDFs."
        ),
    )
    max_concurrent: int = Field(
        1,
        ge=1,
        le=16,
        description="Number of files to POST to the ingestor concurrently (default 1).",
    )


class IngestBinariesResponse(BaseModel):
    task_id: str
    message: str = "Binary ingest started"


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_all_manifests(registry_dir: str) -> list[dict]:
    """Load and combine every *_binary_manifest.csv in the registry directory."""
    rows: list[dict] = []
    if not os.path.isdir(registry_dir):
        return rows
    for fname in os.listdir(registry_dir):
        if not fname.endswith("_binary_manifest.csv"):
            continue
        path = os.path.join(registry_dir, fname)
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
        except Exception as exc:
            logger.warning("Could not read manifest %s: %s", path, exc)
    return rows


def _resolve_method_for_file(
    filename: str, requested: ExtractionMethod
) -> list[ExtractionMethod]:
    """Return the list of extraction methods to apply to a single file."""
    ext = os.path.splitext(filename)[1].lower()
    if requested == ExtractionMethod.auto:
        if ext == ".pdf":
            return [ExtractionMethod.nemotron_parse]
        return [ExtractionMethod.nv_ingest]
    if requested == ExtractionMethod.both:
        return [ExtractionMethod.nemotron_parse, ExtractionMethod.nv_ingest]
    return [requested]


# ── Background runners ────────────────────────────────────────────────────────

async def _run_crawl(task_id: str, req: CrawlRequest) -> None:
    """Async task that drives a full crawl lifecycle.

    Domain config is resolved first; request fields override it.
    """
    try:
        # Resolve domain config (longest-match on hostname)
        domain_cfg = _domain_cfg.match_config(req.start_url)

        # use_sitemap: request > domain config > False
        effective_sitemap = req.use_sitemap or (
            domain_cfg.sitemap.enabled if domain_cfg else False
        )

        # allowed_url_prefixes: request (explicit list) > domain config defaults
        if req.allowed_url_prefixes is not None:
            effective_prefixes = req.allowed_url_prefixes or None
        else:
            dc_prefixes = (domain_cfg.crawl_defaults.allowed_url_prefixes if domain_cfg else [])
            effective_prefixes = dc_prefixes if dc_prefixes else None

        # use_product_url_map: request (explicit bool) > domain config mode
        if req.use_product_url_map is not None:
            effective_purl_map = req.use_product_url_map
        elif domain_cfg:
            effective_purl_map = domain_cfg.collection_routing.mode == "product_url_map"
        else:
            effective_purl_map = req.collection_name is None  # compatibility: null collection -> map

        # max_depth / batch_ingest_size: request > domain config defaults
        effective_max_depth = req.max_depth
        if effective_max_depth is None and domain_cfg:
            effective_max_depth = domain_cfg.crawl_defaults.max_depth

        effective_batch = req.batch_ingest_size
        if effective_batch == 50 and domain_cfg:
            effective_batch = domain_cfg.crawl_defaults.batch_ingest_size

        logger.info(
            "_run_crawl: task=%s url=%s sitemap=%s purl_map=%s prefixes=%s",
            task_id, req.start_url, effective_sitemap, effective_purl_map, effective_prefixes,
        )

        # Fallback collection: explicit request value > domain config default > "default"
        fallback_collection = (
            req.collection_name
            or (domain_cfg.collection_routing.default_collection if domain_cfg else None)
            or "default"
        )

        crawler = SimpleWebCrawler(
            start_url=req.start_url,
            max_pages=req.max_pages,
            max_depth=effective_max_depth,
            batch_size=effective_batch,
            collection_name=fallback_collection,
            extract_linked_files=req.extract_linked_files,
            use_sitemap=effective_sitemap,
            allowed_url_prefixes=effective_prefixes,
            use_product_url_map=effective_purl_map,
            blocked_url_patterns=req.blocked_url_patterns,
            unblock_url_patterns=req.unblock_url_patterns,
            binary_host_allowlist=req.binary_host_allowlist,
            seed_urls=req.seed_urls,
            task_id=task_id,
        )
        result = await crawler.crawl(collection_name=fallback_collection)
        task_handler.set_task(task_id, "FINISHED", result or {})
    except Exception as exc:
        logger.exception("_run_crawl: unhandled exception for task %s", task_id)
        task_handler.set_task(task_id, "FAILURE", {"error": str(exc)})


async def _run_ingest_binaries(task_id: str, req: IngestBinariesRequest) -> None:
    """Async task that POSTs manifest binary files to the ingestor.

    Collection routing: each manifest row carries a per-file collection_name
    set during crawl by the product URL map.  That value is used as-is; the
    request-level collection_name is only a fallback for rows that lack one.

    Method routing (method=auto):
      .pdf  → nemotron_parse with force_nemotron_parse=True
      other → nv_ingest (use_nemotron_parse=False)
    """
    # Load manifest(s) — all CSVs in registry when collection_name is null
    if req.collection_name:
        manifest: list[dict] = load_binary_manifest(req.collection_name, config.REGISTRY_DIR)
        if not manifest:
            manifest = load_binary_manifest(req.collection_name, config.EXPORT_DIR)
    else:
        manifest = _load_all_manifests(config.REGISTRY_DIR)
        if not manifest:
            manifest = _load_all_manifests(config.EXPORT_DIR)

    pending = [
        row for row in manifest
        if row.get("local_path") and os.path.exists(row["local_path"])
        and row.get("media_type", "document") in ("document", "inline")
        and (req.force_reingest or row.get("content_hash") != row.get("last_ingested_hash", ""))
    ]
    if req.url_filter:
        pending = [
            row for row in pending
            if req.url_filter in (row.get("source_uri") or "")
            or req.url_filter in (row.get("referring_page_url") or "")
        ]
    if req.max_files is not None:
        pending = pending[: req.max_files]

    task_handler.set_progress(task_id, {
        "task_type": "ingest_binaries",
        "collection_name": req.collection_name or "all",
        "method": req.method,
        "total": len(pending),
        "ingested": 0,
        "failed": 0,
    })

    counters = {"ingested": 0, "failed": 0}
    ingested_urls: dict[str, dict[str, str]] = {}
    sem = asyncio.Semaphore(req.max_concurrent)

    async def _ingest_one(session: aiohttp.ClientSession, row: dict) -> None:
        async with sem:
            filename = row["filename"]
            source_uri = row["source_uri"]
            collection = row.get("collection_name") or req.collection_name or ""
            if not collection:
                logger.warning("ingest_binaries: no collection for %s — skipping", filename)
                counters["failed"] += 1
                return

            file_methods = _resolve_method_for_file(filename, req.method)
            file_ok = True

            for method in file_methods:
                use_nemotron = method == ExtractionMethod.nemotron_parse
                ext = os.path.splitext(filename)[1].lower()
                force_nemotron = use_nemotron and ext == ".pdf"
                metadata = {
                    "source_uri": source_uri,
                    "filename": filename,
                    "referring_page_url": row.get("referring_page_url", ""),
                    "crawl_depth": int(row.get("crawl_depth", 0) or 0),
                    "document_type": ext.lstrip("."),
                    "source_system": row.get("source_system") or "web_crawl",
                    "source_kind": row.get("source_kind") or "downloaded_asset",
                    "modality": row.get("modality") or row.get("media_type") or "document",
                    "source_revision_id": row.get("source_revision_id", ""),
                    "ingestion_run_id": row.get("ingestion_run_id", ""),
                    "crawl_run_id": row.get("crawl_run_id", ""),
                    "source_content_hash": row.get("source_content_hash") or row.get("content_hash", ""),
                    "raw_sha256": row.get("raw_sha256") or row.get("content_hash", ""),
                    "retrieved_at": row.get("retrieved_at") or row.get("downloaded_at", ""),
                    "final_uri": row.get("final_uri") or source_uri,
                    "content_type": row.get("content_type", ""),
                    "parser_version": row.get("parser_version", ""),
                    "chunker_version": row.get("chunker_version", ""),
                    "extraction_method": method.value,
                }
                try:
                    with open(row["local_path"], "rb") as fh:
                        file_bytes = fh.read()
                    post_data = aiohttp.FormData()
                    post_data.add_field("data", json.dumps({
                        "collection_name": collection,
                        "use_nemotron_parse": use_nemotron,
                        "force_nemotron_parse": force_nemotron,
                        "custom_metadata": [{"filename": filename, "metadata": metadata}],
                    }))
                    post_data.add_field(
                        "documents", file_bytes,
                        filename=filename,
                        content_type="application/octet-stream",
                    )
                    async with session.post(
                        f"{config.INGESTOR_URL}/documents",
                        data=post_data,
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        resp.raise_for_status()
                    logger.info(
                        "ingest_binaries: %s → %s [%s%s]",
                        filename, collection, method.value,
                        " forced" if force_nemotron else "",
                    )
                except Exception as exc:
                    logger.warning(
                        "ingest_binaries: failed %s [%s]: %r", filename, method.value, exc
                    )
                    counters["failed"] += 1
                    file_ok = False

            if file_ok:
                counters["ingested"] += 1
                row["last_ingested_hash"] = row.get("content_hash", "")
                ingested_urls[source_uri] = {
                    "collection_name": row.get("collection_name", ""),
                    "content_hash": row.get("content_hash", ""),
                }
            task_handler.set_progress(task_id, {
                "task_type": "ingest_binaries",
                "collection_name": req.collection_name or "all",
                "method": req.method,
                "total": len(pending),
                "ingested": counters["ingested"],
                "failed": counters["failed"],
            })

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[_ingest_one(session, row) for row in pending])

    ingested = counters["ingested"]
    failed = counters["failed"]

    # Persist updated last_ingested_hash back to each manifest file
    if req.collection_name:
        _save_updated_manifest(manifest, req.collection_name)
    else:
        _save_all_updated_manifests(manifest, config.REGISTRY_DIR)

    # Update URL registry with last_ingested / last_ingested_hash
    if ingested_urls:
        _update_registries_after_ingest(ingested_urls, config.REGISTRY_DIR)

    task_handler.set_task(task_id, "FINISHED", {
        "collection_name": req.collection_name or "all",
        "method": req.method,
        "total": len(pending),
        "ingested": ingested,
        "failed": failed,
        "message": (
            f"Binary ingest complete: {ingested}/{len(pending)} files ingested "
            f"via {req.method.value} ({failed} failed)."
        ),
    })


def _save_updated_manifest(manifest: list[dict], collection_name: str) -> None:
    """Write the manifest back to the registry dir with updated last_ingested_hash."""
    if not manifest:
        return
    slug = re.sub(r"[^a-z0-9_-]", "_", collection_name.lower()).strip("_")
    path = os.path.join(config.REGISTRY_DIR, f"{slug}_binary_manifest.csv")
    try:
        fieldnames = list(manifest[0].keys())
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)
    except Exception as exc:
        logger.warning("Could not save updated manifest to %s: %s", path, exc)


def _save_all_updated_manifests(rows: list[dict], registry_dir: str) -> None:
    """After a combined ingest, re-partition rows back to their source manifest files."""
    if not rows or not os.path.isdir(registry_dir):
        return
    # Group rows by their originating manifest slug (inferred from source_uri netloc
    # or collection_name — same logic the crawler used when writing them)
    from urllib.parse import urlparse as _urlparse
    by_file: dict[str, list[dict]] = {}
    for row in rows:
        coll = row.get("collection_name", "")
        # Try to find which manifest file this row came from by scanning existing files
        # Best effort: use the netloc of source_uri as the slug
        uri = row.get("source_uri", "")
        if uri:
            netloc = _urlparse(uri).netloc.split(":")[0].replace(".", "_").replace("-", "_")
        else:
            netloc = re.sub(r"[^a-z0-9_-]", "_", coll.lower()).strip("_")
        by_file.setdefault(netloc, []).append(row)

    for slug, slug_rows in by_file.items():
        path = os.path.join(registry_dir, f"{slug}_binary_manifest.csv")
        if not os.path.exists(path):
            continue  # only update files that already exist
        try:
            fieldnames = list(slug_rows[0].keys())
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(slug_rows)
        except Exception as exc:
            logger.warning("Could not save updated manifest to %s: %s", path, exc)


def _update_registries_after_ingest(
    ingested: dict[str, dict[str, str]],
    registry_dir: str,
) -> None:
    """Update URL registry entries for successfully ingested binary files.

    Args:
        ingested: mapping of {source_uri: {"collection_name": ..., "content_hash": ...}}
        registry_dir: path to the NFS registry directory
    """
    from datetime import datetime, timezone

    # Group URIs by collection
    by_collection: dict[str, dict[str, str]] = {}
    for uri, meta in ingested.items():
        coll = meta.get("collection_name", "")
        if coll:
            by_collection.setdefault(coll, {})[uri] = meta.get("content_hash", "")

    now_iso = datetime.now(timezone.utc).isoformat()
    for collection_name, url_hashes in by_collection.items():
        registry = load_url_registry(collection_name, registry_dir)
        for uri, content_hash in url_hashes.items():
            entry = registry.get(uri, {})
            registry[uri] = {
                **entry,
                "last_ingested": now_iso,
                "last_ingested_hash": content_hash,
            }
        save_url_registry(collection_name, registry_dir, registry)
        logger.info(
            "_update_registries_after_ingest: updated %d entries in %s registry",
            len(url_hashes), collection_name,
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/crawl", response_model=CrawlResponse, status_code=202)
async def start_crawl(req: CrawlRequest) -> CrawlResponse:
    """Start a crawl in the background and return a task_id for polling."""
    task_id = task_handler.new_task_id()
    task_handler.set_task(task_id, "PENDING")
    asyncio.create_task(_run_crawl(task_id, req))
    logger.info("start_crawl: task_id=%s url=%s", task_id, req.start_url)
    return CrawlResponse(task_id=task_id)


@app.get("/status", response_model=StatusResponse)
async def get_status(task_id: str) -> StatusResponse:
    """Return current state and optional progress for a crawl task."""
    record = task_handler.get_task(task_id)
    if record.get("state") == "UNKNOWN":
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    progress = task_handler.get_progress(task_id)
    return StatusResponse(
        task_id=task_id,
        state=record["state"],
        result=record.get("result", {}),
        progress=progress,
    )


@app.post("/cancel", status_code=202)
async def cancel_crawl(req: CancelRequest) -> dict[str, str]:
    """Request cancellation of a running crawl."""
    record = task_handler.get_task(req.task_id)
    if record.get("state") == "UNKNOWN":
        raise HTTPException(status_code=404, detail=f"Task {req.task_id!r} not found")
    task_handler.request_cancel(req.task_id)
    logger.info("cancel_crawl: requested cancel for task_id=%s", req.task_id)
    return {"task_id": req.task_id, "message": "Cancel requested"}


@app.post("/ingest-binaries", response_model=IngestBinariesResponse, status_code=202)
async def ingest_binaries(req: IngestBinariesRequest) -> IngestBinariesResponse:
    """
    Trigger Phase 3: POST binary files from the NFS manifest to the ingestor.

    The extraction method is recorded in each chunk's metadata so results from
    different pipelines can be compared for latency and accuracy.
    """
    task_id = task_handler.new_task_id()
    task_handler.set_task(task_id, "PENDING")
    asyncio.create_task(_run_ingest_binaries(task_id, req))
    logger.info(
        "ingest_binaries: task_id=%s collection=%s method=%s",
        task_id, req.collection_name, req.method,
    )
    return IngestBinariesResponse(task_id=task_id)


@app.get("/binary-manifest")
async def get_binary_manifest(collection_name: str | None = None) -> dict[str, Any]:
    """
    Return the binary manifest — pending docs ready for Phase 3 ingest.
    If collection_name is omitted, combines all manifests from the registry.
    Filters to files that exist on disk and haven't been ingested yet.
    """
    if collection_name:
        manifest = load_binary_manifest(collection_name, config.REGISTRY_DIR)
        if not manifest:
            manifest = load_binary_manifest(collection_name, config.EXPORT_DIR)
    else:
        manifest = _load_all_manifests(config.REGISTRY_DIR)
        if not manifest:
            manifest = _load_all_manifests(config.EXPORT_DIR)

    pending_docs = [
        r for r in manifest
        if r.get("media_type", "document") in ("document", "inline")
        and r.get("local_path")
        and os.path.exists(r["local_path"])
        and r.get("content_hash") != r.get("last_ingested_hash", "")
    ]
    pending_media = [
        r for r in manifest
        if r.get("media_type") in ("audio", "video")
        and r.get("local_path")
        and os.path.exists(r["local_path"])
    ]
    return {
        "collection_name": collection_name,
        "pending_docs": pending_docs,
        "pending_media": pending_media,
        "total": len(manifest),
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/collections")
async def list_collections() -> dict[str, Any]:
    """
    Return the distinct collection (ES index) names that would be produced
    by the product URL map, along with their product families.
    """
    seen: dict[str, list[str]] = {}
    for prefix, family, product_name in CRAWLER_PRODUCT_URL_MAP:
        slug = _slugify(family)
        seen.setdefault(slug, [])
        if product_name and product_name not in seen[slug]:
            seen[slug].append(product_name)

    collections = [
        {"index": slug, "products": products}
        for slug, products in sorted(seen.items())
    ]
    return {"count": len(collections), "collections": collections}


def _slugify(name: str) -> str:
    """Convert a product family name to an ES-safe index slug."""
    return re.sub(r"[^a-z0-9_-]", "_", name.lower()).strip("_")


# ── Schedule endpoints ────────────────────────────────────────────────────────

class CreateScheduleRequest(BaseModel):
    label: str = Field(..., description="Display name for the schedule")
    start_url: str = Field(..., description="Seed URL to crawl")
    collection_name: str | None = Field(None, description="Target collection; null = product URL map routing")
    max_pages: int | None = Field(None, description="Page limit; null = unlimited")
    max_depth: int | None = Field(None, description="BFS depth; null = unlimited")
    batch_ingest_size: int = Field(50, description="Pages to buffer per ES flush")
    extract_linked_files: bool = Field(False, description="Download PDFs/DOCX to NFS")
    cron_expression: str = Field(..., description='5-field cron, e.g. "0 2 * * 0" = weekly Sunday 2am UTC')
    enabled: bool = Field(True, description="Whether the schedule is active")


class UpdateScheduleRequest(BaseModel):
    label: str | None = None
    cron_expression: str | None = None
    enabled: bool | None = None
    max_pages: int | None = None
    max_depth: int | None = None


@app.get("/schedules")
async def list_schedules() -> dict[str, Any]:
    """Return all configured crawl schedules."""
    return {"schedules": [s.model_dump() for s in _scheduler.list_schedules()]}


@app.post("/schedules", status_code=201)
async def create_schedule(req: CreateScheduleRequest) -> dict[str, Any]:
    """Create a new recurring crawl schedule."""
    try:
        sched = _scheduler.create_schedule(req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    logger.info("create_schedule: id=%s cron=%s url=%s", sched.id, sched.cron_expression, sched.start_url)
    return sched.model_dump()


@app.patch("/schedules/{schedule_id}")
async def update_schedule(schedule_id: str, req: UpdateScheduleRequest) -> dict[str, Any]:
    """Enable/disable a schedule or update its cron expression."""
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    sched = _scheduler.update_schedule(schedule_id, updates)
    if sched is None:
        raise HTTPException(status_code=404, detail=f"Schedule {schedule_id!r} not found")
    return sched.model_dump()


@app.delete("/schedules/{schedule_id}", status_code=204)
async def delete_schedule(schedule_id: str) -> None:
    """Remove a schedule permanently."""
    if not _scheduler.delete_schedule(schedule_id):
        raise HTTPException(status_code=404, detail=f"Schedule {schedule_id!r} not found")
    logger.info("delete_schedule: id=%s", schedule_id)


# ── Domain config endpoints ───────────────────────────────────────────────────

@app.get("/domain-configs")
async def list_domain_configs() -> dict[str, Any]:
    """Return all per-domain crawl profiles."""
    return {"configs": [c.model_dump() for c in _domain_cfg.list_configs()]}


@app.get("/domain-configs/{slug}")
async def get_domain_config(slug: str) -> dict[str, Any]:
    """Return a single domain config by slug (e.g. docs_nvidia_com)."""
    cfg = _domain_cfg.load_config(slug)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Domain config {slug!r} not found")
    return cfg.model_dump()


@app.put("/domain-configs/{slug}", status_code=200)
async def put_domain_config(slug: str, body: dict) -> dict[str, Any]:
    """Create or replace a domain config. Body must be a DomainConfig JSON object."""
    try:
        cfg = _domain_cfg.DomainConfig(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    # Enforce slug consistency: use the slug from the URL, not the body domain
    if _domain_cfg.domain_to_slug(cfg.domain) != slug:
        raise HTTPException(
            status_code=422,
            detail=f"Slug mismatch: URL slug {slug!r} does not match domain {cfg.domain!r} → {_domain_cfg.domain_to_slug(cfg.domain)!r}",
        )
    try:
        _domain_cfg.save_config(cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    logger.info("put_domain_config: saved %s", slug)
    return cfg.model_dump()


@app.delete("/domain-configs/{slug}", status_code=204)
async def delete_domain_config(slug: str) -> None:
    """Delete a domain config permanently."""
    if not _domain_cfg.delete_config(slug):
        raise HTTPException(status_code=404, detail=f"Domain config {slug!r} not found")
    logger.info("delete_domain_config: deleted %s", slug)


@app.post("/domain-configs/generate", status_code=200)
async def generate_domain_config(body: dict) -> dict[str, Any]:
    """
    Draft a DomainConfig by inspecting robots.txt and sitemaps.
    Body: {"start_url": "https://example.com/docs"}
    """
    start_url = body.get("start_url", "")
    if not start_url:
        raise HTTPException(status_code=422, detail="start_url is required")
    try:
        cfg = await _domain_cfg.generate_from_sitemap(start_url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return cfg.model_dump()
