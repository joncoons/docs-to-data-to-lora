"""
crawl.py — streaming-batch BFS web crawler with direct embed→ES write.

HTML pages are chunked semantically by the crawler itself (element-based,
same taxonomy as nemoretriever-parse).  Chunks are embedded directly via
the NVIDIA NIM embedding endpoint and bulk-written to Elasticsearch.
No ingestor container is involved in the HTML ingestion path.

Binary files (PDF, DOCX, etc.) discovered during crawl are written to
the binary manifest.  Phase 3 hand-off POSTs them to the ingestor via
its /documents endpoint, since those require Nemotron-Parse.

Phase 1 — BFS + rolling embed/ES dispatch
Phase 2 — Drain remaining in-flight tasks
Phase 3 — Binary ingest via ingestor (skipped when skip_phase3=True)
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import ssl
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
import requests

from . import config, es_client, embed_client, task_handler
from .chunker import html_to_elements, md_to_elements, rst_to_elements, split_by_semantic_elements, xml_to_markdown
from .product_url_map import CRAWLER_PRODUCT_URL_MAP
from .provenance import (
    compact_dict,
    hashes_equal,
    normalize_registry_entry,
    normalize_sha256_hash,
    registry_provenance_fields,
    sha256_bytes,
    stable_id,
    source_revision_id_for,
)

logger = logging.getLogger(__name__)

# ── Sentinels ────────────────────────────────────────────────────────────────


class _FetchUnchanged:
    __slots__ = ("hrefs",)
    def __init__(self, hrefs: list[str]) -> None:
        self.hrefs = hrefs


_UNCHANGED: tuple = ()
_MANIFEST: tuple = (None,)
_INLINE_CHUNKS: tuple = (None, None)  # sentinel: inline text chunks ready to embed


def _write_bytes(path: str, data: bytes) -> None:
    """Synchronous file write — called via asyncio.to_thread."""
    with open(path, "wb") as fh:
        fh.write(data)

# ── Extension sets ────────────────────────────────────────────────────────────

_NFS_DOCUMENT_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
})
_NFS_AUDIO_EXTENSIONS = frozenset({
    ".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus",
})
_NFS_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m4v",
})
_NFS_MEDIA_EXTENSIONS = _NFS_AUDIO_EXTENSIONS | _NFS_VIDEO_EXTENSIONS

_BINARY_MANIFEST_COLUMNS = [
    "source_uri", "filename", "local_path", "referring_page_url",
    "content_hash", "crawl_depth", "file_size_bytes", "content_type",
    "downloaded_at", "last_ingested_hash", "collection_name",
    "media_type", "schema_version", "provenance_schema_version",
    "ingestion_run_id", "crawl_run_id", "source_revision_id",
    "source_system", "source_kind", "modality", "final_uri",
    "raw_sha256", "source_content_hash", "retrieved_at",
    "parser_version", "chunker_version",
]

_BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls",
    ".md", ".rst", ".txt",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff",
    ".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m4v",
    ".xml",
})

# Plain-text markup files embedded inline (same chunk→embed→ES pipeline as HTML)
_INLINE_TEXT_EXTENSIONS: frozenset[str] = frozenset({".md", ".rst", ".txt"})

_BLOCKED_URL_SUFFIXES: frozenset[str] = frozenset({".inv"})

_DEFAULT_BLOCKED_PATTERNS: tuple[str, ...] = (
    "mailto:", "javascript:", "tel:", "ftp:",
    "docscontent.nvidia.com",
    "developer.download.nvidia.com/compute",
    "network.nvidia.com/pdf",
    "forums.developer.nvidia.com/uploads",
    "github.com",
    "gitlab.com",
)

# Extensions that should not be fetched as HTML pages (images, data files, scripts).
# .md/.rst/.txt are intentionally excluded — they are routed through _collect_binary_file
# as inline text and embedded directly via md_to_elements / rst_to_elements.
_RAW_TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".css", ".js", ".json", ".xml", ".csv",
})

HTML_CHUNK_MAX_TOKENS: int = 2048


def _is_binary_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if any(path.endswith(s) for s in _BLOCKED_URL_SUFFIXES):
        return False
    if any(path.endswith(ext) for ext in _RAW_TEXT_EXTENSIONS):
        return True  # treat raw text files as binary (skip HTML fetch path)
    return any(path.endswith(ext) for ext in _BINARY_EXTENSIONS)


_GITHUB_BLOB_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$",
    re.IGNORECASE,
)


def _maybe_rewrite_github_url(url: str) -> str:
    """Rewrite github.com/<org>/<repo>/blob/<branch>/<path> to its raw equivalent.

    GitHub serves the *rendered HTML* view for /blob/ URLs, not the raw file
    content.  Crawling the blob URL would yield navigation chrome instead of
    the actual README/markdown.  Rewriting to raw.githubusercontent.com gets
    the actual file content.
    """
    m = _GITHUB_BLOB_RE.match(url)
    if not m:
        return url
    org, repo, branch, path = m.groups()
    return f"https://raw.githubusercontent.com/{org}/{repo}/{branch}/{path}"


def _same_domain(url: str, netloc: str) -> bool:
    parsed = urlparse(url)
    if not parsed.netloc:
        return True
    return parsed.netloc == netloc or parsed.netloc.endswith("." + netloc)


def _sha256(data: bytes) -> str:
    return sha256_bytes(data)


def _collection_slug_for(collection_name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", collection_name.lower())


def load_binary_manifest(collection_name: str, registry_dir: str) -> list[dict]:
    slug = _collection_slug_for(collection_name)
    path = os.path.join(registry_dir, f"{slug}_binary_manifest.csv")
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception as exc:
        logger.warning("Could not load binary manifest at %s: %s", path, exc)
        return []


def save_binary_manifest(
    manifest: list[dict], collection_name: str, registry_dir: str
) -> None:
    slug = _collection_slug_for(collection_name)
    path = os.path.join(registry_dir, f"{slug}_binary_manifest.csv")
    try:
        os.makedirs(registry_dir, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=_BINARY_MANIFEST_COLUMNS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(manifest)
    except Exception as exc:
        logger.warning("Could not save binary manifest to %s: %s", path, exc)


def load_url_registry(collection_name: str, registry_dir: str) -> dict[str, dict]:
    """Load the URL registry JSON for a collection. Returns {} if not found."""
    slug = _collection_slug_for(collection_name)
    path = os.path.join(registry_dir, f"{slug}_url_registry.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for entry in data.values():
                if isinstance(entry, dict):
                    normalize_registry_entry(entry)
            return data
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Could not load URL registry from %s: %s", path, exc)
    return {}


def save_url_registry(collection_name: str, registry_dir: str, registry: dict[str, dict]) -> None:
    """Persist the URL registry JSON for a collection (atomic write)."""
    slug = _collection_slug_for(collection_name)
    path = os.path.join(registry_dir, f"{slug}_url_registry.json")
    tmp_path = path + ".tmp"
    try:
        os.makedirs(registry_dir, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(registry, fh, separators=(",", ":"))
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.warning("Could not save URL registry to %s: %s", path, exc)


def _load_known_permanent_errors(collection_name: str, registry_dir: str) -> set[str]:
    """Load URLs with permanent errors (404/410) from the error matrix CSV.

    Returns a set of URLs to skip during BFS. Only 'broken_links' category
    entries are included — transient errors (5xx, timeouts) are not pre-skipped.
    force_recrawl=True bypasses this (caller clears the set before the loop).
    """
    slug = _collection_slug_for(collection_name)
    path = os.path.join(registry_dir, f"{slug}_error_matrix.csv")
    result: set[str] = set()
    if not os.path.exists(path):
        return result
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("category") == "broken_links" and row.get("url"):
                    result.add(row["url"])
    except Exception as exc:
        logger.warning("Could not load error matrix from %s: %s", path, exc)
    return result


def _binary_unchanged(file_reg: dict, *, force_recrawl: bool) -> bool:
    """True if the binary was previously ingested at its current content hash.

    Used as a pre-download check — if True, skip the HTTP request entirely.
    ETag/If-None-Match inside _collect_binary_file remains as a secondary check.
    """
    if force_recrawl:
        return False
    ingested = file_reg.get("last_ingested_hash", "")
    stored = file_reg.get("content_hash", "")
    return hashes_equal(ingested, stored)


class SimpleWebCrawler:
    """
    Streaming-batch BFS web crawler.

    HTML chunks are embedded directly via the NIM embedding endpoint and
    bulk-written to Elasticsearch.  No ingestor is involved for HTML.
    Binary files are handed off to the ingestor's /documents endpoint (Phase 3).
    """

    _FETCH_CONCURRENCY: int = 10
    _FETCH_304 = object()

    def __init__(
        self,
        start_url: str,
        max_pages: int | None = 50,
        extract_linked_files: bool = False,
        batch_size: int = 20,           # chunks before triggering embed+ES dispatch
        binary_batch_size: int = 1,
        max_concurrent_batches: int = 4,
        force_recrawl: bool = False,
        registry_dir: str = "",
        collection_name: str = "",
        request_timeout: int = 30,
        user_agent: str = "NVIDIA-RAG-Crawler/1.0",
        html_chunk_max_tokens: int = HTML_CHUNK_MAX_TOKENS,
        export_dir: str = "",
        pdf_repo_dir: str = "",
        docs_repo_dir: str = "",
        audio_repo_dir: str = "",
        video_repo_dir: str = "",
        max_media_file_mb: int = 500,
        allowed_url_prefixes: list[str] | None = None,
        use_selenium: bool = True,
        selenium_content_threshold: int = 300,
        selenium_screenshot_fallback: bool = False,
        selenium_wait_timeout: int = 20,
        max_depth: int | None = None,
        blocked_url_patterns: list[str] | None = None,
        unblock_url_patterns: list[str] | None = None,
        binary_host_allowlist: list[str] | None = None,
        seed_urls: list[str] | None = None,
        use_sitemap: bool = False,
        use_product_url_map: bool = True,
        task_id: str | None = None,
        extra_metadata: dict | None = None,
    ) -> None:
        # Preserve trailing slash on directory-style URLs so that relative-link
        # resolution via urljoin() inside the BFS keeps relative paths under
        # the seed directory.  For file-style URLs (path's last segment
        # contains a `.`), strip any trailing slash to normalize identity.
        _parsed_seed = urlparse(start_url)
        _last_seg = _parsed_seed.path.rsplit("/", 1)[-1] if _parsed_seed.path else ""
        if "." in _last_seg:
            self.start_url = start_url.rstrip("/")
        else:
            self.start_url = start_url if start_url.endswith("/") else start_url + "/"
        self.task_id = task_id
        self.max_pages = max_pages
        self.extract_linked_files = extract_linked_files
        self.batch_size = max(1, batch_size)
        self.binary_batch_size = max(1, binary_batch_size)
        self.max_concurrent_batches = max(1, max_concurrent_batches)
        self.force_recrawl = force_recrawl
        self.registry_dir = registry_dir or config.REGISTRY_DIR
        self.collection_name = collection_name
        self.request_timeout = request_timeout
        self._html_chunk_max_tokens = html_chunk_max_tokens
        self.export_dir = export_dir or config.EXPORT_DIR
        self.pdf_repo_dir = pdf_repo_dir or config.PDF_REPO_DIR
        self.docs_repo_dir = docs_repo_dir or config.DOCS_REPO_DIR
        self.audio_repo_dir = audio_repo_dir or config.AUDIO_REPO_DIR
        self.video_repo_dir = video_repo_dir or config.VIDEO_REPO_DIR
        self.max_media_file_mb = max(1, max_media_file_mb)
        self.extra_metadata: dict = dict(extra_metadata or {})

        # Load product URL map (allow JSON file override via env)
        _map_override = config.PRODUCT_MAP_PATH
        if _map_override:
            try:
                with open(_map_override) as f:
                    self._product_url_map: list[tuple[str, str, str | None]] = [
                        (e[0], e[1], e[2] if len(e) > 2 else None) for e in json.load(f)
                    ]
            except Exception as exc:
                logger.warning("Failed to load product map from %s: %s", _map_override, exc)
                self._product_url_map = CRAWLER_PRODUCT_URL_MAP
        else:
            self._product_url_map = CRAWLER_PRODUCT_URL_MAP

        self.allowed_url_prefixes = [p.rstrip("/") for p in allowed_url_prefixes] if allowed_url_prefixes else None
        self.max_depth = max_depth
        # Compute final block list: defaults MINUS unblock list, PLUS additive blocks.
        _unblock = set(unblock_url_patterns or [])
        _defaults_after_unblock = [p for p in _DEFAULT_BLOCKED_PATTERNS if p not in _unblock]
        self.blocked_url_patterns = _defaults_after_unblock + [
            p.rstrip("/") for p in (blocked_url_patterns or [])
        ]
        # Cross-host allowlist for binary/inline-text downloads.
        self.binary_host_allowlist = [p.rstrip("/") for p in (binary_host_allowlist or [])]
        # Additional URLs to enqueue at BFS depth 0 alongside start_url.
        # Useful for docs sites whose sidebar nav is JS-rendered and not
        # exposed in initial HTML — caller seeds known section-index pages.
        self.seed_urls = list(seed_urls or [])
        self.use_sitemap = use_sitemap
        self.use_product_url_map = use_product_url_map
        self.use_selenium = use_selenium
        self._selenium_content_threshold = selenium_content_threshold
        self._selenium_screenshot_fallback = selenium_screenshot_fallback
        self._selenium_wait_timeout = selenium_wait_timeout
        self._user_agent = user_agent

        self._selenium_driver: Any = None
        self._session: aiohttp.ClientSession | None = None
        self._selenium_lock: asyncio.Lock | None = None
        self._manifest_lock: asyncio.Lock | None = None
        self._live_manifest: list[dict] = []
        self._netloc = urlparse(start_url).netloc
        # SSL context for ES (built once per crawl)
        self._es_ssl: ssl.SSLContext | None = None
        self._ingestion_run_id: str = ""
        self._ingestion_run_started_at: str = ""

    # ── URL resolution ────────────────────────────────────────────────────────

    def _resolve_product_metadata(self, url: str) -> dict:
        lower_url = url.lower()
        for prefix, family, name in self._product_url_map:
            if prefix.lower() in lower_url:
                meta: dict = {"product_family": family}
                if name is not None:
                    meta["product_name"] = name
                return meta
        return {}

    @staticmethod
    def _slugify_collection(name: str) -> str:
        slug = re.sub(r"[^a-z0-9-]", "", name.lower().replace(" ", "-")).strip("-")
        return slug or "general"

    def _resolve_collection_name(self, url: str) -> str:
        if not self.use_product_url_map:
            return self.collection_name
        lower_url = url.lower()
        for prefix, family, _name in self._product_url_map:
            if prefix.lower() in lower_url:
                return self._slugify_collection(family)
        return self.collection_name

    def _new_ingestion_run_id(self, collection_name: str, started_at: str) -> str:
        return stable_id("ingest", collection_name, self.start_url, started_at)

    def _source_provenance_metadata(
        self,
        *,
        url: str,
        source_system: str,
        source_kind: str,
        modality: str,
        content_hash: str | None,
        retrieved_at: str,
        status_code: int | None = None,
        last_modified: str | None = None,
        etag: str | None = None,
        final_uri: str | None = None,
        content_type: str | None = None,
        parser_version: str | None = None,
        chunker_version: str | None = "semantic-elements-v1",
        transforms: list[dict] | None = None,
    ) -> dict[str, Any]:
        raw_sha256 = normalize_sha256_hash(content_hash)
        source_revision_id = source_revision_id_for(url, raw_sha256, retrieved_at)
        return compact_dict({
            "source_system": source_system,
            "source_kind": source_kind,
            "modality": modality,
            "ingestion_run_id": self._ingestion_run_id,
            "crawl_run_id": self._ingestion_run_id,
            "source_revision_id": source_revision_id,
            "source_content_hash": raw_sha256,
            "raw_sha256": raw_sha256,
            "retrieved_at": retrieved_at,
            "final_uri": final_uri or url,
            "http_status_code": status_code,
            "http_last_modified": last_modified,
            "http_etag": etag,
            "content_type": content_type,
            "parser_version": parser_version,
            "chunker_version": chunker_version,
            "transforms": transforms or [],
        })

    # ── Public API ────────────────────────────────────────────────────────────

    async def crawl(self, collection_name: str) -> dict[str, Any]:
        """BFS-crawl the site and embed+store all discovered HTML content."""
        if self.task_id:
            task_handler.clear_cancel(self.task_id)
        try:
            return await self._crawl_async(collection_name)
        finally:
            pass

    # ── Core async implementation ─────────────────────────────────────────────

    async def _crawl_async(self, collection_name: str) -> dict[str, Any]:
        self._es_ssl = config.build_es_ssl_ctx()

        try:
            import certifi
            _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            _ssl_ctx = ssl.create_default_context()

        conn = aiohttp.TCPConnector(ssl=_ssl_ctx, limit=20)
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        headers = {"User-Agent": self._user_agent}

        self._selenium_lock = asyncio.Lock()
        self._manifest_lock = asyncio.Lock()

        visited_html: set[str] = set()
        visited_files: set[str] = set()
        queued_html: set[str] = {self.start_url}   # dedup at enqueue time
        queue: deque[tuple[str, int]] = deque([(self.start_url, 0)])

        # Enqueue caller-provided seed URLs at depth 0 alongside start_url.
        # Each seed must pass the same gates as a BFS-discovered URL: domain
        # allowlist, prefix filter, block list, and binary/text classification.
        for _seed in self.seed_urls:
            _seed_norm = _maybe_rewrite_github_url(_seed)
            if _seed_norm in queued_html:
                continue
            if self._is_blocked_url(_seed_norm):
                logger.info("seed_urls: blocked, skipping %s", _seed_norm)
                continue
            if not self._link_followable(_seed_norm):
                logger.info("seed_urls: not link-followable, skipping %s", _seed_norm)
                continue
            if self.allowed_url_prefixes and not self._is_allowed_url(_seed_norm):
                logger.info("seed_urls: outside allowed_url_prefixes, skipping %s", _seed_norm)
                continue
            queue.append((_seed_norm, 0))
            queued_html.add(_seed_norm)
        if self.seed_urls:
            logger.info("seed_urls: enqueued %d additional seeds at depth 0", len(queued_html) - 1)

        errors: list[dict] = []
        error_matrix: dict[str, list[dict]] = {
            "broken_links": [], "missing_files": [],
            "ingest_failures": [], "batch_errors": [],
        }
        pages_crawled = 0
        pages_skipped = 0
        pages_contentless = 0
        files_skipped = 0
        total_chunks_dispatched = 0
        chunks_written = 0
        binary_files_manifest = 0
        binary_files_skipped = 0

        changed_urls: set[str] = set()
        deleted_urls: set[str] = set()
        redirected_urls: set[str] = set()

        # pending: (chunk_text, metadata, target_collection) — no files
        pending: list[tuple[str, dict, str]] = []
        created_collections: set[str] = set()

        # in_flight: list of asyncio.Tasks
        in_flight: list[asyncio.Task] = []
        batch_num = 0

        registry: dict[str, dict] = self._load_registry()
        self._registry_last_saved = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()
        self._ingestion_run_started_at = now_iso
        self._ingestion_run_id = self._new_ingestion_run_id(collection_name, now_iso)

        known_errors: set[str] = set()
        if not self.force_recrawl:
            coll_for_errors = self.collection_name or self._collection_slug()
            known_errors = _load_known_permanent_errors(coll_for_errors, self.registry_dir)
            if known_errors:
                logger.info("Loaded %d known-broken URLs from error matrix — will skip", len(known_errors))

        # Semaphore is per-batch (created inside _embed_and_store) to avoid
        # cross-batch semaphore starvation when one batch hangs.

        # ── Inner helpers ─────────────────────────────────────────────────────

        def _harvest_done() -> None:
            nonlocal chunks_written
            remaining: list[asyncio.Task] = []
            for task in in_flight:
                if not task.done():
                    remaining.append(task)
                    continue
                try:
                    written = task.result()
                    chunks_written += written or 0
                except Exception as exc:
                    logger.error("Embed/ES batch failed: %r", exc)
                    errors.append({
                        "url": "batch_error",
                        "error_type": "batch_error",
                        "status_code": None,
                        "error": repr(exc),
                    })
            in_flight[:] = remaining

        async def _embed_and_store(
            batch: list[tuple[str, dict, str]],
            bnum: int,
        ) -> int:
            """Embed all chunks in *batch* and bulk-write to ES, grouped by collection."""
            if not batch:
                return 0

            groups: dict[str, list[tuple[str, dict]]] = {}
            for text, meta, coll in batch:
                groups.setdefault(coll, []).append((text, meta))

            total = 0

            # Per-batch semaphore — prevents a hung batch from blocking other batches
            batch_embed_semaphore = asyncio.Semaphore(config.EMBED_CONCURRENCY)

            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=False),
                timeout=aiohttp.ClientTimeout(total=120),
            ) as embed_session:
                async with aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(ssl=self._es_ssl or False),
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as es_session:
                    for coll, items in groups.items():
                        texts = [t for t, _ in items]
                        metas = [m for _, m in items]

                        # Ensure index exists (no-op if already created)
                        if coll not in created_collections:
                            await es_client.ensure_index(coll, es_session, self._es_ssl)
                            created_collections.add(coll)

                        # Delete stale chunks for changed URLs in this collection
                        uris_to_delete = [
                            m["source_uri"] for m in metas
                            if m.get("source_uri") in changed_urls
                        ]
                        if uris_to_delete:
                            await es_client.delete_by_source_uris(
                                coll, list(set(uris_to_delete)), es_session, self._es_ssl
                            )

                        # Embed
                        vectors = await embed_client.embed_texts(
                            texts, embed_session, batch_embed_semaphore
                        )

                        # Bulk write
                        written = await es_client.bulk_write(
                            coll, list(zip(texts, metas)), vectors,
                            es_session, self._es_ssl,
                        )
                        logger.info(
                            "Batch %d [%s]: %d/%d chunks written to ES",
                            bnum, coll, written, len(items),
                        )
                        total += written

            return total

        async def _dispatch_batch_async(batch: list[tuple[str, dict, str]]) -> None:
            nonlocal batch_num, total_chunks_dispatched

            if not batch:
                return

            if len(in_flight) >= self.max_concurrent_batches:
                await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                _harvest_done()

            batch_num += 1
            total_chunks_dispatched += len(batch)

            logger.info(
                "Dispatching embed batch %d: %d chunks  [%d/%d in-flight]",
                batch_num, len(batch), len(in_flight), self.max_concurrent_batches,
            )

            task = asyncio.create_task(
                asyncio.wait_for(_embed_and_store(list(batch), batch_num), timeout=300)
            )
            in_flight.append(task)

            self._save_registry(registry)
            self._flush_errors_to_csv(errors, error_matrix)
            self._export_crawl_artifacts()

        # ── Open HTTP session ─────────────────────────────────────────────────
        async with aiohttp.ClientSession(
            connector=conn, timeout=timeout, headers=headers,
        ) as session:
            self._session = session
            max_pages_display = self.max_pages if self.max_pages is not None else "unlimited"

            if self.use_sitemap:
                seed_urls = await asyncio.to_thread(self._fetch_sitemap_seeds)
                for seed_url in seed_urls:
                    if seed_url not in queued_html:
                        queue.append((seed_url, 0))
                        queued_html.add(seed_url)
                logger.info("sitemap: seeded BFS with %d URLs", len(seed_urls))

            logger.info(
                "Crawl starting: %s (max_pages=%s, batch=%d, max_depth=%s, "
                "registry=%d entries)",
                self.start_url, max_pages_display, self.batch_size,
                self.max_depth, len(registry),
            )

            # Ensure the fallback collection index exists upfront
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=self._es_ssl or False),
            ) as _es_sess:
                await es_client.ensure_index(collection_name, _es_sess, self._es_ssl)
            created_collections.add(collection_name)

            if self.task_id:
                task_handler.set_progress(self.task_id, {
                    "task_type": "crawl",
                    "start_url": self.start_url,
                    "collection_name": self.collection_name,
                    "pages_crawled": 0,
                    "pages_queued": len(queue),
                    "pages_skipped": 0,
                    "chunks_dispatched": 0,
                })

            if self.use_selenium:
                # Run Selenium init in a dedicated daemon thread (NOT the asyncio thread pool).
                # asyncio.to_thread() puts work into the shared default ThreadPoolExecutor;
                # if the init hangs, wait_for() cancels the coroutine wrapper but the thread
                # keeps running and holds a pool slot, starving all subsequent page fetches.
                # A daemon thread + asyncio.Future avoids that entirely.
                loop = asyncio.get_running_loop()
                _sel_future: asyncio.Future = loop.create_future()

                def _init_driver_thread() -> None:
                    driver = self._create_selenium_driver()
                    loop.call_soon_threadsafe(
                        lambda: _sel_future.set_result(driver)
                        if not _sel_future.done() else None
                    )

                threading.Thread(
                    target=_init_driver_thread,
                    daemon=True,
                    name="selenium-init",
                ).start()

                try:
                    self._selenium_driver = await asyncio.wait_for(_sel_future, timeout=30)
                except asyncio.TimeoutError:
                    logger.warning("Selenium driver init timed out (30s) — proceeding without Selenium")
                    self._selenium_driver = None

            self._live_manifest = self._load_binary_manifest()

            try:
                # ── Phase 1: BFS ──────────────────────────────────────────────
                while queue and (self.max_pages is None or pages_crawled < self.max_pages):
                    if self.task_id and task_handler.is_cancel_requested(self.task_id):
                        logger.info("Crawl cancelled after %d pages", pages_crawled)
                        break

                    to_fetch: list[tuple[str, int]] = []
                    while queue and len(to_fetch) < self._FETCH_CONCURRENCY:
                        url, depth = queue.popleft()
                        if url in visited_html:
                            continue
                        visited_html.add(url)
                        to_fetch.append((url, depth))

                    if not to_fetch:
                        break

                    _harvest_done()

                    fetch_results = await asyncio.gather(
                        *[self._fetch_html(url, reg_entry=registry.get(url)) for url, _ in to_fetch],
                        return_exceptions=True,
                    )

                    binary_fetch_args: list[tuple] = []

                    for (url, depth), result in zip(to_fetch, fetch_results):
                        if isinstance(result, Exception):
                            logger.warning("Fetch exception %s: %r", url, result)
                            errors.append({"url": url, "error_type": "broken_link",
                                           "status_code": None, "error": repr(result)})
                            continue

                        if result is self._FETCH_304:
                            pages_crawled += 1
                            pages_skipped += 1
                            registry[url] = {**registry.get(url, {}), "last_seen": now_iso}
                            continue

                        if isinstance(result, _FetchUnchanged):
                            pages_crawled += 1
                            pages_skipped += 1
                            registry[url] = {**registry.get(url, {}), "last_seen": now_iso}
                            for href in result.hrefs:
                                href = href.split("#")[0]
                                if not href:
                                    continue
                                try:
                                    abs_href = urljoin(url, href)
                                except ValueError:
                                    continue
                                # Rewrite GitHub /blob/ URLs to their raw equivalent before
                                # any predicate checks, so the URL we record / fetch is the
                                # one that returns actual file content.
                                abs_href = _maybe_rewrite_github_url(abs_href)
                                if self._is_blocked_url(abs_href):
                                    continue
                                if self.extract_linked_files and _is_binary_url(abs_href):
                                    if abs_href not in visited_files and abs_href not in known_errors:
                                        visited_files.add(abs_href)
                                        file_reg = registry.get(abs_href, {})
                                        if _binary_unchanged(file_reg, force_recrawl=self.force_recrawl):
                                            files_skipped += 1
                                            registry[abs_href] = {**file_reg, "last_seen": now_iso}
                                        else:
                                            binary_fetch_args.append((
                                                abs_href, depth + 1, file_reg,
                                                file_reg.get("etag") if not self.force_recrawl else None,
                                                file_reg.get("last_modified") if not self.force_recrawl else None,
                                                url,
                                            ))
                                elif (
                                    not _is_binary_url(abs_href)
                                    and self._link_followable(abs_href)
                                    and self._is_allowed_url(abs_href)
                                    and not self._is_blocked_url(abs_href)
                                    and abs_href not in visited_html
                                    and abs_href not in queued_html
                                    and abs_href not in known_errors
                                    and (self.max_pages is None or pages_crawled < self.max_pages)
                                    and (self.max_depth is None or depth + 1 <= self.max_depth)
                                ):
                                    queue.append((abs_href, depth + 1))
                                    queued_html.add(abs_href)
                            continue

                        html_content, page_title, meta_desc, section_h1, linked_urls, fetch_error, resp_meta = result

                        if html_content is None:
                            if fetch_error:
                                errors.append({"url": url, **fetch_error})
                                if fetch_error.get("status_code") in (404, 410):
                                    reg_entry = registry.get(url, {})
                                    if reg_entry.get("last_ingested"):
                                        deleted_urls.add(url)
                            continue

                        pages_crawled += 1

                        final_url = resp_meta.pop("final_url", None)
                        if final_url:
                            final_url = _maybe_rewrite_github_url(final_url)
                            if (
                                not _is_binary_url(final_url)
                                and self._link_followable(final_url)
                                and self._is_allowed_url(final_url)
                                and not self._is_blocked_url(final_url)
                                and final_url not in visited_html
                                and final_url not in queued_html
                                and (self.max_depth is None or depth + 1 <= self.max_depth)
                            ):
                                queue.appendleft((final_url, depth))
                                queued_html.add(final_url)
                            reg_entry = registry.get(url, {})
                            if reg_entry.get("last_ingested"):
                                redirected_urls.add(url)
                            registry[url] = {
                                "redirect_to": final_url,
                                "last_seen": now_iso,
                                "status_code": resp_meta.get("status_code", 301),
                            }
                            continue

                        if self.task_id and pages_crawled % 5 == 0:
                            task_handler.set_progress(self.task_id, {
                                "task_type": "crawl",
                                "start_url": self.start_url,
                                "collection_name": self.collection_name,
                                "pages_crawled": pages_crawled,
                                "pages_queued": len(queue) + len(in_flight),
                                "pages_skipped": pages_skipped,
                                "chunks_dispatched": total_chunks_dispatched,
                            })

                        new_hash = resp_meta.get("content_hash", "")
                        reg_entry = registry.get(url, {})
                        stored_hash = reg_entry.get("content_hash", "")

                        if not self.force_recrawl and hashes_equal(stored_hash, new_hash):
                            pages_skipped += 1
                            registry[url] = {**reg_entry, "last_seen": now_iso}
                        else:
                            is_changed = bool(
                                reg_entry.get("last_ingested") and (
                                    self.force_recrawl or (stored_hash and not hashes_equal(stored_hash, new_hash))
                                )
                            )
                            if is_changed:
                                changed_urls.add(url)

                            target_coll = self._resolve_collection_name(url)
                            provenance_meta = self._source_provenance_metadata(
                                url=url,
                                source_system="web_crawl",
                                source_kind="web_page",
                                modality="text",
                                content_hash=new_hash,
                                retrieved_at=now_iso,
                                status_code=resp_meta.get("status_code"),
                                last_modified=resp_meta.get("last_modified"),
                                etag=resp_meta.get("etag"),
                                final_uri=resp_meta.get("final_url") or url,
                                content_type=resp_meta.get("content_type"),
                                parser_version="html-to-elements-v1",
                            )
                            base_meta = {
                                **self._resolve_product_metadata(url),
                                **self.extra_metadata,
                                **provenance_meta,
                                "source_uri": url,
                                "filename": url.rstrip("/").rsplit("/", 1)[-1] or self._netloc,
                                "page_title": page_title,
                                "crawl_depth": depth,
                                "section_h1": section_h1,
                                "meta_description": meta_desc,
                                "document_type": "html",
                            }

                            html_elements = html_to_elements(html_content)
                            if html_elements:
                                chunk_pairs = split_by_semantic_elements(
                                    html_elements, self._html_chunk_max_tokens,
                                    chunk_overlap=150,
                                )
                                for chunk_text, section_path in chunk_pairs:
                                    if not chunk_text.strip():
                                        continue
                                    meta = {**base_meta}
                                    if section_path:
                                        meta["section_path"] = section_path
                                        meta["heading"] = section_path.split(" > ")[-1].strip()
                                    pending.append((chunk_text, meta, target_coll))
                            else:
                                screenshot_added = False
                                if self.use_selenium and self._selenium_screenshot_fallback:
                                    ss_dir = tempfile.gettempdir()
                                    ss_path = await asyncio.to_thread(
                                        self._capture_screenshot, url, ss_dir
                                    )
                                    if ss_path:
                                        # Screenshots go to ingestor for VLM; record in manifest
                                        ss_bytes = open(ss_path, "rb").read()
                                        ss_hash = _sha256(ss_bytes)
                                        ss_downloaded_at = datetime.now(UTC).isoformat()
                                        ss_provenance = self._source_provenance_metadata(
                                            url=url,
                                            source_system="web_crawl",
                                            source_kind="rendered_screenshot",
                                            modality="image",
                                            content_hash=ss_hash,
                                            retrieved_at=ss_downloaded_at,
                                            status_code=resp_meta.get("status_code"),
                                            last_modified=resp_meta.get("last_modified"),
                                            etag=resp_meta.get("etag"),
                                            final_uri=url,
                                            content_type="image/jpeg",
                                            parser_version="selenium-screenshot-v1",
                                            chunker_version=None,
                                        )
                                        async with self._manifest_lock:
                                            self._live_manifest.append({
                                                "source_uri": url,
                                                "filename": os.path.basename(ss_path),
                                                "local_path": ss_path,
                                                "referring_page_url": url,
                                                "content_hash": ss_hash,
                                                "crawl_depth": depth,
                                                "file_size_bytes": os.path.getsize(ss_path),
                                                "content_type": "image/jpeg",
                                                "downloaded_at": ss_downloaded_at,
                                                "last_ingested_hash": "",
                                                "collection_name": self.collection_name,
                                                "media_type": "image",
                                                **registry_provenance_fields(ss_provenance),
                                            })
                                        screenshot_added = True
                                if not screenshot_added:
                                    logger.debug("Skipping content-less page: %s", url)
                                    pages_contentless += 1

                            reg_update = {
                                "last_seen": now_iso,
                                "last_modified": resp_meta.get("last_modified"),
                                "etag": resp_meta.get("etag"),
                                "content_hash": new_hash,
                                "status_code": resp_meta.get("status_code", 200),
                                "linked_hrefs": linked_urls,
                                "collection": target_coll,
                                **registry_provenance_fields(base_meta),
                            }
                            if not is_changed:
                                reg_update["last_ingested"] = now_iso
                            registry[url] = reg_update
                            logger.info(
                                "Collected page %d/%s: %s  [pending=%d chunks, in_flight=%d]",
                                pages_crawled, max_pages_display, url, len(pending), len(in_flight),
                            )

                        # Binary file links
                        for href in linked_urls:
                            href = href.split("#")[0]
                            if not href:
                                continue
                            try:
                                abs_href = urljoin(url, href)
                            except ValueError:
                                continue
                            abs_href = _maybe_rewrite_github_url(abs_href)
                            if self._is_blocked_url(abs_href):
                                continue
                            if self.extract_linked_files and _is_binary_url(abs_href):
                                if abs_href not in visited_files and abs_href not in known_errors:
                                    visited_files.add(abs_href)
                                    file_reg = registry.get(abs_href, {})
                                    if _binary_unchanged(file_reg, force_recrawl=self.force_recrawl):
                                        files_skipped += 1
                                        registry[abs_href] = {**file_reg, "last_seen": now_iso}
                                    else:
                                        binary_fetch_args.append((
                                            abs_href, depth + 1, file_reg,
                                            file_reg.get("etag") if not self.force_recrawl else None,
                                            file_reg.get("last_modified") if not self.force_recrawl else None,
                                            url,
                                        ))
                            elif (
                                not _is_binary_url(abs_href)
                                and self._link_followable(abs_href)
                                and self._is_allowed_url(abs_href)
                                and not self._is_blocked_url(abs_href)
                                and abs_href not in visited_html
                                and abs_href not in queued_html
                                and abs_href not in known_errors
                                and (self.max_pages is None or pages_crawled < self.max_pages)
                                and (self.max_depth is None or depth + 1 <= self.max_depth)
                            ):
                                queue.append((abs_href, depth + 1))
                                queued_html.add(abs_href)

                    # Binary downloads
                    if binary_fetch_args:
                        binary_results = await asyncio.gather(
                            *[
                                self._collect_binary_file(
                                    abs_href, d, errors,
                                    referring_page_url=ref_url,
                                    if_none_match=ifnm,
                                    if_modified_since=ifms,
                                )
                                for abs_href, d, file_reg, ifnm, ifms, ref_url in binary_fetch_args
                            ],
                            return_exceptions=True,
                        )
                        for (abs_href, d, file_reg, ifnm, ifms, ref_url), br in zip(binary_fetch_args, binary_results):
                            if isinstance(br, Exception):
                                logger.warning("Binary fetch exception %s: %r", abs_href, br)
                                continue
                            entry, file_meta = br
                            if entry is _INLINE_CHUNKS:
                                # Plain-text markup — add chunks directly to pending
                                inline_chunks = file_meta.get("chunks", [])
                                pending.extend(inline_chunks)
                                registry[abs_href] = {
                                    "last_seen": now_iso,
                                    "last_modified": file_meta.get("last_modified"),
                                    "etag": file_meta.get("etag"),
                                    "content_hash": file_meta.get("content_hash", ""),
                                    "status_code": file_meta.get("status_code", 200),
                                    "collection": self._resolve_collection_name(ref_url or abs_href),
                                    "last_ingested": now_iso,
                                    **registry_provenance_fields(file_meta),
                                }
                            elif entry is _UNCHANGED:
                                files_skipped += 1
                                registry[abs_href] = {**file_reg, "last_seen": now_iso}
                            elif entry is None:
                                if file_meta.get("status_code") in (404, 410) and file_reg.get("last_ingested"):
                                    deleted_urls.add(abs_href)
                            elif entry is _MANIFEST:
                                # Binary saved to NFS — queued for explicit Phase 3 ingest
                                binary_files_manifest += 1
                                registry[abs_href] = {
                                    "last_seen": now_iso,
                                    "last_modified": file_meta.get("last_modified"),
                                    "etag": file_meta.get("etag"),
                                    "content_hash": file_meta.get("content_hash"),
                                    "status_code": file_meta.get("status_code", 200),
                                    "collection": self._resolve_collection_name(ref_url or abs_href),
                                    **registry_provenance_fields(file_meta),
                                }

                    # Dispatch when threshold reached
                    if len(pending) >= self.batch_size:
                        await _dispatch_batch_async(pending)
                        pending = []

                # ── Phase 2: drain ────────────────────────────────────────────
                await _dispatch_batch_async(pending)
                pending = []

                if in_flight:
                    drain_deadline = asyncio.get_event_loop().time() + 2100
                    while in_flight:
                        if asyncio.get_event_loop().time() > drain_deadline:
                            logger.warning("Drain deadline exceeded — cancelling %d tasks", len(in_flight))
                            for t in in_flight:
                                t.cancel()
                            in_flight.clear()
                            break
                        done_tasks = [t for t in in_flight if t.done()]
                        if done_tasks:
                            _harvest_done()
                        else:
                            await asyncio.sleep(2)
                            _harvest_done()

                # ── Phase 3: binary ingest via ingestor ───────────────────────
                # Binaries are now POSTed inline during Phase 1 fetch, so there
                # is nothing to do here.  The deferred disk-cache path is
                # commented out below for easy restoration if needed.
                binary_files_skipped = 0
                # if self.skip_phase3 or (self.task_id and task_handler.is_cancel_requested(self.task_id)):
                #     logger.info("Phase 3 binary ingest skipped.")
                #     binary_files_ingested = 0
                #     binary_files_skipped = 0
                # else:
                #     binary_files_ingested, binary_files_skipped = await self._run_phase3(
                #         session, changed_urls, created_collections,
                #     )

            finally:
                self._session = None
                self._selenium_lock = None
                self._manifest_lock = None

                if self._selenium_driver is not None:
                    try:
                        self._selenium_driver.quit()
                    except Exception:
                        pass
                    self._selenium_driver = None

                # Purge redirected / deleted URLs from ES
                if redirected_urls or deleted_urls:
                    async with aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(ssl=self._es_ssl or False),
                    ) as _es_sess:
                        for uri_set, label in ((redirected_urls, "redirected"), (deleted_urls, "deleted")):
                            if not uri_set:
                                continue
                            _by_coll: dict[str, list[str]] = {}
                            for _uri in uri_set:
                                _by_coll.setdefault(self._resolve_collection_name(_uri), []).append(_uri)
                            for _coll, _uris in _by_coll.items():
                                await es_client.delete_by_source_uris(
                                    _coll, _uris, _es_sess, self._es_ssl
                                )
                            if label == "deleted":
                                for uri in deleted_urls:
                                    registry.pop(uri, None)

                self._save_registry(registry, force=True)
                self._save_binary_manifest(self._live_manifest)
                self._flush_errors_to_csv(errors, error_matrix)
                self._export_crawl_artifacts()

        html_pages_with_content = pages_crawled - pages_skipped - pages_contentless

        return {
            "message": (
                f"Crawl complete: {html_pages_with_content} HTML pages "
                f"({chunks_written} chunks), {binary_files_manifest} binary files "
                f"saved to NFS (pending Phase 3 ingest) "
                f"({pages_skipped} pages unchanged/skipped, "
                f"{pages_contentless} content-less, "
                f"{files_skipped + binary_files_skipped} files skipped)."
            ),
            "task_type": "crawl",
            "start_url": self.start_url,
            "collection_name": self.collection_name,
            "pages_crawled": pages_crawled,
            "pages_skipped": pages_skipped,
            "pages_contentless": pages_contentless,
            "chunks_written": chunks_written,
            "files_saved": binary_files_manifest,
            "files_skipped": files_skipped + binary_files_skipped,
            "errors": errors,
            "error_matrix": error_matrix,
        }

    # ── Phase 3: binary hand-off to ingestor ──────────────────────────────────

    async def _run_phase3(
        self,
        session: aiohttp.ClientSession,
        changed_urls: set[str],
        created_collections: set[str],
    ) -> tuple[int, int]:
        """Deferred binary ingest via ingestor (currently disabled — binaries are
        POSTed inline during Phase 1).  Restore by uncommenting below and
        re-enabling the _run_phase3 call in _crawl_async."""
        return 0, 0

        # manifest = self._live_manifest
        # pending_docs = [
        #     r for r in manifest
        #     if r.get("media_type", "document") in ("document", "inline")
        #     and r.get("local_path")
        #     and os.path.exists(r["local_path"])
        #     and r.get("content_hash") != r.get("last_ingested_hash", "")
        # ]
        # skipped = len(manifest) - len(pending_docs)
        # if not pending_docs:
        #     return 0, skipped
        # logger.info("Phase 3: %d binary files to ingest via ingestor", len(pending_docs))
        # ingested = 0
        # for row in pending_docs:
        #     coll = self._resolve_collection_name(
        #         row.get("referring_page_url") or row["source_uri"]
        #     )
        #     metadata = {
        #         **self._resolve_product_metadata(row.get("referring_page_url") or row["source_uri"]),
        #         **self.extra_metadata,
        #         "source_uri": row["source_uri"],
        #         "filename": row["filename"],
        #         "referring_page_url": row.get("referring_page_url", ""),
        #         "crawl_depth": int(row.get("crawl_depth", 0)),
        #         "document_type": Path(row["filename"]).suffix.lstrip(".").lower(),
        #         "source_system": "web_crawl",
        #     }
        #     try:
        #         data = aiohttp.FormData()
        #         data.add_field("data", json.dumps({
        #             "collection_name": coll,
        #             "use_nemotron_parse": True,
        #             "force_nemotron_parse": False,
        #             "custom_metadata": [{"filename": row["filename"], "metadata": metadata}],
        #         }))
        #         with open(row["local_path"], "rb") as fh:
        #             data.add_field(
        #                 "documents", fh.read(),
        #                 filename=row["filename"],
        #                 content_type="application/octet-stream",
        #             )
        #         async with session.post(
        #             f"{config.INGESTOR_URL}/documents",
        #             data=data,
        #             timeout=aiohttp.ClientTimeout(total=300),
        #         ) as resp:
        #             resp.raise_for_status()
        #             row["last_ingested_hash"] = row["content_hash"]
        #             ingested += 1
        #     except Exception as exc:
        #         logger.warning("Phase 3 failed for %s: %r", row["source_uri"], exc)
        # self._save_binary_manifest(manifest)
        # logger.info("Phase 3 complete: %d ingested, %d skipped", ingested, skipped)
        # return ingested, skipped

    # ── HTML fetch ────────────────────────────────────────────────────────────

    async def _fetch_html(
        self, url: str, reg_entry: dict | None = None
    ) -> tuple:
        empty_meta: dict = {}
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 not installed")
            return None, "", "", "", [], {"error_type": "broken_link", "status_code": None,
                                         "error": "beautifulsoup4 not installed"}, empty_meta

        session = self._session
        if session is None:
            return None, "", "", "", [], {"error_type": "broken_link", "status_code": None,
                                         "error": "session not initialised"}, empty_meta

        req_headers: dict[str, str] = {}
        if reg_entry and not self.force_recrawl:
            if reg_entry.get("etag"):
                req_headers["If-None-Match"] = reg_entry["etag"]
            elif reg_entry.get("last_modified"):
                req_headers["If-Modified-Since"] = reg_entry["last_modified"]

        try:
            async with session.get(url, allow_redirects=True, headers=req_headers) as resp:
                if resp.status == 304:
                    return self._FETCH_304
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return None, "", "", "", [], None, empty_meta
                html_text = await resp.text()
                new_hash = _sha256(html_text.encode("utf-8"))
                resp_meta: dict = {
                    "last_modified": resp.headers.get("Last-Modified"),
                    "etag": resp.headers.get("ETag"),
                    "content_hash": new_hash,
                    "status_code": resp.status,
                    "content_type": content_type,
                }
                final_url = str(resp.url)
                if final_url and final_url.rstrip("/") != url.rstrip("/"):
                    resp_meta["final_url"] = final_url
                if (
                    not self.force_recrawl
                    and reg_entry
                    and hashes_equal(reg_entry.get("content_hash"), new_hash)
                    and reg_entry.get("linked_hrefs") is not None
                ):
                    return _FetchUnchanged(hrefs=reg_entry["linked_hrefs"])
        except aiohttp.ClientResponseError as exc:
            return None, "", "", "", [], {"error_type": "broken_link", "status_code": exc.status,
                                         "error": str(exc)}, empty_meta
        except Exception as exc:
            return None, "", "", "", [], {"error_type": "broken_link", "status_code": None,
                                         "error": str(exc)}, empty_meta

        if self.use_selenium and self._is_js_sparse(html_text):
            async with self._selenium_lock:
                try:
                    rendered_html = await asyncio.wait_for(
                        asyncio.to_thread(self._render_with_selenium, url),
                        timeout=self._selenium_wait_timeout + 15,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Selenium hard timeout for %s — skipping render", url)
                    rendered_html = None
            if rendered_html:
                html_text = rendered_html
                resp_meta["content_hash"] = _sha256(html_text.encode("utf-8"))

        try:
            soup = BeautifulSoup(html_text, "html.parser")
            title = (soup.title.string or "").strip() if soup.title else ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            meta_desc = str(meta_tag.get("content", "")).strip() if meta_tag else ""
            h1_tag = soup.find("h1")
            section_h1 = h1_tag.get_text(strip=True) if h1_tag else ""
            hrefs = [
                str(a.get("href", ""))
                for a in soup.find_all("a", href=True)
                if a.get("href")
            ]
        except Exception as exc:
            return html_text, "", "", "", [], {"error_type": "broken_link", "status_code": None,
                                               "error": f"parse error: {exc}"}, resp_meta

        return html_text, title, meta_desc, section_h1, hrefs, None, resp_meta

    # ── Selenium ──────────────────────────────────────────────────────────────

    def _create_selenium_driver(self) -> Any:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
        except ImportError:
            logger.warning("selenium not installed; use_selenium has no effect")
            return None

        # Prefer Chromium (installed in image); fall back through known paths.
        for candidate in (
            os.environ.get("CHROMIUM_BIN", ""),
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/google-chrome",
        ):
            if candidate and os.path.exists(candidate):
                chromium_bin = candidate
                break
        else:
            logger.warning("No Chrome/Chromium binary found; Selenium disabled")
            return None

        # Use the system chromedriver directly — skip selenium-manager network lookup.
        chromedriver = None
        for cd in ("/usr/bin/chromedriver", "/usr/local/bin/chromedriver"):
            if os.path.exists(cd):
                chromedriver = cd
                break

        user_data_dir = tempfile.mkdtemp(prefix="crawler-chrome-")
        opts = Options()
        opts.binary_location = chromium_bin
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument(f"--user-data-dir={user_data_dir}")
        opts.add_argument(f"--user-agent={self._user_agent}")

        try:
            service = Service(executable_path=chromedriver) if chromedriver else Service()
            driver = webdriver.Chrome(service=service, options=opts)
            driver.implicitly_wait(2)
            driver.set_page_load_timeout(self._selenium_wait_timeout + 10)
            driver.set_script_timeout(self._selenium_wait_timeout)

            # Inject fetch/XHR request tracker via CDP — runs before every page load.
            # This lets us detect network idle (all async data fetches complete) for SPAs.
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    window.__pendingRequests = 0;
                    window.__requestCount = 0;
                    const _origFetch = window.fetch;
                    window.fetch = function(...args) {
                        window.__pendingRequests++;
                        window.__requestCount++;
                        return _origFetch.apply(this, args).finally(
                            () => { window.__pendingRequests = Math.max(0, window.__pendingRequests - 1); }
                        );
                    };
                    const _origOpen = XMLHttpRequest.prototype.open;
                    XMLHttpRequest.prototype.open = function(...args) {
                        this.addEventListener('loadend', () => {
                            window.__pendingRequests = Math.max(0, window.__pendingRequests - 1);
                        });
                        window.__pendingRequests++;
                        window.__requestCount++;
                        return _origOpen.apply(this, args);
                    };
                """
            })

            logger.info("Selenium driver initialised (%s, driver=%s)", chromium_bin, chromedriver)
            return driver
        except Exception as exc:
            logger.warning("Selenium driver failed to start: %s", exc)
            return None

    def _is_js_sparse(self, html_text: str) -> bool:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, "html.parser")
            if soup.body:
                if bool(soup.body.find_all(["script", "noscript"])) and len(soup.body.find_all(True)) <= 4:
                    return True
            for tag in soup(["script", "style", "noscript", "head"]):
                tag.decompose()
            return len(soup.get_text(separator=" ", strip=True)) < self._selenium_content_threshold
        except Exception:
            return False

    def _render_with_selenium(self, url: str) -> str | None:
        driver = self._selenium_driver
        if driver is None:
            return None
        try:
            driver.get(url)

            # Phase 1: wait for network idle — all async fetch/XHR requests settle.
            # The CDP-injected tracker sets __pendingRequests; we poll until it hits 0
            # AND at least one request has fired (confirming the SPA bootstrapped).
            deadline = time.monotonic() + self._selenium_wait_timeout
            while time.monotonic() < deadline:
                pending = driver.execute_script("return window.__pendingRequests ?? -1")
                count = driver.execute_script("return window.__requestCount ?? 0")
                if count > 0 and pending == 0:
                    break
                time.sleep(0.25)

            # Phase 2: JS content extraction — pull innerHTML from the main content
            # element directly from the live DOM, bypassing page_source serialisation.
            # Falls back through progressively broader selectors to document.body.
            content_html = driver.execute_script("""
                const selectors = [
                    'main', 'article', '[role="main"]',
                    '.rst-content', '.doc-content', '.document',
                    '.content', '#content', '.container'
                ];
                for (const sel of selectors) {
                    const el = document.querySelector(sel);
                    if (el && el.innerText.trim().length > 200) {
                        return el.innerHTML;
                    }
                }
                return document.body.innerHTML;
            """)

            if content_html:
                logger.debug("Selenium rendered %s (%d chars)", url, len(content_html))
                return content_html
            return driver.page_source

        except Exception as exc:
            logger.warning("Selenium render error %s: %s", url, exc)
            return None

    def _capture_screenshot(self, url: str, dest_dir: str) -> str | None:
        driver = self._selenium_driver
        if driver is None:
            return None
        try:
            from selenium.webdriver.support.ui import WebDriverWait
            driver.get(url)
            try:
                WebDriverWait(driver, self._selenium_wait_timeout).until(
                    lambda d: len(d.find_element("tag name", "body").text.strip()) > 50
                )
            except Exception:
                pass
            width = driver.execute_script(
                "return Math.max(document.body.scrollWidth, document.documentElement.scrollWidth);"
            )
            height = driver.execute_script(
                "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);"
            )
            driver.set_window_size(width, min(height, 16000))
            time.sleep(2)
            png_data = driver.get_screenshot_as_png()
        except Exception as exc:
            logger.warning("Screenshot capture error %s: %s", url, exc)
            return None

        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(png_data)).convert("RGB")
            safe_name = hashlib.sha256(url.encode()).hexdigest()[:16]
            jpg_path = os.path.join(dest_dir, f"screenshot_{safe_name}.jpg")
            img.save(jpg_path, "JPEG", quality=85)
            return jpg_path
        except Exception as exc:
            logger.warning("Screenshot JPEG conversion failed %s: %s", url, exc)
            return None

    # ── Binary file collection ────────────────────────────────────────────────

    async def _collect_binary_file(
        self,
        url: str,
        depth: int,
        errors: list[dict],
        referring_page_url: str = "",
        if_none_match: str | None = None,
        if_modified_since: str | None = None,
    ) -> tuple:
        suffix = Path(urlparse(url).path).suffix or ".bin"
        empty_meta: dict = {}
        req_headers: dict[str, str] = {}
        if if_none_match:
            req_headers["If-None-Match"] = if_none_match
        elif if_modified_since:
            req_headers["If-Modified-Since"] = if_modified_since

        session = self._session
        if session is None:
            return None, empty_meta

        try:
            async with session.get(url, headers=req_headers) as resp:
                if resp.status == 304:
                    return _UNCHANGED, empty_meta
                resp.raise_for_status()

                suffix_lower = suffix.lower()

                # ── Inline plain-text markup (Markdown / RST / TXT) ──────────
                # Fetch, parse into elements, chunk, and embed inline — same
                # pipeline as HTML pages.  No NFS write; no Phase 3 hand-off.
                if suffix_lower in _INLINE_TEXT_EXTENSIONS:
                    raw_bytes = await resp.read()
                    content_hash = _sha256(raw_bytes)
                    try:
                        text_content = raw_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        return None, {"status_code": resp.status}

                    if suffix_lower == ".rst":
                        text_elements = rst_to_elements(text_content)
                    elif suffix_lower in (".md", ".txt"):
                        text_elements = md_to_elements(text_content)
                    else:
                        text_elements = md_to_elements(text_content)

                    chunk_pairs = split_by_semantic_elements(
                        text_elements, HTML_CHUNK_MAX_TOKENS, chunk_overlap=150
                    )

                    # Extract page title from first Title element (if any)
                    page_title = next(
                        (t for cls, t in text_elements if cls == "Title"), ""
                    )
                    filename = Path(urlparse(url).path).name or url.rsplit("/", 1)[-1]
                    target_coll = self._resolve_collection_name(referring_page_url or url)
                    retrieved_at = datetime.now(UTC).isoformat()
                    content_type = resp.headers.get("Content-Type", "")
                    provenance_meta = self._source_provenance_metadata(
                        url=url,
                        source_system="web_crawl",
                        source_kind="inline_text",
                        modality="text",
                        content_hash=content_hash,
                        retrieved_at=retrieved_at,
                        status_code=resp.status,
                        last_modified=resp.headers.get("Last-Modified"),
                        etag=resp.headers.get("ETag"),
                        final_uri=url,
                        content_type=content_type,
                        parser_version=f"{suffix_lower.lstrip('.')}-to-elements-v1",
                    )
                    base_meta = {
                        **self._resolve_product_metadata(url),
                        **self.extra_metadata,
                        **provenance_meta,
                        "source_uri": url,
                        "filename": filename,
                        "page_title": page_title,
                        "crawl_depth": depth,
                        "document_type": suffix_lower.lstrip("."),
                    }

                    chunks: list[tuple[str, dict, str]] = []
                    for chunk_text, section_path in chunk_pairs:
                        if not chunk_text.strip():
                            continue
                        meta = {**base_meta}
                        if section_path:
                            meta["section_path"] = section_path
                            meta["heading"] = section_path.split(" > ")[-1].strip()
                        chunks.append((chunk_text, meta, target_coll))

                    logger.info(
                        "Inline text %s: %d elements → %d chunks",
                        filename, len(text_elements), len(chunks),
                    )
                    return _INLINE_CHUNKS, {
                        "chunks": chunks,
                        "content_hash": content_hash,
                        "status_code": resp.status,
                        "last_modified": resp.headers.get("Last-Modified"),
                        "etag": resp.headers.get("ETag"),
                        "content_type": content_type,
                        **registry_provenance_fields(base_meta),
                    }

                is_nfs_document = suffix_lower in _NFS_DOCUMENT_EXTENSIONS
                is_nfs_media = suffix_lower in _NFS_MEDIA_EXTENSIONS

                if is_nfs_media:
                    content_length = int(resp.headers.get("Content-Length", 0))
                    max_bytes = self.max_media_file_mb * 1024 * 1024
                    if content_length > 0 and content_length > max_bytes:
                        errors.append({"url": url, "error_type": "missing_file",
                                       "status_code": resp.status,
                                       "error": f"exceeds {self.max_media_file_mb} MB limit"})
                        return None, {"status_code": resp.status}

                # ── NFS disk-write path ───────────────────────────────────────
                # Binary files are streamed to NFS-backed repo directories.
                # Phase 3 ingest is a deliberate, separate action via POST
                # /ingest-binaries so the user can choose the extraction method
                # (nemotron_parse / nv_ingest / both) after the crawl completes.
                if is_nfs_document:
                    is_pdf = suffix_lower == ".pdf"
                    nfs_root = self.pdf_repo_dir if is_pdf else self.docs_repo_dir
                    manifest_media_type = "document"
                    if not nfs_root:
                        is_nfs_document = False
                elif is_nfs_media:
                    if suffix_lower in _NFS_AUDIO_EXTENSIONS and self.audio_repo_dir:
                        nfs_root = self.audio_repo_dir
                        manifest_media_type = "audio"
                    elif suffix_lower in _NFS_VIDEO_EXTENSIONS and self.video_repo_dir:
                        nfs_root = self.video_repo_dir
                        manifest_media_type = "video"
                    else:
                        is_nfs_media = False

                if is_nfs_document or is_nfs_media:
                    target_coll = self._resolve_collection_name(referring_page_url or url)
                    dest_dir = Path(nfs_root) / target_coll
                    await asyncio.to_thread(dest_dir.mkdir, parents=True, exist_ok=True)
                    filename = Path(urlparse(url).path).name or f"download{suffix}"
                    local_path = str(dest_dir / filename)
                    file_size = 0
                    raw_bytes = await resp.read()
                    file_size = len(raw_bytes)
                    content_hash = _sha256(raw_bytes)
                    await asyncio.to_thread(_write_bytes, local_path, raw_bytes)
                    downloaded_at = datetime.now(UTC).isoformat()
                    content_type = resp.headers.get("Content-Type", "")
                    provenance_meta = self._source_provenance_metadata(
                        url=url,
                        source_system="web_crawl",
                        source_kind="downloaded_asset",
                        modality=manifest_media_type,
                        content_hash=content_hash,
                        retrieved_at=downloaded_at,
                        status_code=resp.status,
                        last_modified=resp.headers.get("Last-Modified"),
                        etag=resp.headers.get("ETag"),
                        final_uri=url,
                        content_type=content_type,
                        parser_version=None,
                        chunker_version=None,
                    )
                    file_meta = {
                        "last_modified": resp.headers.get("Last-Modified"),
                        "etag": resp.headers.get("ETag"),
                        "content_hash": content_hash,
                        "status_code": resp.status,
                        "content_type": content_type,
                        **registry_provenance_fields(provenance_meta),
                    }
                    logger.info(
                        "Binary saved: %s → %s (%d bytes)",
                        filename, local_path, file_size,
                    )
                    async with self._manifest_lock:
                        self._live_manifest = [
                            r for r in self._live_manifest if r.get("source_uri") != url
                        ]
                        self._live_manifest.append({
                            "source_uri": url,
                            "filename": filename,
                            "local_path": local_path,
                            "referring_page_url": referring_page_url,
                            "content_hash": content_hash,
                            "crawl_depth": depth,
                            "file_size_bytes": file_size,
                            "content_type": content_type,
                            "downloaded_at": downloaded_at,
                            "last_ingested_hash": "",
                            "collection_name": target_coll,
                            "media_type": manifest_media_type,
                            **registry_provenance_fields(provenance_meta),
                        })
                    return _MANIFEST, file_meta
                else:
                    file_meta = {
                        "last_modified": resp.headers.get("Last-Modified"),
                        "etag": resp.headers.get("ETag"),
                        "content_hash": "",
                        "status_code": resp.status,
                    }
                    filename = Path(urlparse(url).path).name or f"download{suffix}"
                    new_row = {
                        "source_uri": url, "filename": filename, "local_path": "",
                        "referring_page_url": referring_page_url, "content_hash": "",
                        "crawl_depth": depth,
                        "file_size_bytes": int(resp.headers.get("Content-Length") or 0),
                        "content_type": resp.headers.get("Content-Type", ""),
                        "downloaded_at": "", "last_ingested_hash": "",
                        "collection_name": self.collection_name, "media_type": "inline",
                    }
                    async with self._manifest_lock:
                        self._live_manifest = [r for r in self._live_manifest if r.get("source_uri") != url]
                        self._live_manifest.append(new_row)
                    return _MANIFEST, file_meta

        except aiohttp.ClientResponseError as exc:
            errors.append({"url": url, "error_type": "missing_file",
                           "status_code": exc.status, "error": str(exc)})
            return None, {"status_code": exc.status}
        except Exception as exc:
            errors.append({"url": url, "error_type": "missing_file",
                           "status_code": None, "error": str(exc)})
            return None, empty_meta

    # ── URL filtering ─────────────────────────────────────────────────────────

    def _is_blocked_url(self, url: str) -> bool:
        return any(pat in url for pat in self.blocked_url_patterns)

    def _link_followable(self, url: str) -> bool:
        """Return True if `url` may be enqueued.

        Same-host URLs are always followable.  Cross-host URLs are followable
        only when they match a `binary_host_allowlist` entry AND point to a
        binary or inline-text file (so HTML link-following remains same-host).
        """
        if _same_domain(url, self._netloc):
            return True
        if not self.binary_host_allowlist:
            return False
        url_lower = url.lower()
        if not any(allowed.lower() in url_lower for allowed in self.binary_host_allowlist):
            return False
        # Cross-host is permitted only for binary or inline-text URLs.
        path = urlparse(url).path.lower()
        if any(path.endswith(ext) for ext in _INLINE_TEXT_EXTENSIONS):
            return True
        return _is_binary_url(url)

    def _is_allowed_url(self, url: str) -> bool:
        if not self.allowed_url_prefixes:
            return True
        for prefix in self.allowed_url_prefixes:
            if url == prefix or url.startswith(prefix + "/") \
                    or url.startswith(prefix + "?") or url.startswith(prefix + "#"):
                return True
        return False

    # ── Sitemap ───────────────────────────────────────────────────────────────

    def _fetch_sitemap_seeds(self) -> list[str]:
        parsed = urlparse(self.start_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        headers = {"User-Agent": self._user_agent}
        try:
            import certifi
            _verify: str | bool = certifi.where()
        except ImportError:
            _verify = True

        sitemap_urls: list[str] = []
        try:
            resp = requests.get(f"{base}/robots.txt", headers=headers, timeout=15, verify=_verify)
            if resp.status_code == 200:
                for line in resp.text.splitlines():
                    if line.strip().lower().startswith("sitemap:"):
                        sitemap_urls.append(line.split(":", 1)[1].strip())
        except Exception as exc:
            logger.warning("sitemap: could not fetch robots.txt: %r", exc)

        if not sitemap_urls:
            return []

        page_urls: list[str] = []
        visited_sitemaps: set[str] = set()

        def _expand(url: str, depth: int = 0) -> None:
            if url in visited_sitemaps or depth > 5:
                return
            visited_sitemaps.add(url)
            try:
                r = requests.get(url, headers=headers, timeout=30, verify=_verify)
                if r.status_code != 200:
                    return
                root = ET.fromstring(r.content)
                ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
                def tag(t: str) -> str:
                    return f"{{{ns}}}{t}" if ns else t
                for loc in root.findall(f".//{tag('sitemap')}/{tag('loc')}"):
                    _expand(loc.text.strip(), depth + 1)
                for loc in root.findall(f".//{tag('url')}/{tag('loc')}"):
                    u = loc.text.strip()
                    if (
                        _same_domain(u, self._netloc)
                        and self._is_allowed_url(u)
                        and not self._is_blocked_url(u)
                        and not _is_binary_url(u)
                    ):
                        page_urls.append(u)
            except Exception as exc:
                logger.warning("sitemap: error fetching %s: %r", url, exc)

        for sm_url in sitemap_urls:
            _expand(sm_url)

        return page_urls

    # ── Registry / artifact helpers ───────────────────────────────────────────

    def _collection_slug(self) -> str:
        # For product-URL-map crawls the registry is always keyed by hostname so
        # it's reused across runs regardless of which fallback collection_name was
        # chosen.  Fixed-collection crawls key by collection_name as before.
        if not self.use_product_url_map and self.collection_name:
            return self.collection_name.replace(".", "_").replace("-", "_")
        netloc = self._netloc.split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc.replace(".", "_").replace("-", "_")

    def _registry_path(self) -> str:
        return os.path.join(self.registry_dir, f"{self._collection_slug()}_url_registry.json")

    def _load_registry(self) -> dict[str, dict]:
        path = self._registry_path()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                logger.info("Loaded URL registry from %s (%d entries)", path, len(data))
                return data
        except FileNotFoundError:
            logger.info("No registry at %s — starting fresh", path)
        except Exception as exc:
            logger.warning("Could not load registry from %s: %s", path, exc)
        return {}

    def _save_registry(self, registry: dict[str, dict], *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._registry_last_saved) < 60.0:
            return
        self._registry_last_saved = now
        path = self._registry_path()
        tmp_path = path + ".tmp"
        try:
            os.makedirs(self.registry_dir, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(registry, fh, separators=(",", ":"))
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.warning("Could not save registry to %s: %s", path, exc)

    def _binary_manifest_path(self) -> str:
        return os.path.join(self.registry_dir, f"{self._collection_slug()}_binary_manifest.csv")

    def _load_binary_manifest(self) -> list[dict]:
        path = self._binary_manifest_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        except Exception as exc:
            logger.warning("Could not load binary manifest at %s: %s", path, exc)
            return []

    def _save_binary_manifest(self, manifest: list[dict]) -> None:
        path = self._binary_manifest_path()
        try:
            os.makedirs(self.registry_dir, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_BINARY_MANIFEST_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(manifest)
        except Exception as exc:
            logger.warning("Could not save binary manifest to %s: %s", path, exc)

    def _flush_errors_to_csv(self, errors: list[dict], error_matrix: dict[str, list[dict]]) -> None:
        for key in ("broken_links", "missing_files", "ingest_failures", "batch_errors", "corrupt_files"):
            error_matrix[key] = []
        for e in errors:
            etype = e.get("error_type", "other")
            entry = {k: v for k, v in e.items() if k != "error_type"}
            bucket = {
                "broken_link": "broken_links",
                "missing_file": "missing_files",
                "ingest_failure": "ingest_failures",
                "batch_error": "batch_errors",
                "corrupt_file": "corrupt_files",
            }.get(etype, "other")
            error_matrix.setdefault(bucket, []).append(entry)
        csv_path = os.path.join(self.registry_dir, f"{self._collection_slug()}_error_matrix.csv")
        total = sum(len(v) for v in error_matrix.values())
        try:
            os.makedirs(self.registry_dir, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["category", "url", "status_code", "error"],
                                        extrasaction="ignore")
                writer.writeheader()
                for category, entries in error_matrix.items():
                    for entry in entries:
                        writer.writerow({"category": category, **entry})
            logger.info("Error matrix written to %s (%d entries)", csv_path, total)
        except Exception as exc:
            logger.warning("Could not write error matrix to %s: %s", csv_path, exc)

    def _export_crawl_artifacts(self) -> None:
        if not self.export_dir:
            return
        import shutil
        slug = self._collection_slug()
        sources = [
            os.path.join(self.registry_dir, f"{slug}_url_registry.json"),
            os.path.join(self.registry_dir, f"{slug}_error_matrix.csv"),
            os.path.join(self.registry_dir, f"{slug}_binary_manifest.csv"),
        ]
        try:
            os.makedirs(self.export_dir, exist_ok=True)
        except OSError:
            return
        for src in sources:
            if not os.path.exists(src):
                continue
            dst = os.path.join(self.export_dir, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            try:
                shutil.copy2(src, dst)
            except OSError:
                pass

    @staticmethod
    async def _save_temp_stream_async(resp: aiohttp.ClientResponse, suffix: str = ".bin") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="crawler_")
        with os.fdopen(fd, "wb") as fh:
            async for chunk in resp.content.iter_chunked(65536):
                if chunk:
                    fh.write(chunk)
        return path
