# rag-crawler

Standalone async web crawler for RAG corpus ingestion. HTML and inline-text
(.md/.rst/.txt) content is chunked semantically and bulk-written to
Elasticsearch via a NIM embedding endpoint. Binary files (PDF/DOCX/PPTX)
are downloaded to NFS for later Phase 3 ingest through a separate
extraction pipeline.

Companion methodology + worked examples at:
<https://github.com/joncoons/docs-to-data-to-lora> (private)

## Features

- **BFS crawl** with persistent URL registry (NFS-backed for restart resilience)
- **Selenium SPA rendering** (Chromium headless) for JS-heavy docs sites
- **Hash-based delta upsert** — re-crawl skips unchanged pages
- **Cross-host content allowlist** — capture GitHub READMEs and CDN-hosted
  binaries while keeping HTML link-following scoped to the docs host
- **Seed URLs** for SPA sites where sidebar nav doesn't expose all sub-pages
  in the initial DOM
- **Phase 3 deferral** — downloads binaries during the HTML crawl, leaves
  PDF/DOCX parsing as a separate explicit action (so the GPU-heavy parse
  can be scheduled independently)
- **Per-domain config profiles** + **APScheduler-backed recurring crawls**
- **Redis-backed task state** so pod restarts don't lose in-flight crawl
  visibility (falls back to in-memory store if Redis is unavailable)

## Quick start

```bash
# build
docker build -t rag-crawler:latest .

# run (set env vars per src/crawler/config.py)
docker run -p 8085:8085 \
  -e CRAWLER_EMBED_URL=http://your-embedding-nim:8000/v1/embeddings \
  -e CRAWLER_ES_URL=https://your-es:9200 \
  -e CRAWLER_ES_USER=elastic \
  -e CRAWLER_ES_PASS=... \
  -v /your/registry:/workspace/artifacts/crawler-registry \
  rag-crawler:latest

# crawl
curl -X POST http://localhost:8085/crawl -H 'Content-Type: application/json' -d @- <<EOF
{
  "start_url": "https://docs.example.com/foo/latest/",
  "collection_name": "foo_curated",
  "use_product_url_map": false,
  "max_depth": null,
  "extract_linked_files": true,
  "allowed_url_prefixes": ["https://docs.example.com/foo/latest"]
}
EOF
```

See `src/crawler/server.py:CrawlRequest` for the full API.

## API surface

- `POST /crawl` — start an async crawl, returns `{task_id}`
- `GET /status?task_id=...` — current state + progress + final result
- `POST /cancel` — cancel an in-flight crawl
- `GET /collections` — list ES collections + chunk counts
- `GET /binary-manifest` — across all collections
- `POST /ingest-binaries` — Phase 3: PDF/DOCX/PPTX → parser → ES
- `GET /schedules` / `POST /schedules` — recurring crawls via APScheduler
- `GET /domain-configs` / `POST /domain-configs/generate` — per-domain crawl
  profiles

## Architecture notes

- BFS popleft with `_FETCH_CONCURRENCY=10` parallel `_fetch_html` coroutines
- All Selenium renders serialized through `_selenium_lock` (single browser
  instance) — known throughput bottleneck on Fern-style SPA docs sites
- Inline-text content (.md/.rst/.txt) chunked through the same semantic
  chunker as HTML and embedded inline (no NFS write, no Phase 3)
- Binary downloads stream to `/<pdf|docs|audio|video>-repo/<collection>/`
  on NFS and recorded in a per-collection manifest CSV
- Registry stored as `<collection>_url_registry.json` on NFS — keys are
  full URLs; values include `content_hash` (sha256), `last_seen`,
  `last_ingested`, `linked_hrefs`, ETag/Last-Modified for conditional GETs
- ES chunk metadata includes source-agnostic provenance fields under
  `metadata.provenance` plus join keys in `metadata.source` and
  `metadata.content_metadata`. This keeps webcrawl aligned with future
  document capture, dense image captioning, audio transcription, and video
  summarization sources.

## Not in this repo (operational concerns)

- Kubernetes manifests for ECK-managed Elasticsearch
- NIM operator setup for embedding/ranking models
- Run.ai PodGroup config
- Per-organization product URL map (`src/crawler/product_url_map.py`
  contains a sample — substitute your own)

## License

(none yet — private repo)
