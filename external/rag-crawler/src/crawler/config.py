"""
config.py — environment-driven configuration for rag-crawler.

All external service coordinates come from env vars so the container
needs no code changes when deploying to different namespaces or clusters.
"""

import os
import ssl

# ── Embedding NIM ─────────────────────────────────────────────────────────────
EMBED_URL: str = os.environ.get(
    "CRAWLER_EMBED_URL",
    "http://nemoretriever-embedding-ms:8000/v1/embeddings",
)
EMBED_MODEL: str = os.environ.get(
    "CRAWLER_EMBED_MODEL",
    "nvidia/nv-embedqa-e5-v5",
)
EMBED_DIMENSIONS: int = int(os.environ.get("CRAWLER_EMBED_DIMENSIONS", "2048"))
# Max texts per single POST /embeddings call
EMBED_BATCH_SIZE: int = int(os.environ.get("CRAWLER_EMBED_BATCH_SIZE", "32"))
# Max concurrent embedding requests in-flight at once
EMBED_CONCURRENCY: int = int(os.environ.get("CRAWLER_EMBED_CONCURRENCY", "4"))

# ── Elasticsearch ─────────────────────────────────────────────────────────────
ES_URL: str = os.environ.get(
    "CRAWLER_ES_URL",
    "https://rag-eck-elasticsearch-es-http:9200",
)
ES_USER: str = os.environ.get("CRAWLER_ES_USER", "elastic")
ES_PASS: str = os.environ.get("CRAWLER_ES_PASS", "")
# Path to CA cert for ECK TLS (mounted as a ConfigMap in k8s)
ES_CA_CERT: str = os.environ.get("CRAWLER_ES_CA_CERT", "/etc/ssl/eck/ca.crt")
# ES bulk batch size (docs per _bulk request)
ES_BULK_BATCH_SIZE: int = int(os.environ.get("CRAWLER_ES_BULK_BATCH_SIZE", "200"))

# ── Ingestor (binary Phase 3 hand-off) ───────────────────────────────────────
INGESTOR_URL: str = os.environ.get(
    "CRAWLER_INGESTOR_URL",
    "http://ingestor-server:8082",
)

# ── Crawler runtime ───────────────────────────────────────────────────────────
REGISTRY_DIR: str = os.environ.get("CRAWLER_REGISTRY_DIR", "/tmp/crawler-registry")
CONFIGS_DIR: str = os.environ.get("CRAWLER_CONFIGS_DIR", "/workspace/artifacts/crawler-configs")
EXPORT_DIR: str = os.environ.get("CRAWLER_EXPORT_DIR", "/crawl-exports")
PDF_REPO_DIR: str = os.environ.get("CRAWLER_PDF_REPO_DIR", "/workspace/artifacts/pdf-repo")
DOCS_REPO_DIR: str = os.environ.get("CRAWLER_DOCS_REPO_DIR", "/workspace/artifacts/docs-repo")
AUDIO_REPO_DIR: str = os.environ.get("CRAWLER_AUDIO_REPO_DIR", "/workspace/artifacts/audio-repo")
VIDEO_REPO_DIR: str = os.environ.get("CRAWLER_VIDEO_REPO_DIR", "/workspace/artifacts/video-repo")
# Optional path to JSON override for CRAWLER_PRODUCT_URL_MAP
PRODUCT_MAP_PATH: str = os.environ.get("APP_CRAWLER_PRODUCT_MAP", "")

# ── Redis task backend ────────────────────────────────────────────────────────
REDIS_HOST: str = os.environ.get("REDIS_HOST", "rag-redis-master")
REDIS_PORT: int = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB: int = int(os.environ.get("REDIS_DB", "0"))
REDIS_TTL_SECONDS: int = int(os.environ.get("REDIS_STATUS_TTL_SECONDS", "172800"))  # 48 h
ENABLE_REDIS_BACKEND: bool = os.environ.get("ENABLE_REDIS_BACKEND", "true").lower() == "true"


def build_es_ssl_ctx() -> ssl.SSLContext | None:
    """Return an SSL context for ES if CA cert exists, else None."""
    ca = ES_CA_CERT
    if ca and os.path.exists(ca):
        ctx = ssl.create_default_context(cafile=ca)
        return ctx
    return None
