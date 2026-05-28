FROM python:3.12-slim

# ── System dependencies ────────────────────────────────────────────────────────
# chromium + chromedriver for Selenium JavaScript rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        chromium-driver \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

# ── Runtime ───────────────────────────────────────────────────────────────────
# Writable dirs for registry/exports (override with volume mounts)
ENV CRAWLER_REGISTRY_DIR=/tmp/crawler-registry \
    CRAWLER_EXPORT_DIR=/tmp/crawl-exports \
    CHROMIUM_BIN=/usr/bin/chromium \
    SE_CACHE_PATH=/tmp/.selenium

EXPOSE 8085

CMD ["uvicorn", "crawler.server:app", "--host", "0.0.0.0", "--port", "8085", "--workers", "1"]
