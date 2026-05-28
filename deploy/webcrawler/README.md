# Webcrawler Kubernetes Deployment

This target makes the in-repo crawler copy at `external/rag-crawler` a
K8s-native ingestion component for the docs-to-data-to-lora showcase. It is the
source acquisition layer before Stage 0 corpus preparation.

The crawler writes HTML and inline text chunks directly to Elasticsearch through
the configured embedding NIM. It also writes crawler registry, crawl exports,
and downloaded document/media assets to persistent volumes for later Stage 0
provenance enrichment and optional binary ingestion.

## Build Image

```bash
docker build \
  -f deploy/webcrawler/Containerfile \
  -t <registry>/docs-to-data-to-lora/webcrawler:latest .

docker push <registry>/docs-to-data-to-lora/webcrawler:latest
```

Update `deploy/webcrawler/deployment.yaml` with that image.

## Runtime Shape

The webcrawler has two K8s roles:

- `deployment.yaml`: long-running FastAPI service for crawl requests, status,
  schedules, registry access, and binary-manifest access.
- `crawl-job.yaml` and `crawl-cronjob.yaml`: deterministic per-collection submit
  jobs that call the service API and wait for task completion.

This keeps crawl execution observable and retryable without exposing the service
outside the cluster. The Service is `ClusterIP` by default.

## Required Dependencies

Patch `webcrawler-config` in `deployment.yaml` for the cluster-local service
names in the target namespace:

```text
CRAWLER_EMBED_URL=http://<embedding-nim-service>:8000/v1/embeddings
CRAWLER_ES_URL=https://<elasticsearch-service>:9200
CRAWLER_INGESTOR_URL=http://<ingestor-service>:8082
REDIS_HOST=<redis-service>
```

The template expects the Elasticsearch password in the same Secret used by the
Stage 0 job:

```text
Secret: pipeline-elasticsearch
Key:    PIPELINE_ES_PASSWORD
```

If the ECK CA ConfigMap is available, mount it as `eck-es-ca-cert`. The volume is
optional so local non-TLS or separately trusted ES deployments do not need to
carry that ConfigMap.

## Persistent Volumes

The deployment uses PVCs instead of `hostPath`:

| PVC | Mount | Purpose |
|---|---|---|
| `crawler-registry` | `/crawler-registry` | URL registry, binary manifests, error matrices |
| `crawler-configs` | `/crawler-configs` | Generated domain config profiles |
| `crawler-exports` | `/crawl-exports` | Crawl artifacts and exported reports |
| `crawler-source-assets` | `/source-assets` | Downloaded PDFs, docs, audio, and video assets |

Create or patch these claim names to match the storage class and retention policy
for the cluster. Stage 0 can mount `crawler-registry` read-only to enrich
`source_revisions.jsonl` and MLflow-ready observability metrics.

## Submit A Crawl

Patch the request body in `crawl-job.yaml`, then apply the deployment and job:

```bash
kubectl apply -f deploy/webcrawler/deployment.yaml
kubectl rollout status deploy/webcrawler -n nemo-peft
kubectl apply -f deploy/webcrawler/crawl-job.yaml
```

The Job posts `/crawl`, extracts the returned `task_id`, polls `/status`, and
exits non-zero if the crawler reports `FAILURE` or the task cannot be found.

For scheduled recrawls, patch the request and schedule in `crawl-cronjob.yaml`
and apply it after the Deployment is healthy. The default schedule is weekly on
Sunday at 05:00 UTC.

## Downstream Flow

After a crawl completes, run Stage 0 with the registry mounted:

```text
--index nim_curated
--url-registry /crawler-registry/nim_curated_url_registry.json
```

The crawler now stamps source-agnostic provenance into ES chunks under
`metadata.provenance` and writes registry records with normalized
`sha256:<hex>` source-content hashes. Stage 0 still treats the registry as an
optional compatibility/artifact input, while ES metadata becomes the durable
cross-source join point for web, document capture, dense image captioning, and
video/audio summarization text.
