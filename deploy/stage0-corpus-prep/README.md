# Stage 0 Corpus Prep Kubernetes Job

This Job reconstructs Stage 0 passages from the Elasticsearch crawl/vector
index, optionally enriches provenance from the crawler URL registry, writes
provenance sidecars, and emits MLflow-ready observability JSON. It is the
first K8s-native source pipeline step before Stage 1 entailment extraction.

For each collection, the Job writes:

```text
/datasets/<collection>/
  passages.jsonl
  manifests/crawl_run.json
  provenance/source_revisions.jsonl
  provenance/source_chunks.jsonl

/observability/stage0-corpus-prep/<collection>/
  run_context.json
  metrics.json
  artifacts_manifest.json
  service_refs.json
```

## Build Image

```bash
docker build \
  -f deploy/stage0-corpus-prep/Containerfile \
  -t <registry>/docs-to-data-to-lora/stage0-corpus-prep:latest .

docker push <registry>/docs-to-data-to-lora/stage0-corpus-prep:latest
```

Update `deploy/stage0-corpus-prep/job.yaml` with that image.

## Elasticsearch Secret

The Job expects an Elasticsearch password Secret in the Job namespace:

```bash
kubectl create secret generic pipeline-elasticsearch \
  -n nemo-peft \
  --from-literal=PIPELINE_ES_PASSWORD='<elastic-password>'
```

If the password is managed by ECK in another namespace, copy or external-secret
sync it into the namespace that runs this Job. Kubernetes Secrets are
namespace-scoped, so the Job cannot mount the ECK Secret directly unless it runs
in the same namespace.

## Collection Pattern

The template is for `nim_curated`:

```text
--index nim_curated
--output-dir /datasets/nim_curated
--observability-dir /observability/stage0-corpus-prep/nim_curated
--url-registry /crawler-registry/nim_curated_url_registry.json
```

Create a second Job or patch these values for `nemo_usvcs_curated`. Keeping one
collection per Job makes recrawl deltas, retries, and MLflow child runs easier
to audit. The `crawler-registry` PVC name in `job.yaml` is a placeholder for the
volume that contains `<collection>_url_registry.json`; patch it to the crawler
export PVC used in the cluster.

## Observability

The Job does not install the MLflow client. It writes the standard repository
observability files so a later MLflow export Job can log metrics and artifacts.
The emitted metrics include ES hit counts, extracted chunk counts, passage
counts, source revision/chunk counts, token totals, doc-kind/product-family
breakdowns, and URL registry coverage metrics when a registry is supplied. The
registry artifact is also listed in `artifacts_manifest.json`.

## Local Run

```bash
export PIPELINE_ES_PASSWORD='<elastic-password>'
python scripts/pipeline/stage0_corpus_prep.py \
  --index nim_curated \
  --es-host https://rag-eck-elasticsearch-es-http.runai-rag:9200 \
  --output-dir /tmp/nim_curated \
  --observability-dir /tmp/stage0-observability/nim_curated \
  --url-registry /mnt/nvme2/crawler-registry/nim_curated_url_registry.json
```
