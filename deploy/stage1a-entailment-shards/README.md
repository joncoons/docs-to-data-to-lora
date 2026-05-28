# Stage 1A Entailment Shards Kubernetes Job

This Indexed Job runs Stage 1A logical entailment extraction over the Stage 0
`passages.jsonl` output. It calls an OpenAI-compatible NIM endpoint, expands
logical entailments into source-grounded KVP rows, writes per-shard provenance,
and emits MLflow-ready observability files.

The template uses stable hash sharding by `passage_id`. That keeps shard
assignment stable when recrawls append or update unrelated passages.

## Build Image

```bash
docker build \
  -f deploy/stage1a-entailment-shards/Containerfile \
  -t <registry>/docs-to-data-to-lora/stage1a-entailment-shards:latest .

docker push <registry>/docs-to-data-to-lora/stage1a-entailment-shards:latest
```

Update `deploy/stage1a-entailment-shards/job.yaml` with that image.

## Inputs

The Job expects Stage 0 artifacts on the dataset PVC:

```text
/datasets/nim_curated/
  passages.jsonl
  manifests/crawl_run.json
  provenance/source_revisions.jsonl
  provenance/source_chunks.jsonl
```

Patch the collection paths for `nemo_usvcs_curated` or create a second Job.

## Shard Outputs

For `SHARD_COUNT=8`, each indexed pod writes distinct files:

```text
/datasets/nim_curated/
  stage1a_le.shard-00000-of-00008.jsonl
  stage1a_le.shard-00001-of-00008.jsonl
  ...
  provenance/entailments.shard-00000-of-00008.jsonl
  provenance/entailments.shard-00001-of-00008.jsonl
  ...

/observability/stage1a-le-kvp/nim_curated/<shard-index>/
  run_context.json
  metrics.json
  artifacts_manifest.json
  service_refs.json
```

A later aggregation step should concatenate shard files in lexical order into
`stage1a_le.jsonl` and `provenance/entailments.jsonl` for compatibility with the
current monolithic downstream pipeline.

## NIM Endpoint Configuration

The default template targets the in-cluster Super 120B NIM endpoint:

```text
PIPELINE_NIM_ENDPOINTS=http://nim-llm-super-120b-bw.runai-rag:8000/v1
PIPELINE_LLM_MODEL=nvidia/nemotron-3-super-120b-a12b
PIPELINE_NIM_API_KEY=local
```

Set `PIPELINE_NIM_ENDPOINTS` to a comma-separated list to round-robin across
multiple compatible endpoints. If the endpoint requires auth, replace the
literal `PIPELINE_NIM_API_KEY=local` with a Secret reference.

## Local Shard Run

```bash
python scripts/pipeline/stage1a_le_kvp.py \
  --input-passages /tmp/nim_curated/passages.jsonl \
  --output-dir /tmp/nim_curated \
  --observability-dir /tmp/stage1a-observability/nim_curated/0 \
  --shard-index 0 \
  --shard-count 8 \
  --nim-endpoints http://nim-llm-super-120b-bw.runai-rag:8000/v1 \
  --model nvidia/nemotron-3-super-120b-a12b
```
