# Dataset Finalization Kubernetes Job

This Job closes the dataset lineage loop before NeMo Data Store and Entity
Store registration. It reads the finalized split files and provenance sidecars
for one collection, writes `manifests/dataset_version_manifest.json`, and emits
MLflow-ready observability JSON.

For each collection, the Job expects files like:

```text
/datasets/<collection>/
  stage2_eval.jsonl
  training.jsonl
  validation.jsonl
  test_set.jsonl
  test_kvp_uids.json
  manifests/crawl_run.json
  provenance/source_revisions.jsonl
  provenance/source_chunks.jsonl
  provenance/entailments.jsonl
  provenance/dataset_samples.jsonl
  curator/curation_manifest.json        # optional, after NeMo Curator collect
```

If `provenance/dataset_samples.jsonl` is missing but `stage2_eval.jsonl` is
present, the Job backfills the sample sidecar from Stage 2 rows before creating
the version manifest.

## Build Image

```bash
docker build \
  -f deploy/dataset-finalization/Containerfile \
  -t <registry>/docs-to-data-to-lora/dataset-finalization:latest .

docker push <registry>/docs-to-data-to-lora/dataset-finalization:latest
```

Update `deploy/dataset-finalization/job.yaml` with that image.

## Output

```text
/datasets/<collection>/manifests/dataset_version_manifest.json

/observability/dataset-finalization/<collection>/
  run_context.json
  metrics.json
  artifacts_manifest.json
  service_refs.json
```

The manifest includes a deterministic `dataset_version_id`, split counts,
artifact checksums, source revision/chunk coverage, source system/kind/modality
composition, synthetic versus grounded counts, Data Designer job IDs when
present in sample lineage, and Curator job/config metadata when
`curator/curation_manifest.json` is present.

## Local Run

```bash
python scripts/pipeline/finalize_dataset.py \
  --dataset-dir <DATASET_ROOT>/nim_curated \
  --dataset-name nim_curated \
  --observability-dir /tmp/dataset-finalization-observability/nim_curated
```
