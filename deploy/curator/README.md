# NeMo Curator Kubernetes Handoff

This template moves Stage 3 generic curation toward native NeMo Curator. The
repository still owns the SFT sample schema, lineage sidecars, split writing,
and MLflow-ready observability JSON. NeMo Curator owns quality filtering and
deduplication over the normalized `text` field.

The handoff has three phases:

1. `prepare`: convert `provenance/dataset_samples.jsonl` into
   `curator/input/dataset_samples.jsonl`, copy the Curator config, and write a
   submission plan.
2. `native-filter`: run NeMo Curator `filter_documents` in the Curator image,
   with retained records written to `curator/retained/`, removed records
   written to `curator/removed/`, and scores written to `curator/scores/`.
3. `collect`: read retained/removed records, write curated sample sidecars,
   create `training.jsonl` and `validation.jsonl`, and emit observability.

## Inputs

```text
/datasets/<collection>/
  provenance/dataset_samples.jsonl
```

## Prepare Outputs

```text
/datasets/<collection>/curator/
  input/dataset_samples.jsonl
  curator_config.yaml
  submission_plan.json
```

## Collect Outputs

```text
/datasets/<collection>/
  training.jsonl
  validation.jsonl
  curator/accepted_samples.jsonl
  curator/rejected_samples.jsonl
  curator/rejection_report.jsonl
  curator/curation_manifest.json
```

## Build Image

```bash
docker build \
  -f deploy/curator/Containerfile \
  -t <registry>/docs-to-data-to-lora/curator:latest .

docker push <registry>/docs-to-data-to-lora/curator:latest
```

The image extends `nvcr.io/nvidia/nemo-curator:26.04` by default, matching the
current NGC Curator tag checked on 2026-05-29. Override `NEMO_CURATOR_IMAGE` at
build time if the showcase cluster is pinned to a different Curator release.

## Local Prepare

```bash
python scripts/pipeline/curator_handoff.py \
  --dataset-dir /mnt/nvme2/peft/datasets/v2/nim_curated \
  --collection nim_curated \
  --mode prepare \
  --config-file configs/curator/sft-dedup-quality.yaml \
  --observability-dir /tmp/curator-observability/nim_curated
```

## Native Filter Job

After `prepare`, apply `deploy/curator/native-filter-job.yaml`. It runs the
Curator `filter_documents` CLI against `curator/input/` using
`configs/curator/sft-filter-documents.yaml` and writes the exact retained,
removed, and score directories expected by `collect`.

```bash
kubectl apply -f deploy/curator/native-filter-job.yaml
kubectl wait -n nemo-peft --for=condition=complete job/curator-native-filter-nim-curated
```

The default job uses `CURATOR_DEVICE=cpu`, which is suitable for the lightweight
heuristic filter pass. For GPU-backed exact/fuzzy deduplication, use the same
prepared input and Curator cache/output directories but switch the job to a GPU
node pool, add `nvidia.com/gpu` resource limits, set `CURATOR_DEVICE=gpu`, and
run Curator's documented exact/fuzzy dedup commands before `collect`. Semantic
dedup is intentionally disabled in `configs/curator/sft-dedup-quality.yaml`
until the showcase embedding model and GPU budget are selected.

## Local Collect

```bash
python scripts/pipeline/curator_handoff.py \
  --dataset-dir /mnt/nvme2/peft/datasets/v2/nim_curated \
  --collection nim_curated \
  --mode collect \
  --curator-job-id <curator-job-id> \
  --accepted-dir /mnt/nvme2/peft/datasets/v2/nim_curated/curator/retained \
  --rejected-dir /mnt/nvme2/peft/datasets/v2/nim_curated/curator/removed \
  --observability-dir /tmp/curator-observability/nim_curated
```

After `collect`, run dataset finalization. It records
`curator/curation_manifest.json`, `curator_config_hash`, Curator job ID, and
curation artifacts in the dataset version manifest.
