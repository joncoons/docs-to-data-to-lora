# Dataset Registration Kubernetes Job

This Job registers Stage 3 datasets in NeMo Data Store and NeMo Entity Store.
It is the first dataset-lineage control point in the K8s-native showcase.

The Job registers, per collection:

- `stage3-<collection>` training/validation dataset for NeMo Customizer
- `stage3-<collection>-test` bare held-out evaluation dataset
- `stage3-<collection>-test-with-context` context-baked Evaluator dataset

It also emits MLflow-ready observability JSON files. The Job does not install
the MLflow client or log directly to MLflow.

## Build Image

```bash
docker build \
  -f deploy/dataset-registration/Containerfile \
  -t <registry>/docs-to-data-to-lora/dataset-registration:latest .

docker push <registry>/docs-to-data-to-lora/dataset-registration:latest
```

Update `deploy/dataset-registration/job.yaml` with that image.

## Inputs

The template expects the dataset artifact PVC mounted at `/datasets`:

```text
/datasets/
  nim_curated/
    training.jsonl
    validation.jsonl
    test_set.jsonl
    test_set_with_context.jsonl
    provenance/
    manifests/
  nemo_usvcs_curated/
    ...
```

`manifests/dataset_version_manifest.json` is required by default. The Job
uploads it and recognized lineage sidecars from `manifests/`, `provenance/`,
and `data_designer/` to the NeMo Data Store dataset repository alongside the
split JSONL files. Use
`--allow-missing-lineage` only for legacy or ad hoc dry runs that have not yet
passed through dataset finalization.

## Secrets

Create the Data Store Git/API credential Secret:

```bash
kubectl create secret generic nemo-data-store-git \
  -n nemo-peft \
  --from-literal=DATA_STORE_USER='<username>' \
  --from-literal=DATA_STORE_PASSWORD='<password>'
```

## Service Configuration

The Job sources `deploy/nemo-platform/service-plane-configmap.yaml` and maps
the Platform keys into the legacy variable names used by the uploader:

```text
ENTITY_STORE_URL <- NMP_ENTITY_STORE_URL
DATA_STORE_URL <- NMP_DATASTORE_URL
DATA_STORE_GIT_BASE <- NMP_DATASTORE_GIT_BASE
DATA_STORE_HF_ENDPOINT <- NMP_DATASTORE_HF_ENDPOINT
```

`DATA_STORE_GIT_BASE` remains a compatibility endpoint for the current
HF/Git-style upload path. The Platform-native follow-up is to register/upload
training artifacts as FileSets and hand `fileset://workspace/name` URIs to
Customizer.

## Observability Output

The Job writes:

```text
/outputs/observability/dataset-registration/
  run_context.json
  metrics.json
  artifacts_manifest.json
  service_refs.json
```

These files are designed for a later MLflow export Job. They include dataset
row counts, file checksums, NeMo Data Store URIs, Entity Store refs, dataset
version IDs, uploaded lineage file lists, and provenance sidecar references.
When `provenance/dataset_samples.jsonl` is present, registration also emits
source composition metrics for source systems, source kinds, modalities,
distinct source revisions/chunks, entailments, and synthetic versus grounded
samples. Curator input, config, manifest, accepted samples, rejected samples,
and rejection reports are uploaded as lineage sidecars when present.

## Local Dry Run

```bash
python scripts/eval/upload_test_datasets.py \
  --base-dir /mnt/nvme2/peft/datasets/v2 \
  --collections nim_curated nemo_usvcs_curated \
  --include-train \
  --include-test \
  --include-context-test \
  --observability-dir /tmp/dataset-registration-observability \
  --dry-run
```

Dry run validates files and writes observability output without calling NeMo
services or pushing to Data Store. It still requires the finalized dataset
manifest unless `--allow-missing-lineage` is supplied.
