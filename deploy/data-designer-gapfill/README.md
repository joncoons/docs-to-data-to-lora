# Data Designer Gap-Fill Kubernetes Job

This Job bridges the custom Stage 1.5 coverage analysis to native NeMo Data
Designer execution. It consumes `data_designer/gapfill_requests.jsonl`, prepares
a datastore-backed seed dataset plan, optionally submits a Data Designer job, and
normalizes returned synthetic pairs back into Stage 1.5 rows with provenance.
The downstream Stage 2 QA and dataset-finalization path reads those rows and
admits `provenance/data_designer_samples.jsonl` by `sample_id`, so Data
Designer job and gap lineage survives prompt/answer refinement.

The implementation follows the NeMo Data Designer service model: seed data must
be uploaded to the configured datastore before generation, and live job creation
uses the NVIDIA Python SDK when that SDK is installed in the image.

## Modes

- `prepare`: write `data_designer/seed_dataset.csv` and
  `data_designer/submission_plan.json` without calling Data Designer.
- `submit`: upload the seed CSV and create a Data Designer job through the SDK.
- `collect`: read downloaded/generated results and write normalized artifacts.
- `submit-and-collect`: submit and collect in one invocation when results are
  available to the same pod.

## Inputs

```text
/datasets/<collection>/
  provenance/gap_manifest.json
  data_designer/gapfill_requests.jsonl
```

## Outputs

Prepare/submit outputs:

```text
/datasets/<collection>/data_designer/
  seed_dataset.csv
  submission_plan.json
```

Collect outputs:

```text
/datasets/<collection>/
  stage1_5_gapfill.jsonl
  data_designer/result_manifest.json
  data_designer/generated_samples.jsonl
  provenance/data_designer_samples.jsonl
```

Observability output:

```text
/observability/data-designer-gapfill/<collection>/
  run_context.json
  metrics.json
  artifacts_manifest.json
  service_refs.json
```

## Build Image

For prepare/collect-only usage:

```bash
docker build \
  -f deploy/data-designer-gapfill/Containerfile \
  -t <registry>/docs-to-data-to-lora/data-designer-gapfill:latest .
```

For live submission, add the NVIDIA SDK packages from the package source used
by your NeMo Microservices deployment:

```bash
docker build \
  -f deploy/data-designer-gapfill/Containerfile \
  --build-arg NEMO_SDK_REQUIREMENTS='<nemo-sdk-package-specs>' \
  -t <registry>/docs-to-data-to-lora/data-designer-gapfill:latest .
```

## Local Prepare

```bash
python scripts/pipeline/data_designer_gapfill.py \
  --dataset-dir /mnt/nvme2/peft/datasets/v2/nim_curated \
  --collection nim_curated \
  --mode prepare \
  --observability-dir /tmp/data-designer-gapfill-observability/nim_curated
```

## Local Collect

```bash
python scripts/pipeline/data_designer_gapfill.py \
  --dataset-dir /mnt/nvme2/peft/datasets/v2/nim_curated \
  --collection nim_curated \
  --mode collect \
  --job-id <data-designer-job-id> \
  --results-jsonl /path/to/data-designer-results.jsonl \
  --observability-dir /tmp/data-designer-gapfill-observability/nim_curated
```

`collect` accepts JSONL records that preserve seed columns and include either a
`qa_pairs_json` field containing `{"pairs": [...]}` or direct `question` and
`answer` fields.
