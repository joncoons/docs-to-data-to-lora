# Integration Plan

## Objective

Create a repeatable adapter-training workflow where MLflow records lineage and
promotion state, and NeMo Platform performs the actual training/evaluation work:

- NeMo Platform FileSets store Customizer-ready dataset and model files.
- NeMo Platform Model Entities identify the trainable base models.
- NeMo Customizer trains LoRA adapters.
- NeMo Evaluator scores adapters, bases, and the 49B comparator target.
- NeMo Data Store and Entity Store remain compatibility/provenance surfaces.
- MLflow links every step with a stable run graph.

## Control Plane Boundary

| Concern | Owner | Notes |
|---|---|---|
| Run identity, experiment grouping, lineage, promotion tags | MLflow | One parent run per adapter build; child runs for dataset, train, eval, promotion |
| Dataset bytes | NeMo Platform FileSets plus Data Store compatibility | FileSets are canonical for Customizer; HF-style dataset repos remain for lineage/eval compatibility |
| Dataset entity metadata | NeMo Entity Store | Existing refs use `default/stage3-...`; FileSet URIs use `fileset://default/stage3-...` |
| Training execution | NeMo Platform Customizer | Submit SDK create-job args with `spec.model`, `spec.dataset=fileset://...`, `spec.training`, and optional MLflow integration |
| Evaluation execution | NeMo Evaluator | Existing job endpoint is `/v1/evaluation/jobs` |
| Adapter serving | NIM / deployment layer | Current docs mention `NIM_PEFT_SOURCE` path-based loading |

## Phase 1: Dataset Registration

Input:

- Stage 2 output directory, usually
  `/mnt/nvme2/peft/datasets/v2/<collection>/`
- `training.jsonl`
- `validation.jsonl`
- optional `test_set.jsonl`
- validation reports and bias reports

Steps:

1. Compute content fingerprints for the dataset files.
2. Create or update the NeMo Data Store dataset repo.
3. Push `training.jsonl` and `validation.jsonl`.
4. Register or patch the NeMo Entity Store dataset.
5. Start or update an MLflow child run named `dataset-registration`.
6. Log dataset checksums, row counts, source collection, Data Store URI, and
   Entity Store dataset reference, provenance manifests, and dataset version ID.

Recommended NeMo dataset names:

| Collection | Training dataset entity | Test dataset entity |
|---|---|---|
| `nim_curated` | `default/stage3-nim-curated` | `default/stage3-nim-curated-test` |
| `nemo_usvcs_curated` | `default/stage3-nemo-usvcs-curated` | `default/stage3-nemo-usvcs-curated-test` |

## Phase 2: Platform Handoff and Customizer Training

Input:

- collection,
- base model,
- LoRA rank and alpha,
- dataset entity ref,
- Platform dataset FileSet URI,
- Platform base Model Entity ref,
- MLflow tracking URI.

Steps:

1. Create an MLflow child run named `platform-handoff`.
2. Verify the base Model Entity and model FileSet by running
   `scripts/stage3/platform_model_entities.py --verify` or the
   `deploy/platform-models/` Job. Use `--create-missing` only for deliberate
   first-cluster bootstrap after licenses/secrets are ready.
3. Ensure the dataset FileSet exists by running `scripts/eval/upload_platform_filesets.py`
   or the `deploy/platform-filesets/` Job.
4. Log `platform/model_entities_manifest.json` and
   `platform/filesets_manifest.json` to MLflow.
5. Create an MLflow child run named `customizer-training`.
6. Build the Platform Customizer payload using
   `scripts/stage3/train_adapter.py --payload-format platform`.
4. Include MLflow integration fields in the Platform `spec.integrations` block
   when `MLFLOW_TRACKING_URI` or `MLFLOW_EXPERIMENT_NAME` is configured:

```json
{
  "name": "<adapter-name>",
  "workspace": "default",
  "spec": {
    "model": "default/<model-entity>",
    "dataset": "fileset://default/<dataset-fileset>",
    "training": {
      "type": "sft",
      "peft": {"type": "lora", "rank": 16, "alpha": 32}
    },
    "integrations": {
      "mlflow": {
        "experiment_name": "docs-to-data-to-lora",
        "tracking_uri": "<mlflow-tracking-uri>",
        "tags": {
          "collection": "<collection>",
          "base_model": "<base-model>",
          "rank": "16"
        }
      }
    }
  }
}
```

7. Submit the Customizer job through the NeMo Platform SDK.
8. Log the returned `nemo_customizer_job_id` and
   `nemo_platform_customizer_job_name` to MLflow immediately.
9. Poll the job by name/workspace with the Platform SDK status API.
10. Log `customizer/platform_job_payload.json`,
    `customizer/platform_job_final.json`, output adapter/model references, and
    training metrics available from Customizer or native MLflow export.

Notes:

- Use `--payload-format legacy` only for rollback against standalone Customizer.
- Platform payloads should reference FileSet URIs, not local paths or legacy
  Entity Store dataset strings.
- If Customizer also logs into MLflow directly, the wrapper still logs the
  Customizer job ID/name and parent MLflow run ID so both MLflow runs can be
  reconciled.
- `deploy/customizer-training/` provides Kubernetes submission/polling Jobs for
  dense and MoE examples.

## Phase 3: Evaluator Runs

Input:

- adapter inventory,
- test dataset entity refs,
- Evaluator targets and configs,
- 49B comparator target.

Steps:

1. Create an MLflow child run named `evaluator-matrix`.
2. Register or patch Evaluator targets and configs with
   `scripts/eval/register_evaluator_entities.py`.
3. Submit single-axis and pairwise jobs with
   `scripts/eval/run_evaluation_matrix.py`.
4. Log every Evaluator job ID as an MLflow artifact or tag group.
5. Poll jobs until terminal.
6. Fetch results and log normalized metrics into MLflow.
7. If using the NeMo Evaluator MLflow exporter, record the exported run ID on
   the wrapper run so service-native and repository-native observability can be reconciled.

Minimum metrics to normalize:

| Metric | Meaning |
|---|---|
| `eval.accuracy.mean` | Mean rubric accuracy across dataset rows |
| `eval.completeness.mean` | Mean rubric completeness |
| `eval.faithfulness.mean` | Mean rubric faithfulness |
| `eval.clarity.mean` | Mean rubric clarity |
| `eval.win_rate_vs_base` | Pairwise win rate over no-LoRA base |
| `eval.win_rate_vs_49b` | Pairwise win rate over the 49B comparator |

## Phase 4: Promotion

Promotion is optional in the first pass. When added, keep it explicit:

1. Define acceptance thresholds in config.
2. Compare metrics against the no-LoRA base and 49B comparator.
3. Mark the MLflow run with one of:
   - `promotion_status=candidate`
   - `promotion_status=approved`
   - `promotion_status=rejected`
   - `promotion_status=deployed`
4. If using MLflow Model Registry, register a lightweight model package that
   points to the NeMo adapter entity and serving configuration.
5. If using NeMo/NIM deployment directly, log the deployment manifest and
   `NIM_PEFT_SOURCE` path as MLflow artifacts.

## Phase 5: Event-Driven Automation

After the basic script is reliable, MLflow webhooks can trigger external
automation on registry events. Recommended webhook use:

- model version created -> submit NeMo Evaluator matrix,
- model version alias set to `candidate` -> run promotion checks,
- model version alias set to `production` -> update deployment manifests.

Do not rely on MLflow webhooks as the only source of truth for NeMo job state.
The orchestrator should still poll NeMo services and log terminal status.

## Failure Handling

Each NeMo operation should be idempotent where possible:

- Data Store repo exists: continue.
- Entity Store dataset exists: patch mutable fields, keep stable identity.
- Evaluator target/config exists: continue or patch depending on schema.
- Customizer job exists: do not resubmit unless the same adapter build was
  explicitly marked retryable.

Log enough state in MLflow to resume:

- collection,
- base model,
- rank,
- dataset entity,
- output model entity,
- Customizer job ID,
- Evaluator job IDs,
- adapter artifact path.
