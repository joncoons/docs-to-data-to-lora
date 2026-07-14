# Integration Plan

## Objective

Create a repeatable adapter-training workflow where MLflow records lineage and
promotion state, and NeMo Microservices perform the actual work:

- NeMo Data Store stores dataset files.
- NeMo Entity Store registers datasets and model entities.
- NeMo Customizer trains LoRA adapters.
- NeMo Evaluator scores adapters, bases, and the 49B comparator target.
- MLflow links every step with a stable run graph.

## Control Plane Boundary

| Concern | Owner | Notes |
|---|---|---|
| Run identity, experiment grouping, lineage, promotion tags | MLflow | One parent run per adapter build; child runs for dataset, train, eval, promotion |
| Dataset bytes | NeMo Data Store | Existing scripts use HF-style dataset repos |
| Dataset entity metadata | NeMo Entity Store | Existing refs use `default/stage3-...` |
| Training execution | NeMo Customizer | Existing job endpoint is `/v1/customization/jobs` |
| Evaluation execution | NeMo Evaluator | Existing job endpoint is `/v1/evaluation/jobs` |
| Adapter serving | NIM / deployment layer | Current docs mention `NIM_PEFT_SOURCE` path-based loading |

## Phase 1: Dataset Registration

Input:

- Stage 2 output directory, usually
  `<DATASET_ROOT>/<collection>/`
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

## Phase 2: Customizer Training

Input:

- collection,
- base model,
- LoRA rank and alpha,
- dataset entity ref,
- Customizer config template ref,
- MLflow tracking URI.

Steps:

1. Create an MLflow child run named `customizer-training`.
2. Build the Customizer payload using `scripts/stage3/train_adapter.py`.
3. Add MLflow integration fields when the target Customizer version supports
   them:

```json
{
  "integrations": {
    "mlflow": {
      "experiment_name": "docs-to-data-to-lora",
      "run_name": "<adapter-name>",
      "tracking_uri": "<mlflow-tracking-uri>",
      "tags": {
        "collection": "<collection>",
        "dataset_entity": "<namespace/name>",
        "output_model_entity": "<namespace/name>"
      }
    }
  }
}
```

4. Submit the Customizer job.
5. Log the returned `nemo_customizer_job_id` to MLflow immediately.
6. Poll the job until terminal.
7. Log final status, output path, output model entity, and training metrics
   available from the Customizer job detail endpoint.

Notes:

- Keep the existing `sequence_packing_enabled: false` behavior for Blackwell
  compatibility.
- Keep the current dataset and output model references as Entity Store strings,
  not local paths.
- If Customizer also logs into MLflow directly, the wrapper still logs the
  Customizer job ID so both MLflow runs can be reconciled.

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
