# MLflow + NeMo Integration Notes

This section documents the MLflow surfaces that exist in this repo today and a
future extension path for teams that want MLflow to become the parent control-plane
record for NeMo Data Store, Entity Store, Customizer, and Evaluator activity.

The current implementation does not ship a single MLflow orchestrator command.
Deployable repo support is limited to MLflow-ready observability files for
dataset registration and MLflow export helpers for evaluation artifacts and
metrics. The orchestration plan and skeleton are optional future-extension
guidance, not completed runtime code.

## Deployable Now

| Area | Deployable repo support | Evidence |
|---|---|---|
| Dataset registration | `scripts/eval/upload_test_datasets.py` registers datasets in NeMo Data Store / Entity Store and emits MLflow-ready observability JSON with tracking URI, experiment name, and parent run ID. It does not log directly to MLflow. | `tests/test_dataset_registration.py` covers MLflow parent-run metadata and observability output. |
| Evaluation export | `scripts/eval/mlflow_export.py` logs evaluation tags, params, metrics, and artifact directories to MLflow. Direct single-axis and pairwise evaluators call this helper; `scripts/eval/run_nemo_evaluator_saved_responses.py` has equivalent batch export support. | `tests/test_mlflow_export.py` covers the helper behavior; direct evaluator tests cover scoring outputs that feed export. |
| Stage observability | Stage outputs include run context, metrics, service refs, and artifact manifests that can be uploaded to MLflow by a later export step. | Stage 0 and Stage 1A provenance tests assert MLflow-ready run context fields. |

## Future Extension Path

The future target architecture is a parent MLflow run per adapter build with
child records for dataset registration, Customizer training, Evaluator scoring,
and optional promotion. NeMo remains authoritative for dataset bytes, Customizer job
execution, Evaluator job execution, and adapter artifacts; MLflow records the
cross-service lineage.

The active training script, `scripts/stage3/train_adapter.py`, builds and
submits Customizer jobs today, but it does not currently add an
`integrations.mlflow` block to the Customizer payload. The
[`mlflow-orchestrator-template.md`](mlflow-orchestrator-template.md) skeleton
shows how a wrapper could add that block when the deployed Customizer version
supports it, log the Customizer job ID, and reconcile NeMo-native and
repo-native observability.

## Contents

| File | Use |
|---|---|
| [integration-plan.md](integration-plan.md) | Current/future architecture, phases, and ownership boundaries |
| [metadata-contract.md](metadata-contract.md) | Current and target metadata fields for MLflow and NeMo cross-references |
| [mlflow-orchestrator-template.md](mlflow-orchestrator-template.md) | Future Python orchestration skeleton and run layout |
| [config.example.yaml](config.example.yaml) | Example config shape for the future wrapper |
