# MLflow Orchestrator Template

This is a template for a future script, not committed implementation code. The
script should live under `scripts/integrations/` when implemented.

Recommended file:

```text
scripts/integrations/mlflow_nemo_orchestrator.py
```

## CLI Shape

```bash
python scripts/integrations/mlflow_nemo_orchestrator.py \
  --config docs/integration-templates/mlflow-nemo/config.example.yaml \
  --collection nim_curated \
  --base-model meta/llama-3.2-1b-instruct \
  --rank 16 \
  --stage all
```

Supported `--stage` values:

| Stage | Meaning |
|---|---|
| `dataset` | Build/register dataset and log dataset lineage |
| `platform-handoff` | Verify Model Entities and FileSets, then log handoff manifests |
| `train` | Submit and track a NeMo Platform Customizer job |
| `eval` | Submit and track Evaluator matrix |
| `promote` | Apply promotion rules and write registry/deployment state |
| `all` | Run dataset, Platform handoff, train, eval, and optional promotion |

## Pseudocode

```python
from __future__ import annotations

import json
from pathlib import Path

import mlflow

from scripts.eval.run_evaluation_matrix import (
    build_49b_pairwise_jobs,
    build_pairwise_jobs,
    build_singleaxis_jobs,
)
from scripts.eval.upload_platform_filesets import PlatformFileSetConfig
from scripts.eval.upload_platform_filesets import run as upload_platform_filesets
from scripts.stage3.customizer_client import CustomizerClient, JobStatus
from scripts.stage3.models import AdapterSpec
from scripts.stage3.platform_customizer import fileset_uri_from_ref, model_entity_for_base
from scripts.stage3.platform_model_entities import PlatformModelEntityConfig
from scripts.stage3.platform_model_entities import run as verify_platform_model_entities
from scripts.stage3.train_adapter import _DATASET_FOR_COLLECTION, build_platform_customizer_payload


def adapter_name(collection: str, base_model: str, rank: int) -> str:
    coll_short = "nim" if collection == "nim_curated" else "nemo-usvcs"
    base_short = base_model.split("/")[-1].replace("-instruct", "")
    return f"lora-{coll_short}-{base_short}-r{rank}"


def run(config: dict, collection: str, base_model: str, rank: int) -> None:
    workspace = config["nemo"].get("workspace", "default")
    name = adapter_name(collection, base_model, rank)
    alpha = 2 * rank
    dataset_entity = _DATASET_FOR_COLLECTION[collection]
    output_model_entity = f"{workspace}/{name}"
    model_entity = config["base_models"][base_model].get(
        "model_entity",
        model_entity_for_base(base_model, workspace),
    )
    dataset_fileset_uri = config["collections"][collection].get(
        "train_fileset_uri",
        fileset_uri_from_ref(dataset_entity, default_workspace=workspace),
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"{collection}/{base_model.split('/')[-1]}/r{rank}") as parent:
        mlflow.set_tags({
            "pipeline": "docs-to-data-to-lora",
            "collection": collection,
            "base_model": base_model,
            "adapter_name": name,
            "dataset_entity": dataset_entity,
            "output_model_entity": output_model_entity,
            "nemo_platform_workspace": workspace,
            "nemo_model_entity": model_entity,
            "nemo_dataset_fileset_uri": dataset_fileset_uri,
        })
        mlflow.log_params({
            "lora.rank": rank,
            "lora.alpha": alpha,
            "customizer.payload_format": "platform",
            "train.epochs": config["training_defaults"]["epochs"],
            "train.batch_size": config["training_defaults"]["batch_size"],
            "train.learning_rate": config["training_defaults"]["learning_rate"],
        })

        with mlflow.start_run(run_name="platform-handoff", nested=True):
            model_manifest = verify_platform_model_entities(
                PlatformModelEntityConfig(
                    workspace=workspace,
                    nmp_base_url=config["nemo"]["platform_base_url"],
                    models=(base_model,),
                    verify=True,
                    out=Path(config["nemo"]["model_entities_manifest"]),
                )
            )
            fileset_manifest = upload_platform_filesets(
                PlatformFileSetConfig(
                    base_dir=Path(config["collections"][collection]["stage2_dir"]).parent,
                    collections=[collection],
                    workspace=workspace,
                    nmp_base_url=config["nemo"]["platform_base_url"],
                    observability_dir=Path(config["nemo"]["filesets_observability_dir"]),
                    include_train=True,
                    include_test=True,
                    include_context_test=True,
                    include_lineage=True,
                    dry_run=False,
                )
            )
            mlflow.log_dict(model_manifest, "platform/model_entities_manifest.json")
            mlflow.log_dict(fileset_manifest, "platform/filesets_manifest.json")

        with mlflow.start_run(run_name="customizer-training", nested=True):
            spec = AdapterSpec(
                adapter_name=name,
                collection=collection,
                base_model=base_model,
                rank=rank,
                alpha=alpha,
            )
            payload = build_platform_customizer_payload(
                spec=spec,
                workspace=workspace,
                dataset_entity=dataset_entity,
                output_model_entity=output_model_entity,
                description=f"MLflow tracked Platform Customizer build for {name}",
                model_entity=model_entity,
                dataset_fileset_uri=dataset_fileset_uri,
                batch_size=config["training_defaults"]["batch_size"],
                epochs=config["training_defaults"]["epochs"],
                learning_rate=config["training_defaults"]["learning_rate"],
                mlflow_tracking_uri=config["mlflow"]["tracking_uri"],
                mlflow_experiment_name=config["mlflow"]["experiment_name"],
                mlflow_run_name=name,
            )
            mlflow_tags = (
                payload["spec"]
                .setdefault("integrations", {})
                .setdefault("mlflow", {})
                .setdefault("tags", {})
            )
            mlflow_tags["mlflow_parent_run_id"] = parent.info.run_id

            mlflow.log_dict(payload, "customizer/platform_job_payload.json")
            with CustomizerClient(config["nemo"]["customizer_url"]) as client:
                job_id = client.submit_platform_job(
                    name=payload["name"],
                    workspace=payload["workspace"],
                    spec=payload["spec"],
                )
                mlflow.set_tag("nemo_customizer_job_id", job_id)
                mlflow.set_tag("nemo_platform_customizer_job_name", payload["name"])
                status = client.wait_platform_until_done(
                    name=payload["name"],
                    workspace=payload["workspace"],
                    poll_interval_s=config["polling"]["customizer_interval_s"],
                    timeout_s=config["polling"]["customizer_timeout_s"],
                )
                final_detail = client.get_platform_status_detail(
                    name=payload["name"],
                    workspace=payload["workspace"],
                )
                mlflow.log_dict(final_detail, "customizer/platform_job_final.json")
                mlflow.set_tag("nemo_customizer_status", status.value)
                if status != JobStatus.COMPLETED:
                    raise RuntimeError(f"Customizer job {job_id} ended as {status.value}")

        with mlflow.start_run(run_name="evaluator-matrix", nested=True):
            # Register Evaluator entities if needed.
            # Submit jobs with EvaluatorClient or the Platform Evaluator SDK path
            # selected for this repo.
            # Log payloads, IDs, exported MLflow run IDs, raw results, and metrics.
            pass
```

## Implementation Notes

- Keep NeMo wrappers thin and prefer the Platform SDK for Customizer jobs.
- Treat Model Entity and FileSet manifests as first-class MLflow artifacts.
- Preserve legacy payload support only as a rollback path; new orchestration
  should use `build_platform_customizer_payload`.
- If Customizer and Evaluator export directly to MLflow, log their native run IDs
  on the wrapper run so service-native and repository-native observability can
  be reconciled.
- Store exact request and final status bodies as artifacts so schema drift is
  easy to diagnose.
- Do not log secrets, API keys, Gitea passwords, bearer tokens, or private
  endpoint credentials.

## Suggested Tests

| Test | Assertion |
|---|---|
| `test_adapter_name_nim` | `nim_curated` maps to `lora-nim-...` |
| `test_adapter_name_nemo_usvcs` | `nemo_usvcs_curated` maps to `lora-nemo-usvcs-...` |
| `test_platform_payload_includes_mlflow` | Payload has `spec.integrations.mlflow` and preserves model/FileSet refs |
| `test_platform_handoff_logs_manifests` | Model Entity and FileSet manifests are logged as MLflow artifacts |
| `test_resume_from_platform_job_name` | Existing Platform job name prevents duplicate Customizer submission |
| `test_no_secret_logging` | Config secrets are redacted before artifacts are logged |
