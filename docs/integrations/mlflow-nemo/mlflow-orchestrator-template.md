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
  --config docs/integrations/mlflow-nemo/config.example.yaml \
  --collection nim_curated \
  --base-model meta/llama-3.2-1b-instruct \
  --rank 16 \
  --stage all
```

Supported `--stage` values:

| Stage | Meaning |
|---|---|
| `dataset` | Build/register dataset and log dataset lineage |
| `train` | Submit and track Customizer job |
| `eval` | Submit and track Evaluator matrix |
| `promote` | Apply promotion rules and write registry/deployment state |
| `all` | Run dataset, train, eval, and optional promotion |

## Pseudocode

```python
from __future__ import annotations

import json
from pathlib import Path

import mlflow

from scripts.stage3.train_adapter import (
    _DATASET_FOR_COLLECTION,
    _TEMPLATE_FOR_BASE,
    build_customizer_config,
)
from scripts.stage3.customizer_client import CustomizerClient, JobStatus
from scripts.stage3.models import AdapterSpec
from scripts.eval.evaluator_client import EvaluatorClient
from scripts.eval.run_evaluation_matrix import (
    build_reference_pairwise_jobs,
    build_pairwise_jobs,
    build_singleaxis_jobs,
)


def adapter_name(collection: str, base_model: str, rank: int) -> str:
    coll_short = "nim" if collection == "nim_curated" else "nemo-usvcs"
    base_short = base_model.split("/")[-1].replace("-instruct", "")
    return f"lora-{coll_short}-{base_short}-r{rank}"


def run(config: dict, collection: str, base_model: str, rank: int) -> None:
    name = adapter_name(collection, base_model, rank)
    alpha = 2 * rank
    dataset_entity = _DATASET_FOR_COLLECTION[collection]
    output_model_entity = f"default/{name}"
    base_template = _TEMPLATE_FOR_BASE[base_model]

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
        })
        mlflow.log_params({
            "lora.rank": rank,
            "lora.alpha": alpha,
            "customizer.config_template": base_template,
        })

        # Dataset child run:
        # - call or reuse Data Store / Entity Store registration code
        # - log manifests, checksums, row counts, and NeMo dataset refs

        with mlflow.start_run(run_name="customizer-training", nested=True):
            spec = AdapterSpec(
                adapter_name=name,
                collection=collection,
                base_model=base_model,
                rank=rank,
                alpha=alpha,
            )
            payload = build_customizer_config(
                spec=spec,
                base_template=base_template,
                dataset_entity=dataset_entity,
                output_model_entity=output_model_entity,
                description=f"MLflow tracked build for {name}",
            )
            payload.setdefault("integrations", {})["mlflow"] = {
                "experiment_name": config["mlflow"]["experiment_name"],
                "tracking_uri": config["mlflow"]["tracking_uri"],
                "run_name": name,
                "tags": {
                    "mlflow_parent_run_id": parent.info.run_id,
                    "collection": collection,
                    "dataset_entity": dataset_entity,
                    "output_model_entity": output_model_entity,
                },
            }

            mlflow.log_dict(payload, "customizer/job_payload.json")
            with CustomizerClient(config["nemo"]["customizer_url"]) as client:
                job_id = client.submit_job(payload)
                mlflow.set_tag("nemo_customizer_job_id", job_id)
                status = client.wait_until_done(
                    job_id,
                    poll_interval_s=config["polling"]["customizer_interval_s"],
                    timeout_s=config["polling"]["customizer_timeout_s"],
                )
                mlflow.set_tag("nemo_customizer_status", status.value)
                mlflow.set_tag("nemo_output_path", client.get_output_path(job_id) or "")
                if status != JobStatus.COMPLETED:
                    raise RuntimeError(f"Customizer job {job_id} ended as {status.value}")

        with mlflow.start_run(run_name="evaluator-matrix", nested=True):
            # Register Evaluator entities if needed.
            # Submit jobs with EvaluatorClient.
            # Log job payloads, IDs, raw results, and normalized metrics.
            pass
```

## Implementation Notes

- Keep direct NeMo REST wrappers thin, matching the existing client style.
- Prefer importing pure builders from existing scripts over duplicating payload
  logic.
- Treat MLflow logging as additive: the workflow should still be debuggable
  from NeMo job IDs alone.
- Store exact request and response bodies as artifacts so schema drift is easy
  to diagnose.
- Do not log secrets, API keys, Gitea passwords, bearer tokens, or private
  endpoint credentials.

## Suggested Tests

| Test | Assertion |
|---|---|
| `test_adapter_name_nim` | `nim_curated` maps to `lora-nim-...` |
| `test_adapter_name_nemo_usvcs` | `nemo_usvcs_curated` maps to `lora-nemo-usvcs-...` |
| `test_customizer_payload_includes_mlflow` | Wrapper adds `integrations.mlflow` without changing dataset/output model refs |
| `test_dataset_metadata_contract` | Required tags and params are emitted |
| `test_resume_from_customizer_job_id` | Existing job ID prevents duplicate Customizer submission |
| `test_no_secret_logging` | Config secrets are redacted before artifacts are logged |
