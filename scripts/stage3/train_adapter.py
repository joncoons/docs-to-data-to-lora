"""Stage 3 train-adapter CLI: submit one Customizer LoRA job per AdapterSpec."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.nemo_platform import default_customizer_url, default_nmp_workspace  # noqa: E402
from scripts.stage3.customizer_client import CustomizerClient, JobStatus  # noqa: E402
from scripts.stage3.platform_customizer import (  # noqa: E402
    DEFAULT_LORA_DROPOUT,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_PLATFORM_PAYLOAD_FORMAT,
    build_lora_training_spec,
    build_mlflow_integration,
    build_platform_customizer_job,
    build_platform_customizer_spec,
    fileset_uri_from_ref,
    model_entity_for_base,
)
from scripts.stage3.models import AdapterSpec  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CUSTOMIZER_URL = default_customizer_url()
DEFAULT_WORKSPACE = default_nmp_workspace()

# Map collection → dataset entity already registered in entity-store.
_DATASET_FOR_COLLECTION = {
    "nim_curated": "default/stage3-nim-curated",
    "nemo_usvcs_curated": "default/stage3-nemo-usvcs-curated",
}

# Map base model slug → NeMo Customizer config template ref.
_TEMPLATE_FOR_BASE = {
    "meta/llama-3.2-1b-instruct": "meta/llama-3.2-1b-instruct@v1.0.0+80GB",
    "meta/llama-3.2-3b-instruct": "meta/llama-3.2-3b-instruct@v1.0.0+80GB",
    "meta/llama-3.1-8b-instruct": "meta/llama-3.1-8b-instruct@v1.0.0+80GB",
}


def build_customizer_config(
    spec: AdapterSpec,
    base_template: str,
    dataset_entity: str,
    output_model_entity: str,
    description: str,
) -> dict:
    """Build the legacy-compatible Customizer job payload from an AdapterSpec.

    Shape is cross-validated against known-good completed job
    cust-KwJYTEBNqXoi71d4RQoTk5 (Llama-3.1-8B LoRA SFT, May 2026).
    """
    return {
        "description": description,
        "dataset": dataset_entity,
        "output_model": output_model_entity,
        "config": base_template,
        "hyperparameters": {
            "finetuning_type": "lora",
            "training_type": "sft",
            "warmup_steps": 30,
            "seed": 42,
            "max_steps": -1,
            "optimizer": "adamw_with_cosine_annealing",
            "adam_beta1": 0.9,
            "adam_beta2": 0.99,
            "batch_size": 16,
            "epochs": 2,
            "learning_rate": 1.0e-4,
            "log_every_n_steps": 10,
            "lora": {
                "adapter_dim": spec.rank,
                "alpha": spec.alpha,
                "adapter_dropout": None,
                "target_modules": None,  # use Customizer defaults per base model
            },
            "sequence_packing_enabled": False,  # unsupported on Blackwell sm_120
        },
    }


def submit_adapter_job(
    spec: AdapterSpec,
    base_template: str,
    dataset_entity: str,
    output_model_entity: str,
    description: str,
    client: CustomizerClient,
) -> str:
    """Build config and POST to Customizer; returns the job_id string."""
    cfg = build_customizer_config(
        spec, base_template, dataset_entity, output_model_entity, description
    )
    log.info(
        "Submitting adapter %s on template %s",
        spec.adapter_name,
        base_template,
    )
    return client.submit_job(cfg)


def build_platform_customizer_payload(
    spec: AdapterSpec,
    *,
    workspace: str,
    dataset_entity: str,
    output_model_entity: str,
    description: str,
    model_entity: str | None = None,
    dataset_fileset_uri: str | None = None,
    batch_size: int = 16,
    epochs: int = 2,
    learning_rate: float = 1.0e-4,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
) -> dict:
    """Build NeMo Platform SDK create-job args for this adapter."""
    output_workspace, output_name = output_model_entity.split("/", 1)
    job_workspace = workspace or output_workspace
    training = build_lora_training_spec(
        rank=spec.rank,
        alpha=spec.alpha,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        dropout=lora_dropout,
    )
    integrations = build_mlflow_integration(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name,
        run_name=mlflow_run_name,
        description=description,
        tags={
            "adapter_name": spec.adapter_name,
            "collection": spec.collection,
            "base_model": spec.base_model,
            "rank": str(spec.rank),
            "stage": "stage3",
        },
    )
    platform_spec = build_platform_customizer_spec(
        model_entity=model_entity or model_entity_for_base(spec.base_model, job_workspace),
        dataset_fileset_uri=dataset_fileset_uri
        or fileset_uri_from_ref(dataset_entity, default_workspace=job_workspace),
        output_name=output_name,
        training=training,
        integrations=integrations,
    )
    return build_platform_customizer_job(
        name=spec.adapter_name,
        workspace=job_workspace,
        spec=platform_spec,
    )


def submit_adapter_job_platform(
    spec: AdapterSpec,
    *,
    workspace: str,
    dataset_entity: str,
    output_model_entity: str,
    description: str,
    client: CustomizerClient,
    model_entity: str | None = None,
    dataset_fileset_uri: str | None = None,
) -> str:
    """Build a Platform spec and create the Customizer job through the SDK."""
    payload = build_platform_customizer_payload(
        spec,
        workspace=workspace,
        dataset_entity=dataset_entity,
        output_model_entity=output_model_entity,
        description=description,
        model_entity=model_entity,
        dataset_fileset_uri=dataset_fileset_uri,
    )
    log.info(
        "Submitting Platform adapter %s on model %s",
        spec.adapter_name,
        payload["spec"]["model"],
    )
    return client.submit_platform_job(
        name=payload["name"],
        workspace=payload["workspace"],
        spec=payload["spec"],
    )


# -- CLI -----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit one Customizer LoRA adapter job.")
    ap.add_argument(
        "--collection",
        required=True,
        choices=list(_DATASET_FOR_COLLECTION.keys()),
    )
    ap.add_argument(
        "--base-model",
        required=True,
        choices=list(_TEMPLATE_FOR_BASE.keys()),
    )
    ap.add_argument("--rank", required=True, type=int, choices=[16, 32])
    ap.add_argument(
        "--payload-format",
        choices=["platform", "legacy"],
        default=DEFAULT_PLATFORM_PAYLOAD_FORMAT,
        help="Dry-run/submit payload shape. Defaults to CUSTOMIZER_PAYLOAD_FORMAT or platform.",
    )
    ap.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    ap.add_argument("--model-entity", default=None)
    ap.add_argument("--dataset-fileset-uri", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1.0e-4)
    ap.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    ap.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    ap.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    ap.add_argument("--mlflow-experiment-name", default=os.getenv("MLFLOW_EXPERIMENT_NAME"))
    ap.add_argument("--mlflow-run-name", default=os.getenv("MLFLOW_RUN_NAME"))
    ap.add_argument(
        "--customizer-url",
        default=DEFAULT_CUSTOMIZER_URL,
        help=(
            "Customizer or NeMo Platform API base URL. Defaults to CUSTOMIZER_URL, "
            "CUSTOMIZER_BASE_URL, NMP_CUSTOMIZER_URL, NMP_BASE_URL, or "
            "http://localhost:8080."
        ),
    )
    ap.add_argument("--wait", action="store_true", help="Block until job terminates")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rank = args.rank
    alpha = 2 * rank
    base_short = args.base_model.split("/")[-1].replace("-instruct", "")
    coll_short = "nim" if args.collection == "nim_curated" else "nemo-usvcs"
    adapter_name = f"lora-{coll_short}-{base_short}-r{rank}"

    spec = AdapterSpec(
        adapter_name=adapter_name,
        collection=args.collection,
        base_model=args.base_model,
        rank=rank,
        alpha=alpha,
    )

    dataset_entity = _DATASET_FOR_COLLECTION[args.collection]
    output_model_entity = f"default/{adapter_name}"
    description = (
        f"Stage 3 — {coll_short} × {base_short} LoRA r{rank}"
    )

    if args.payload_format == "platform":
        payload = build_platform_customizer_payload(
            spec,
            workspace=args.workspace,
            dataset_entity=dataset_entity,
            output_model_entity=output_model_entity,
            description=description,
            model_entity=args.model_entity,
            dataset_fileset_uri=args.dataset_fileset_uri,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            lora_dropout=args.lora_dropout,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name,
        )
    else:
        payload = build_customizer_config(
            spec,
            _TEMPLATE_FOR_BASE[args.base_model],
            dataset_entity,
            output_model_entity,
            description,
        )

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    with CustomizerClient(args.customizer_url) as client:
        if args.payload_format == "platform":
            job_id = client.submit_platform_job(
                name=payload["name"],
                workspace=payload["workspace"],
                spec=payload["spec"],
            )
        else:
            job_id = submit_adapter_job(
                spec,
                _TEMPLATE_FOR_BASE[args.base_model],
                dataset_entity=dataset_entity,
                output_model_entity=output_model_entity,
                description=description,
                client=client,
            )
        log.info(
            "Submitted: job_id=%s adapter=%s output_model=%s",
            job_id,
            adapter_name,
            output_model_entity,
        )
        print(job_id)
        if args.wait:
            terminal = client.wait_until_done(job_id)
            log.info("Terminal status: %s", terminal.value)
            return 0 if terminal == JobStatus.COMPLETED else 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
