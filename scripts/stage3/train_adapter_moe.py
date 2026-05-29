"""Stage 3 MoE train-adapter CLI: submit one Customizer LoRA job per shard.

Nemotron-3-Nano-30B-A3B specific — α/r=1.0, batch_size=8, warmup_steps=100,
sequence_packing_enabled=false (MoE + Blackwell sm_120 constraint).

See NEMOTRON_NANO_LORA_METHODOLOGY.md for the full rationale behind these
hyperparameters and the singleGPU+TIES approach.

Does NOT inherit from or modify scripts/stage3/train_adapter.py (dense Llama path).

Usage:
  python3 scripts/stage3/train_adapter_moe.py \\
      --collection nim_curated --rank 16 --shard a [--dry-run] [--wait]
"""
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
from scripts.stage3.moe_models import MoEAdapterSpec  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CUSTOMIZER_URL = default_customizer_url()
DEFAULT_WORKSPACE = default_nmp_workspace()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NANO_BASE_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
NANO_CONFIG_TEMPLATE = "nvidia/nemotron-3-nano-30b-a3b@v1.0+96GB-singleGPU"

# Maps (collection, shard) → fully-qualified entity-store dataset ref.
# These must exist in entity-store before submitting (created by build_moe_shards.py).
_SHARD_DATASET_FOR: dict[tuple[str, str], str] = {
    ("nim_curated", "a"):        "default/stage3-nim-curated-shard-a",
    ("nim_curated", "b"):        "default/stage3-nim-curated-shard-b",
    ("nemo_usvcs_curated", "a"): "default/stage3-nemo-usvcs-curated-shard-a",
    ("nemo_usvcs_curated", "b"): "default/stage3-nemo-usvcs-curated-shard-b",
}

# Short corpus labels used in adapter naming and description
_CORPUS_SHORT = {
    "nim_curated": "nim",
    "nemo_usvcs_curated": "nemo-usvcs",
}


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_customizer_config_moe(
    spec: MoEAdapterSpec,
    base_template: str,
    dataset_entity: str,
    output_model_entity: str,
    description: str,
) -> dict:
    """Build the legacy-compatible Customizer job payload from a MoEAdapterSpec.

    Key MoE-specific hyperparameters (validated methodology):
      - α = rank (α/r = 1.0) — dense default α=2r diverges at end-of-warmup
      - batch_size = 8 (not 16 as in dense path)
      - warmup_steps = 20 — Customizer 25.12 enforces
        warmup_steps < lr_decay_steps = epochs * (N/batch_size) // grad_acc(8).
        Stage 3 MoE shards yield 62 (nemo-usvcs) or 72 (nim) optimizer steps
        over 2 epochs, so 20 is the largest uniform value that fits with margin.
      - sequence_packing_enabled = false (MoE + Blackwell sm_120 constraint)
    """
    return {
        "description": description,
        "dataset": dataset_entity,
        "output_model": output_model_entity,
        "config": base_template,
        "hyperparameters": {
            "finetuning_type": "lora",
            "training_type": "sft",
            "warmup_steps": 20,
            "seed": 42,
            "max_steps": -1,
            "optimizer": "adamw_with_cosine_annealing",
            "adam_beta1": 0.9,
            "adam_beta2": 0.99,
            "batch_size": 8,              # MoE: 8 (dense: 16)
            "epochs": 2,
            "learning_rate": 1.0e-4,
            "log_every_n_steps": 10,
            "lora": {
                "adapter_dim": spec.rank,
                "alpha": spec.alpha,      # MoEAdapterSpec guarantees alpha == rank
                "adapter_dropout": None,
                "target_modules": None,   # use Customizer defaults per base model
            },
            "sequence_packing_enabled": False,  # MoE + sm_120 constraint
        },
    }


# ---------------------------------------------------------------------------
# Job submitter
# ---------------------------------------------------------------------------

def submit_adapter_job_moe(
    spec: MoEAdapterSpec,
    base_template: str,
    dataset_entity: str,
    output_model_entity: str,
    description: str,
    client: CustomizerClient,
) -> str:
    """Build config and POST to Customizer; returns the job_id string."""
    cfg = build_customizer_config_moe(
        spec, base_template, dataset_entity, output_model_entity, description
    )
    log.info(
        "Submitting MoE adapter %s on template %s shard=%s",
        spec.adapter_name,
        base_template,
        spec.shard,
    )
    return client.submit_job(cfg)


def build_platform_customizer_payload_moe(
    spec: MoEAdapterSpec,
    *,
    workspace: str,
    dataset_entity: str,
    output_model_entity: str,
    description: str,
    model_entity: str | None = None,
    dataset_fileset_uri: str | None = None,
    batch_size: int = 8,
    epochs: int = 2,
    learning_rate: float = 1.0e-4,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
) -> dict:
    """Build NeMo Platform SDK create-job args for this MoE shard."""
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
            "shard": spec.shard,
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


def submit_adapter_job_moe_platform(
    spec: MoEAdapterSpec,
    *,
    workspace: str,
    dataset_entity: str,
    output_model_entity: str,
    description: str,
    client: CustomizerClient,
    model_entity: str | None = None,
    dataset_fileset_uri: str | None = None,
) -> str:
    """Build a Platform spec and create the MoE Customizer job through the SDK."""
    payload = build_platform_customizer_payload_moe(
        spec,
        workspace=workspace,
        dataset_entity=dataset_entity,
        output_model_entity=output_model_entity,
        description=description,
        model_entity=model_entity,
        dataset_fileset_uri=dataset_fileset_uri,
    )
    log.info(
        "Submitting Platform MoE adapter %s on model %s shard=%s",
        spec.adapter_name,
        payload["spec"]["model"],
        spec.shard,
    )
    return client.submit_platform_job(
        name=payload["name"],
        workspace=payload["workspace"],
        spec=payload["spec"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Submit one Customizer LoRA job for Nemotron-3-Nano-30B-A3B (MoE)."
    )
    ap.add_argument(
        "--collection",
        required=True,
        choices=list(_CORPUS_SHORT.keys()),
    )
    ap.add_argument("--rank", required=True, type=int, choices=[16, 32])
    ap.add_argument("--shard", required=True, choices=["a", "b"])
    ap.add_argument(
        "--payload-format",
        choices=["platform", "legacy"],
        default=DEFAULT_PLATFORM_PAYLOAD_FORMAT,
        help="Dry-run/submit payload shape. Defaults to CUSTOMIZER_PAYLOAD_FORMAT or platform.",
    )
    ap.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    ap.add_argument("--model-entity", default=None)
    ap.add_argument("--dataset-fileset-uri", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
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
    alpha = rank                     # α/r = 1.0 for MoE
    shard = args.shard
    collection = args.collection
    coll_short = _CORPUS_SHORT[collection]
    adapter_name = f"lora-{coll_short}-nemotron-nano-30b-r{rank}-shard-{shard}"

    spec = MoEAdapterSpec(
        adapter_name=adapter_name,
        collection=collection,
        base_model=NANO_BASE_MODEL,
        rank=rank,
        alpha=alpha,
        shard=shard,
    )

    dataset_key = (collection, shard)
    dataset_entity = _SHARD_DATASET_FOR[dataset_key]
    output_model_entity = f"default/{adapter_name}"
    description = f"Stage 3 MoE — {coll_short} × nemotron-nano-30b r{rank} shard-{shard}"

    if args.payload_format == "platform":
        payload = build_platform_customizer_payload_moe(
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
        payload = build_customizer_config_moe(
            spec,
            NANO_CONFIG_TEMPLATE,
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
            job_id = submit_adapter_job_moe(
                spec,
                NANO_CONFIG_TEMPLATE,
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
