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
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.stage3.customizer_client import CustomizerClient, JobStatus
from scripts.stage3.moe_models import MoEAdapterSpec

log = logging.getLogger(__name__)

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
    """Build the Customizer 25.12 job submission payload from a MoEAdapterSpec.

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
        "--dataset-entity",
        help="Override the registered Customizer shard dataset entity.",
    )
    ap.add_argument(
        "--adapter-name",
        help="Override the generated shard adapter/output model name.",
    )
    ap.add_argument(
        "--output-model-entity",
        help="Override the full Customizer output model entity. Defaults to default/<adapter-name>.",
    )
    ap.add_argument(
        "--description",
        help="Override the Customizer job description.",
    )
    ap.add_argument(
        "--customizer-url",
        default="http://192.168.1.187:30910",
        help="Customizer REST endpoint (NodePort default)",
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
    adapter_name = args.adapter_name or f"lora-{coll_short}-nemotron-nano-30b-r{rank}-shard-{shard}"

    spec = MoEAdapterSpec(
        adapter_name=adapter_name,
        collection=collection,
        base_model=NANO_BASE_MODEL,
        rank=rank,
        alpha=alpha,
        shard=shard,
    )

    dataset_key = (collection, shard)
    dataset_entity = args.dataset_entity or _SHARD_DATASET_FOR[dataset_key]
    output_model_entity = args.output_model_entity or f"default/{adapter_name}"
    description = args.description or (
        f"Stage 3 MoE — {coll_short} × nemotron-nano-30b r{rank} shard-{shard}"
    )

    if args.dry_run:
        cfg = build_customizer_config_moe(
            spec,
            NANO_CONFIG_TEMPLATE,
            dataset_entity,
            output_model_entity,
            description,
        )
        print(json.dumps(cfg, indent=2))
        return 0

    with CustomizerClient(args.customizer_url) as client:
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
