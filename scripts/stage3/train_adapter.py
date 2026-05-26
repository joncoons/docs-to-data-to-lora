"""Stage 3 train-adapter CLI: submit one Customizer LoRA job per AdapterSpec."""
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
from scripts.stage3.models import AdapterSpec

log = logging.getLogger(__name__)

# Map collection → dataset entity already registered in entity-store.
_DATASET_FOR_COLLECTION = {
    "nim_curated": "default/stage3-nim-curated",
    "nemo_usvcs_curated": "default/stage3-nemo-usvcs-curated",
}

# Map base model slug → NeMo Customizer config template ref.
_TEMPLATE_FOR_BASE = {
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
    """Build the Customizer 25.12 job submission payload from an AdapterSpec.

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
        "--customizer-url",
        default="http://192.168.1.187:30910",
        help="Customizer REST endpoint (NodePort default)",
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

    if args.dry_run:
        cfg = build_customizer_config(
            spec,
            _TEMPLATE_FOR_BASE[args.base_model],
            dataset_entity,
            output_model_entity,
            description,
        )
        print(json.dumps(cfg, indent=2))
        return 0

    with CustomizerClient(args.customizer_url) as client:
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
