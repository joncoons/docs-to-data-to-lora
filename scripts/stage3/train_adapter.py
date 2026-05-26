"""Stage 3 train-adapter CLI: submit one Customizer LoRA job per AdapterSpec."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from scripts.stage3.customizer_client import CustomizerClient, JobStatus
from scripts.stage3.models import AdapterSpec

log = logging.getLogger(__name__)


def build_customizer_config(
    spec: AdapterSpec,
    base_template: str,
    adapter_train_path: str,
    adapter_val_path: str,
    output_path: str,
) -> dict:
    """Build a Customizer 25.12 job config from an AdapterSpec.

    The exact field names below should be cross-checked against the cluster's
    Customizer schema (verifiable via `kubectl get cm -n nemo-peft
    nemo-platform-customizer-config -o yaml | grep training_options`). If field
    names differ, adjust here — the AdapterSpec input shape stays stable.
    """
    return {
        "config": base_template,
        "name": spec.adapter_name,
        "dataset": {
            "train_file": adapter_train_path,
            "validation_file": adapter_val_path,
            "format": "completion",  # NeMo Customizer SFT JSONL convention
        },
        "hyperparameters": {
            "training_type": "sft",
            "finetuning_type": "lora",
            "epochs": 2,
            "learning_rate": 1.0e-4,
            "warmup_ratio": 0.03,
            "optimizer": "adamw",
            "precision": "bf16",
            "micro_batch_size": 1,
            "global_batch_size": 16,
            "sequence_packing_enabled": False,  # Blackwell sm_120
            "val_check_interval": 0.25,
            "save_top_k": 3,
            "checkpoint_metric": "val_loss",
            "checkpoint_mode": "min",
            "early_stopping": {
                "enabled": True,
                "patience": 3,
                "min_delta": 0.005,
            },
            "lora": {
                "adapter_dim": spec.rank,
                "alpha": spec.alpha,
                "dropout": 0.0,
                "target_modules": list(spec.target_modules),
            },
        },
        "output_model_path": output_path,
        "output_format": "huggingface_peft",
    }


def submit_adapter_job(
    spec: AdapterSpec,
    base_template: str,
    adapter_train_path: str,
    adapter_val_path: str,
    output_path: str,
    client: CustomizerClient,
) -> str:
    cfg = build_customizer_config(spec, base_template, adapter_train_path,
                                    adapter_val_path, output_path)
    log.info("Submitting adapter %s on template %s", spec.adapter_name, base_template)
    return client.submit_job(cfg)


# -- CLI -----------------------------------------------------------------------

_TEMPLATE_FOR_BASE = {
    "meta/llama-3.2-3b-instruct": "meta/llama-3.2-3b-instruct@v1.0.0+80GB",
    "meta/llama-3.1-8b-instruct": "meta/llama-3.1-8b-instruct@v1.0.0+80GB",
}


def _output_path_for(spec: AdapterSpec, base_root: Path) -> Path:
    base_dir_name = {
        "meta/llama-3.2-3b-instruct": "lora-llama-3.2-3b",
        "meta/llama-3.1-8b-instruct": "lora-llama-3.1-8b",
    }[spec.base_model]
    return base_root / base_dir_name / spec.adapter_name


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit one Customizer LoRA adapter job.")
    ap.add_argument("--collection", required=True,
                    choices=["nim_curated", "nemo_usvcs_curated"])
    ap.add_argument("--base-model", required=True,
                    choices=list(_TEMPLATE_FOR_BASE.keys()))
    ap.add_argument("--rank", required=True, type=int, choices=[16, 32])
    ap.add_argument("--customizer-url",
                    default="http://nemo-customizer.nemo-peft:8000",
                    help="Customizer REST endpoint (cluster-local)")
    ap.add_argument("--datasets-root", type=Path,
                    default=Path("/mnt/nvme2/peft/datasets/v2"))
    ap.add_argument("--checkpoints-root", type=Path,
                    default=Path("/mnt/nvme2/peft/checkpoints"))
    ap.add_argument("--wait", action="store_true",
                    help="Block until job terminates")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

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

    coll_dir = args.datasets_root / args.collection
    output_path = _output_path_for(spec, args.checkpoints_root)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        cfg = build_customizer_config(
            spec, _TEMPLATE_FOR_BASE[args.base_model],
            str(coll_dir / "adapter_train.jsonl"),
            str(coll_dir / "adapter_val.jsonl"),
            str(output_path),
        )
        import json
        print(json.dumps(cfg, indent=2))
        return 0

    with CustomizerClient(args.customizer_url) as client:
        job_id = submit_adapter_job(
            spec, _TEMPLATE_FOR_BASE[args.base_model],
            adapter_train_path=str(coll_dir / "adapter_train.jsonl"),
            adapter_val_path=str(coll_dir / "adapter_val.jsonl"),
            output_path=str(output_path),
            client=client,
        )
        log.info("Submitted: job_id=%s adapter=%s output=%s",
                 job_id, adapter_name, output_path)
        print(job_id)
        if args.wait:
            terminal = client.wait_until_done(job_id)
            log.info("Terminal status: %s", terminal.value)
            return 0 if terminal == JobStatus.COMPLETED else 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
