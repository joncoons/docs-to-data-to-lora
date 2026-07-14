#!/usr/bin/env python3
"""Submit a two-point LR canary for the Curator NIM dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from submit_customizer_training import utc_now, write_json_atomic


BASE_CONFIG = "meta/llama-3.1-8b-instruct@v1.0.0+80GB"
DATASET = "default/stage3-nim-curated-curator-diverseqa-20260629"
CANARIES = {
    "nim_r16_lr5e5": {
        "learning_rate": 5.0e-5,
        "output_model": "default/lora-nim-curator-lr5e5-e2-llama31-8b-r16-20260629",
    },
    "nim_r16_lr6p8e5": {
        "learning_rate": 6.8e-5,
        "output_model": "default/lora-nim-curator-lr6p8e5-e2-llama31-8b-r16-20260629",
    },
}


def payload(key: str, spec: dict) -> dict:
    return {
        "description": f"Curator NIM r16 two-epoch LR canary — {key}",
        "dataset": DATASET,
        "output_model": spec["output_model"],
        "config": BASE_CONFIG,
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
            "learning_rate": spec["learning_rate"],
            "log_every_n_steps": 10,
            "lora": {
                "adapter_dim": 16,
                "alpha": 32,
                "adapter_dropout": None,
                "target_modules": None,
            },
            "sequence_packing_enabled": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://10.43.167.101:8000")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({k: payload(k, v) for k, v in CANARIES.items()}, indent=2))
        return 0

    manifest = {
        "schema_version": "curator_dataset.customizer_lr_canary.v1",
        "created_at": utc_now(),
        "base_config": BASE_CONFIG,
        "base_model": "meta/llama-3.1-8b-instruct",
        "dataset": DATASET,
        "controlled_fields": {
            "epochs": 2,
            "batch_size": 16,
            "warmup_steps": 30,
            "rank": 16,
            "alpha": 32,
            "sequence_packing_enabled": False,
            "seed": 42,
        },
        "comparison": {
            "variable": "learning_rate",
            "prior_lr": 1.0e-4,
            "prior_best_val_loss": 1.4240126609802246,
            "prior_best_epoch": 1,
            "prior_job_id": "cust-6cpWwCz38TzoDCmZcMJK6U",
        },
        "jobs": {},
    }
    if args.manifest.exists():
        raise RuntimeError(f"refusing to overwrite existing manifest: {args.manifest}")

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=60) as client:
        for key, spec in CANARIES.items():
            submitted_at = utc_now()
            response = client.post("/v1/customization/jobs", json=payload(key, spec))
            response.raise_for_status()
            job_id = response.json()["id"]
            snapshot = client.get(f"/v1/customization/jobs/{job_id}")
            snapshot.raise_for_status()
            manifest["jobs"][key] = {
                **spec,
                "dataset": DATASET,
                "rank": 16,
                "epochs": 2,
                "job_id": job_id,
                "submitted_at": submitted_at,
                "initial_status": snapshot.json().get("status"),
            }
            write_json_atomic(args.manifest, manifest)
            print(json.dumps({"key": key, "job_id": job_id, "status": snapshot.json().get("status")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
