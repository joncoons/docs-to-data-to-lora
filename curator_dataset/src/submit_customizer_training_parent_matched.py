#!/usr/bin/env python3
"""Submit Curator datasets using the proven parent e5 Llama 3.1 8B recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from submit_customizer_training import utc_now, write_json_atomic


BASE_CONFIG = "meta/llama-3.1-8b-instruct@v1.0.0+80GB"
JOBS = {
    "nim_r16": {
        "phase": "nim",
        "rank": 16,
        "dataset": "default/stage3-nim-curated-curator-diverseqa-20260629",
        "output_model": "default/lora-nim-curator-e5-llama31-8b-r16-20260629",
    },
    "nim_r32": {
        "phase": "nim",
        "rank": 32,
        "dataset": "default/stage3-nim-curated-curator-diverseqa-20260629",
        "output_model": "default/lora-nim-curator-e5-llama31-8b-r32-20260629",
    },
    "nemo_usvcs_r16": {
        "phase": "nemo_usvcs",
        "rank": 16,
        "dataset": "default/stage3-nemo-usvcs-curated-curator-diverseqa-20260629",
        "output_model": "default/lora-nemo-ms-curator-e5-llama31-8b-r16-20260629",
    },
    "nemo_usvcs_r32": {
        "phase": "nemo_usvcs",
        "rank": 32,
        "dataset": "default/stage3-nemo-usvcs-curated-curator-diverseqa-20260629",
        "output_model": "default/lora-nemo-ms-curator-e5-llama31-8b-r32-20260629",
    },
}


def payload(key: str, spec: dict) -> dict:
    rank = spec["rank"]
    return {
        "description": (
            f"Curator DiverseQA parent-matched e5 — {key} — "
            "Llama 3.1 8B LoRA"
        ),
        "dataset": spec["dataset"],
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
            "epochs": 5,
            "learning_rate": 1.0e-4,
            "log_every_n_steps": 10,
            "lora": {
                "adapter_dim": rank,
                "alpha": 2 * rank,
                "adapter_dropout": None,
                "target_modules": None,
            },
            "sequence_packing_enabled": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("nim", "nemo_usvcs"))
    parser.add_argument("--base-url", default="http://10.43.167.101:8000")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = [(key, spec) for key, spec in JOBS.items() if spec["phase"] == args.phase]
    if args.dry_run:
        print(json.dumps({key: payload(key, spec) for key, spec in selected}, indent=2))
        return 0

    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
    else:
        manifest = {
            "schema_version": "curator_dataset.customizer_training.v1",
            "created_at": utc_now(),
            "run_variant": "parent-matched-llama31-8b-e5",
            "base_config": BASE_CONFIG,
            "base_model": "meta/llama-3.1-8b-instruct",
            "comparison_baseline_jobs": {
                "nim_r16": "cust-GQkpTXu3frr2PnWSnbcgKH",
                "nemo_usvcs_r16": "cust-EDmfZ5Hi9HaCw4Ko4wyLpH",
                "nim_r32": "cust-CiTm8oforcC38Rw67D3RVH",
                "nemo_usvcs_r32": "cust-XZMopAuFBH57vNpVJZzBvj",
            },
            "checkpoint_policy": {
                "maximum_epochs": 5,
                "early_stopping": False,
                "promotion": "Customizer validation-best checkpoint",
            },
            "gpu_policy": {
                "node": "ubuntu-local-dev",
                "physical_gpus": 2,
                "scheduler_capacity": 2,
                "time_slicing_temporarily_disabled": True,
            },
            "jobs": {},
        }

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=60) as client:
        for key, spec in selected:
            if key in manifest["jobs"]:
                raise RuntimeError(f"refusing duplicate submission for {key}")
            submitted_at = utc_now()
            response = client.post("/v1/customization/jobs", json=payload(key, spec))
            response.raise_for_status()
            job_id = response.json()["id"]
            snapshot = client.get(f"/v1/customization/jobs/{job_id}")
            snapshot.raise_for_status()
            manifest["jobs"][key] = {
                **spec,
                "epochs": 5,
                "job_id": job_id,
                "submitted_at": submitted_at,
                "initial_status": snapshot.json().get("status"),
                "customizer_created_at": snapshot.json().get("created_at"),
            }
            write_json_atomic(args.manifest, manifest)
            print(json.dumps({"key": key, "job_id": job_id, "status": snapshot.json().get("status")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
