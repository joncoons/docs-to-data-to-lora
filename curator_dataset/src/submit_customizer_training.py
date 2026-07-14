#!/usr/bin/env python3
"""Submit the Curator dataset LoRA matrix without touching the legacy pipeline."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx


BASE_CONFIG = "nvidia/nemotron-3-nano-30b-a3b@v1.0+96GB-singleGPU"
JOBS = {
    "nim_r16": {
        "phase": "nim",
        "rank": 16,
        "dataset": "default/stage3-nim-curated-curator-diverseqa-20260629",
        "output_model": "default/lora-nim-curator-nemotron-nano-30b-r16-20260629",
    },
    "nim_r32": {
        "phase": "nim",
        "rank": 32,
        "dataset": "default/stage3-nim-curated-curator-diverseqa-20260629",
        "output_model": "default/lora-nim-curator-nemotron-nano-30b-r32-20260629",
    },
    "nemo_usvcs_r16": {
        "phase": "nemo_usvcs",
        "rank": 16,
        "dataset": "default/stage3-nemo-usvcs-curated-curator-diverseqa-20260629",
        "output_model": "default/lora-nemo-usvcs-curator-nemotron-nano-30b-r16-20260629",
    },
    "nemo_usvcs_r32": {
        "phase": "nemo_usvcs",
        "rank": 32,
        "dataset": "default/stage3-nemo-usvcs-curated-curator-diverseqa-20260629",
        "output_model": "default/lora-nemo-usvcs-curator-nemotron-nano-30b-r32-20260629",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def payload(key: str, spec: dict) -> dict:
    rank = spec["rank"]
    return {
        "description": (
            f"Curator DiverseQA 20260629 — {key} — Nemotron Nano 30B single-GPU LoRA"
        ),
        "dataset": spec["dataset"],
        "output_model": spec["output_model"],
        "config": BASE_CONFIG,
        "hyperparameters": {
            "finetuning_type": "lora",
            "training_type": "sft",
            "warmup_steps": 20,
            "seed": 42,
            "max_steps": -1,
            "optimizer": "adamw_with_cosine_annealing",
            "adam_beta1": 0.9,
            "adam_beta2": 0.99,
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 1.0e-4,
            "log_every_n_steps": 10,
            "lora": {
                "adapter_dim": rank,
                "alpha": rank,
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

    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
    else:
        manifest = {
            "schema_version": "curator_dataset.customizer_training.v1",
            "created_at": utc_now(),
            "base_config": BASE_CONFIG,
            "base_model": "nvidia/nemotron-3-nano-30b-a3b",
            "gpu_policy": {
                "node": "ubuntu-local-dev",
                "physical_gpus": 2,
                "scheduler_capacity": 2,
                "time_slicing_temporarily_disabled": True,
                "original_replicas_per_gpu": 4,
            },
            "timing_policy": {
                "queue_delay": "first observed running timestamp minus submitted_at",
                "training_seconds": "Customizer status_details.elapsed_time",
                "total_seconds": "terminal observed timestamp minus submitted_at",
            },
            "jobs": {},
        }

    selected = [(key, spec) for key, spec in JOBS.items() if spec["phase"] == args.phase]
    if args.dry_run:
        print(json.dumps({key: payload(key, spec) for key, spec in selected}, indent=2))
        return 0

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
