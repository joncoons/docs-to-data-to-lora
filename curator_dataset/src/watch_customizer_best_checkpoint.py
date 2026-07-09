#!/usr/bin/env python3
"""Verify validation-best checkpoint promotion for the five-epoch matrix."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


TERMINAL = {"completed", "failed", "cancelled", "canceled"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_atomic(path: Path, value: dict) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--customizer-url", default="http://10.43.167.101:8000")
    parser.add_argument("--entity-store-url", default="http://10.43.187.212:8000")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(str(os.getpid()) + "\n")
    try:
        while True:
            manifest = json.loads(args.manifest.read_text())
            result = {
                "schema_version": "curator_dataset.best_checkpoint_promotion.v1",
                "observed_at": now_iso(),
                "policy": {
                    "maximum_epochs": 5,
                    "early_stopping": False,
                    "selection_metric": "validation loss",
                    "promotion_target": "Customizer output model revision",
                },
                "jobs": {},
            }
            all_done = bool(manifest["jobs"])
            all_promoted_or_terminal_failure = bool(manifest["jobs"])
            with httpx.Client(timeout=30) as client:
                for key, record in manifest["jobs"].items():
                    response = client.get(
                        f"{args.customizer_url.rstrip('/')}/v1/customization/jobs/{record['job_id']}"
                    )
                    response.raise_for_status()
                    body = response.json()
                    status = body["status"].lower()
                    details = body.get("status_details") or {}
                    val_history = (
                        details.get("metrics", {}).get("metrics", {}).get("val_loss", [])
                    )
                    best_observed = min(
                        val_history,
                        key=lambda item: item.get("value", float("inf")),
                        default=None,
                    )
                    artifact_status = None
                    artifact_files_url = None
                    if status == "completed":
                        namespace, name = record["output_model"].split("/", 1)
                        model_ref = f"{name}@{record['job_id']}"
                        artifact_response = client.get(
                            f"{args.entity_store_url.rstrip('/')}/v1/models/{namespace}/{model_ref}"
                        )
                        if artifact_response.status_code == 200:
                            artifact = artifact_response.json().get("artifact") or {}
                            artifact_status = artifact.get("status")
                            artifact_files_url = artifact.get("files_url")
                    promotion_verified = (
                        status == "completed"
                        and artifact_status == "upload_completed"
                        and details.get("best_epoch") is not None
                    )
                    result["jobs"][key] = {
                        "job_id": record["job_id"],
                        "status": status,
                        "epochs_completed": details.get("epochs_completed"),
                        "best_epoch": details.get("best_epoch"),
                        "best_val_loss": (
                            best_observed.get("value") if best_observed else details.get("val_loss")
                        ),
                        "best_val_loss_step": best_observed.get("step") if best_observed else None,
                        "artifact_status": artifact_status,
                        "artifact_files_url": artifact_files_url,
                        "promotion_verified": promotion_verified,
                    }
                    all_done = all_done and status in TERMINAL
                    all_promoted_or_terminal_failure = all_promoted_or_terminal_failure and (
                        promotion_verified or status in {"failed", "cancelled", "canceled"}
                    )
            write_atomic(args.output, result)
            if all_done and all_promoted_or_terminal_failure:
                return 0
            time.sleep(args.interval)
    finally:
        args.pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
