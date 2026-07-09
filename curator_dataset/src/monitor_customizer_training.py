#!/usr/bin/env python3
"""Persist Customizer status transitions and authoritative training timings."""
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


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def write_atomic(path: Path, value: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def snapshot(manifest_path: Path, base_url: str) -> tuple[dict, bool]:
    manifest = json.loads(manifest_path.read_text())
    events_path = manifest_path.with_name("status_events.jsonl")
    all_terminal = bool(manifest["jobs"])
    observed_at = now()
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30) as client:
        for key, record in manifest["jobs"].items():
            response = client.get(f"/v1/customization/jobs/{record['job_id']}")
            response.raise_for_status()
            body = response.json()
            status = body["status"].lower()
            previous = record.get("last_status")
            if previous != status:
                event = {
                    "observed_at": iso(observed_at),
                    "key": key,
                    "job_id": record["job_id"],
                    "previous_status": previous,
                    "status": status,
                }
                with events_path.open("a") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                if status == "running" and not record.get("first_running_at"):
                    record["first_running_at"] = iso(observed_at)
                    record["queue_delay_seconds"] = round(
                        (observed_at - parse_iso(record["submitted_at"])).total_seconds(), 3
                    )
                if status in TERMINAL and not record.get("terminal_observed_at"):
                    record["terminal_observed_at"] = iso(observed_at)
                    record["total_elapsed_seconds"] = round(
                        (observed_at - parse_iso(record["submitted_at"])).total_seconds(), 3
                    )
            details = body.get("status_details") or {}
            record.update(
                {
                    "last_status": status,
                    "last_observed_at": iso(observed_at),
                    "customizer_updated_at": body.get("updated_at"),
                    "training_elapsed_seconds": details.get("elapsed_time"),
                    "steps_per_epoch": details.get("steps_per_epoch"),
                    "steps_completed": details.get("steps_completed"),
                    "epochs_completed": details.get("epochs_completed"),
                    "percentage_done": details.get("percentage_done"),
                    "output_model_revision": body.get("output_model"),
                }
            )
            all_terminal = all_terminal and status in TERMINAL
    manifest["last_observed_at"] = iso(observed_at)
    write_atomic(manifest_path, manifest)
    return manifest, all_terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", default="http://10.43.167.101:8000")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    while True:
        manifest, done = snapshot(args.manifest, args.base_url)
        print(json.dumps({k: v.get("last_status") for k, v in manifest["jobs"].items()}, sort_keys=True))
        if not args.watch or done:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
