#!/usr/bin/env python3
"""Track GPU-worker timing for the isolated Curator Customizer experiment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


TERMINAL = {"completed", "failed", "cancelled", "canceled"}


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def seconds(start: str | None, end: str | None) -> float | None:
    left, right = parse_time(start), parse_time(end)
    if not left or not right:
        return None
    return round((right - left).total_seconds(), 3)


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


def pod_snapshot(namespace: str) -> dict[str, dict]:
    raw = subprocess.check_output(
        ["kubectl", "get", "pods", "-n", namespace, "-o", "json"], text=True
    )
    items = json.loads(raw)["items"]
    return {item["metadata"]["name"]: item for item in items}


def condition_time(pod: dict, condition_type: str) -> str | None:
    for condition in pod.get("status", {}).get("conditions", []):
        if condition.get("type") == condition_type and condition.get("status") == "True":
            return condition.get("lastTransitionTime")
    return None


def container_times(pod: dict) -> tuple[str | None, str | None]:
    statuses = pod.get("status", {}).get("containerStatuses", [])
    for status in statuses:
        if status.get("name") == "main":
            state = status.get("state", {})
            started = (state.get("running") or {}).get("startedAt")
            terminated = state.get("terminated") or {}
            return started or terminated.get("startedAt"), terminated.get("finishedAt")
    return None, None


def collect(manifest_path: Path, output_path: Path, base_url: str, namespace: str) -> bool:
    manifest = json.loads(manifest_path.read_text())
    previous = json.loads(output_path.read_text()) if output_path.exists() else {"jobs": {}}
    pods = pod_snapshot(namespace)
    observed_at = now_iso()
    result = {
        "schema_version": "curator_dataset.customizer_gpu_timing.v1",
        "observed_at": observed_at,
        "jobs": {},
    }
    all_terminal = bool(manifest["jobs"])
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30) as client:
        for key, job in manifest["jobs"].items():
            job_id = job["job_id"]
            worker_name = f"{job_id.lower()}-training-job-worker-0"
            pod = pods.get(worker_name)
            response = client.get(f"/v1/customization/jobs/{job_id}")
            response.raise_for_status()
            customizer = response.json()
            status = customizer["status"].lower()
            details = customizer.get("status_details") or {}
            record = {
                "job_id": job_id,
                "rank": job["rank"],
                "dataset": job["dataset"],
                "submitted_at": job["submitted_at"],
                "customizer_status": status,
                "customizer_elapsed_seconds": details.get("elapsed_time"),
                "steps_per_epoch": details.get("steps_per_epoch"),
                "steps_completed": details.get("steps_completed"),
                "epochs_completed": details.get("epochs_completed"),
                "percentage_done": details.get("percentage_done"),
                "worker_pod": worker_name,
            }
            if pod:
                created = pod["metadata"].get("creationTimestamp")
                scheduled = condition_time(pod, "PodScheduled")
                pod_started = pod.get("status", {}).get("startTime")
                container_started, container_finished = container_times(pod)
                effective_start = container_started or pod_started
                record.update(
                    {
                        "worker_phase": pod.get("status", {}).get("phase"),
                        "worker_node": pod.get("spec", {}).get("nodeName"),
                        "worker_created_at": created,
                        "worker_scheduled_at": scheduled,
                        "worker_pod_started_at": pod_started,
                        "gpu_container_started_at": container_started,
                        "gpu_container_finished_at": container_finished,
                        "worker_queue_seconds": seconds(created, effective_start),
                        "submission_to_gpu_start_seconds": seconds(
                            job["submitted_at"], effective_start
                        ),
                        "gpu_training_seconds": seconds(
                            effective_start, container_finished
                        ),
                    }
                )
            result["jobs"][key] = record
            all_terminal = all_terminal and status in TERMINAL
            old = previous.get("jobs", {}).get(key, {})
            signature = (record.get("customizer_status"), record.get("worker_phase"))
            old_signature = (old.get("customizer_status"), old.get("worker_phase"))
            if signature != old_signature:
                with output_path.with_name("gpu_timing_events.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "observed_at": observed_at,
                                "key": key,
                                "job_id": job_id,
                                "customizer_status": signature[0],
                                "worker_phase": signature[1],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
    write_atomic(output_path, result)
    return all_terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://10.43.167.101:8000")
    parser.add_argument("--namespace", default="nemo-peft")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        done = collect(args.manifest, args.output, args.base_url, args.namespace)
        if not args.watch or done:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
