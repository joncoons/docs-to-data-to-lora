#!/usr/bin/env python3
"""Run augmented Llama 3.2 3B adapter trainings sequentially."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.stage3.customizer_client import CustomizerClient, JobStatus
from scripts.stage3.models import AdapterSpec
from scripts.stage3.train_adapter import build_customizer_config


BASE_MODEL = "meta/llama-3.2-3b-instruct"
BASE_TEMPLATE = "meta/llama-3.2-3b-instruct@v1.0.0+80GB"
ENTITY_STORE_URL = os.getenv("ENTITY_STORE_URL", "http://192.168.1.187:30911")
DATA_STORE_HF_URL = os.getenv("DATA_STORE_HF_URL", "http://192.168.1.187:30912/v1/hf")
REQUIRED_ADAPTER_FILES = {"adapter_config.json", "adapter_model.safetensors"}
DATASETS = {
    "nim_curated": {
        "dataset_entity": "default/stage3-nim-curated-dd-deterministic-5x",
        "prefix": "lora-nim-dd5x",
        "label": "NIM augmented deterministic 5x",
    },
    "nemo_usvcs_curated": {
        "dataset_entity": "default/stage3-nemo-usvcs-curated-dd-deterministic-5x",
        "prefix": "lora-nemo-usvcs-dd5x",
        "label": "NeMo Microservices augmented deterministic 5x",
    },
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_event(path: Path, event: dict) -> None:
    event = {"timestamp": now(), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def verify_exported_artifact(output_model: str, job_id: str) -> dict:
    """Check Entity Store and Data Store for a completed adapter export."""
    namespace, repo_name = output_model.split("/", 1)
    model_ref = f"{output_model}@{job_id}"
    result = {
        "model_ref": model_ref,
        "entity_store_url": ENTITY_STORE_URL,
        "data_store_hf_url": DATA_STORE_HF_URL,
        "verified": False,
    }

    try:
        entity_url = f"{ENTITY_STORE_URL.rstrip("/")}/v1/models/{namespace}/{repo_name}@{job_id}"
        entity_resp = httpx.get(entity_url, timeout=30)
        result["entity_store_status_code"] = entity_resp.status_code
        if entity_resp.is_error:
            result["entity_store_error"] = entity_resp.text[:1000]
            return result

        body = entity_resp.json()
        artifact = body.get("artifact") or {}
        result["entity_store_artifact_status"] = artifact.get("status")
        result["files_url"] = artifact.get("files_url")
    except Exception as exc:  # noqa: BLE001 - operational recovery path
        result["entity_store_error"] = repr(exc)
        return result

    if result.get("entity_store_artifact_status") != "upload_completed":
        return result

    try:
        api = HfApi(endpoint=DATA_STORE_HF_URL)
        files = api.list_repo_files(
            repo_id=output_model,
            revision=job_id,
            repo_type="model",
        )
        result["data_store_file_count"] = len(files)
        result["required_files_present"] = sorted(REQUIRED_ADAPTER_FILES.intersection(files))
        result["verified"] = REQUIRED_ADAPTER_FILES.issubset(files)
    except Exception as exc:  # noqa: BLE001 - operational recovery path
        result["data_store_error"] = repr(exc)

    return result


def run_one(
    client: CustomizerClient,
    event_log: Path,
    collection: str,
    rank: int,
    poll_interval_s: int,
    timeout_s: int,
) -> JobStatus:
    dataset = DATASETS[collection]
    adapter_name = f"{dataset['prefix']}-llama-3.2-3b-r{rank}"
    output_model = f"default/{adapter_name}"
    description = (
        f"Stage 3 augmented deterministic 5x - {dataset['label']} x "
        f"Llama 3.2 3B BF16/mixed LoRA r{rank}; sequential Ada run"
    )
    spec = AdapterSpec(
        adapter_name=adapter_name,
        collection=collection,
        base_model=BASE_MODEL,
        rank=rank,
        alpha=2 * rank,
    )
    cfg = build_customizer_config(
        spec,
        BASE_TEMPLATE,
        dataset["dataset_entity"],
        output_model,
        description,
    )
    write_event(
        event_log,
        {
            "event": "submit",
            "collection": collection,
            "rank": rank,
            "adapter_name": adapter_name,
            "dataset": dataset["dataset_entity"],
            "output_model": output_model,
            "config": BASE_TEMPLATE,
            "precision": "bf16-mixed",
        },
    )
    job_id = client.submit_job(cfg)
    write_event(
        event_log,
        {
            "event": "submitted",
            "collection": collection,
            "rank": rank,
            "adapter_name": adapter_name,
            "job_id": job_id,
        },
    )
    logging.info("Submitted %s as %s", adapter_name, job_id)

    terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    started = time.time()
    last_status = None
    while time.time() - started < timeout_s:
        status = client.get_status(job_id)
        if status != last_status:
            write_event(
                event_log,
                {
                    "event": "status",
                    "collection": collection,
                    "rank": rank,
                    "adapter_name": adapter_name,
                    "job_id": job_id,
                    "status": status.value,
                },
            )
            logging.info("%s status=%s", job_id, status.value)
            last_status = status
        if status in terminal:
            output_path = client.get_output_path(job_id)
            recovered_status = None
            if status == JobStatus.CANCELLED:
                artifact_verification = verify_exported_artifact(output_model, job_id)
                write_event(
                    event_log,
                    {
                        "event": "artifact_verification",
                        "collection": collection,
                        "rank": rank,
                        "adapter_name": adapter_name,
                        "job_id": job_id,
                        **artifact_verification,
                    },
                )
                if artifact_verification.get("verified"):
                    recovered_status = JobStatus.COMPLETED.value

            write_event(
                event_log,
                {
                    "event": "terminal",
                    "collection": collection,
                    "rank": rank,
                    "adapter_name": adapter_name,
                    "job_id": job_id,
                    "status": status.value,
                    "recovered_status": recovered_status,
                    "output_path": output_path,
                },
            )
            if recovered_status == JobStatus.COMPLETED.value:
                logging.info("%s recovered as completed from uploaded artifact", job_id)
                return JobStatus.COMPLETED
            return status
        time.sleep(poll_interval_s)

    raise TimeoutError(f"{job_id} did not finish within {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customizer-url", default="http://192.168.1.187:30910")
    parser.add_argument(
        "--log-dir",
        default="/mnt/nvme2/peft/training-runs/3b-dd5x-sequential",
    )
    parser.add_argument("--poll-interval-s", type=int, default=30)
    parser.add_argument("--timeout-s", type=int, default=8 * 3600)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir = Path(args.log_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    event_log = log_dir / "events.jsonl"

    plan = [
        ("nim_curated", 16),
        ("nim_curated", 32),
        ("nemo_usvcs_curated", 16),
        ("nemo_usvcs_curated", 32),
    ]
    write_event(
        event_log,
        {
            "event": "sequence_start",
            "base_model": BASE_MODEL,
            "config": BASE_TEMPLATE,
            "precision": "bf16-mixed",
            "plan": [{"collection": c, "rank": r} for c, r in plan],
        },
    )

    with CustomizerClient(args.customizer_url, timeout=60) as client:
        for collection, rank in plan:
            status = run_one(
                client,
                event_log,
                collection,
                rank,
                args.poll_interval_s,
                args.timeout_s,
            )
            if status != JobStatus.COMPLETED:
                write_event(
                    event_log,
                    {
                        "event": "sequence_stop",
                        "reason": f"{collection} r{rank} ended {status.value}",
                    },
                )
                return 1

    write_event(event_log, {"event": "sequence_complete"})
    print(event_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
