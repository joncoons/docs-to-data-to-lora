#!/usr/bin/env python3
"""Resume the augmented 3B sequence at the NeMo Microservices adapters."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ops.run_3b_augmented_training_sequence import BASE_MODEL, BASE_TEMPLATE, run_one
from scripts.stage3.customizer_client import CustomizerClient, JobStatus


def write_event(path: Path, event: dict) -> None:
    event = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customizer-url", default="http://192.168.1.187:30910")
    parser.add_argument(
        "--log-dir",
        default="<ARTIFACT_ROOT>/training-runs/3b-dd5x-sequential-fullslice-nemo-resume",
    )
    parser.add_argument("--poll-interval-s", type=int, default=30)
    parser.add_argument("--timeout-s", type=int, default=8 * 3600)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir = Path(args.log_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    event_log = log_dir / "events.jsonl"

    plan = [("nemo_usvcs_curated", 16), ("nemo_usvcs_curated", 32)]
    write_event(
        event_log,
        {
            "event": "sequence_resume_start",
            "base_model": BASE_MODEL,
            "config": BASE_TEMPLATE,
            "precision": "bf16-mixed",
            "plan": [{"collection": c, "rank": r} for c, r in plan],
            "resume_reason": "NIM r32 artifact verified despite Customizer cancelled state",
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
