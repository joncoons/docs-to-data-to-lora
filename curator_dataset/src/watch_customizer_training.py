#!/usr/bin/env python3
"""Resilient detached watcher around track_customizer_gpu_time.py."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


TERMINAL = {"completed", "failed", "cancelled", "canceled"}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--pid-file", type=Path, required=True)
    args = parser.parse_args()
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(str(os.getpid()) + "\n")
    errors = args.output.with_name("timing_watcher_errors.jsonl")
    tracker = Path(__file__).with_name("track_customizer_gpu_time.py")
    try:
        while True:
            command = [
                sys.executable,
                str(tracker),
                "--manifest",
                str(args.manifest),
                "--output",
                str(args.output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                with errors.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "observed_at": timestamp(),
                                "returncode": result.returncode,
                                "stderr": result.stderr[-2000:],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            elif args.output.exists():
                timing = json.loads(args.output.read_text())
                statuses = [
                    item.get("customizer_status")
                    for item in timing.get("jobs", {}).values()
                ]
                if statuses and all(status in TERMINAL for status in statuses):
                    return 0
            time.sleep(args.interval)
    finally:
        args.pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
