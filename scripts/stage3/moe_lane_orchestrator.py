#!/usr/bin/env python3
"""Rank-lane orchestrator for MoE Task 4.6 — used after the 2026-05-27 reboot.

Strategy: one Blackwell handles the r=16 lane, the other handles the r=32 lane.
Per-step time differs by rank → write windows stagger naturally → no concurrent
NFS post-training writes. See INTERLEAVED_GPU_SCHEDULING.md §Pattern 1.

Job queue (6 SFT jobs):
  r=16 lane (2 jobs):   nemo r=16 shard-a → nemo r=16 shard-b
  r=32 lane (4 jobs):   nim r=32 shard-a → nim r=32 shard-b →
                        nemo r=32 shard-a → nemo r=32 shard-b

After all 6 terminate, runs 3 TIES merges:
  nim r=32:        (Round 2 redo, both freshly trained)
  nemo r=16:       (Round 3, both freshly trained)
  nemo r=32:       (Round 4, both freshly trained)

Status log: /tmp/moe-lane-status.log
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/joncoons/claude/docs-to-data-to-lora")
PY_BIN = "/home/joncoons/anaconda3/envs/nat/bin/python3"
TRAIN_MOE = str(REPO / "scripts" / "stage3" / "train_adapter_moe.py")
TIES_MERGE = str(REPO / "scripts" / "stage3" / "ties_merge.py")

CUSTOMIZER = "http://10.43.167.101:8000"
TERMINAL = {"completed", "failed", "cancelled"}
MERGE_ROOT = Path("/mnt/nvme2/peft/checkpoints/lora")
LOG = "/tmp/moe-lane-status.log"

# Lane definitions: each lane is an ORDERED list of (collection, rank, shard).
# Lane runs sequentially within itself; lanes run independently in parallel.
LANE_R16: list[tuple[str, int, str]] = [
    ("nemo_usvcs_curated", 16, "a"),
    ("nemo_usvcs_curated", 16, "b"),
]
LANE_R32: list[tuple[str, int, str]] = [
    ("nim_curated", 32, "a"),
    ("nim_curated", 32, "b"),
    ("nemo_usvcs_curated", 32, "a"),
    ("nemo_usvcs_curated", 32, "b"),
]

# TIES-merge pairings (collection, rank) → looks up (a-jid, b-jid) from completed map
MERGES_TO_DO: list[tuple[str, int]] = [
    ("nim_curated", 32),
    ("nemo_usvcs_curated", 16),
    ("nemo_usvcs_curated", 32),
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fetch_job(jid: str) -> dict:
    url = f"{CUSTOMIZER}/v1/customization/jobs/{jid}"
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read().decode())


def submit_one(collection: str, rank: int, shard: str) -> str:
    log(f"  submitting {collection} × r={rank} × shard-{shard}")
    r = subprocess.run(
        [PY_BIN, TRAIN_MOE,
         "--collection", collection,
         "--rank", str(rank),
         "--shard", shard],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"submit failed: stderr={r.stderr[-400:]}")
    job_id = r.stdout.strip().split("\n")[-1]
    log(f"  → job_id={job_id}")
    return job_id


def wait_terminal(jid: str, poll_s: float = 120, max_wait_s: float = 5 * 3600) -> str:
    """Block until job reaches terminal state; return final status string."""
    deadline = time.monotonic() + max_wait_s
    consecutive_errors = 0
    while time.monotonic() < deadline:
        try:
            d = fetch_job(jid)
            consecutive_errors = 0
            status = d.get("status")
            if status in TERMINAL:
                sd = d.get("status_details", {}) or {}
                log(f"  {jid} terminal: status={status} "
                    f"train_loss={sd.get('train_loss')} val_loss={sd.get('val_loss')}")
                return status
        except Exception as e:
            consecutive_errors += 1
            log(f"  poll error on {jid} (#{consecutive_errors}): {e}")
            if consecutive_errors >= 10:
                raise RuntimeError(
                    f"Job {jid} failed 10 consecutive polls; aborting"
                )
        time.sleep(poll_s)
    raise TimeoutError(f"Job {jid} did not terminate within {max_wait_s}s")


def ties_merge_pair(collection: str, rank: int, job_a: str, job_b: str) -> Path:
    coll_short = "nim" if collection == "nim_curated" else "nemo-usvcs"
    merged_name = f"lora-{coll_short}-nemotron-nano-30b-r{rank}"
    out_dir = MERGE_ROOT / merged_name
    spec_a = f"default/lora-{coll_short}-nemotron-nano-30b-r{rank}-shard-a@{job_a}"
    spec_b = f"default/lora-{coll_short}-nemotron-nano-30b-r{rank}-shard-b@{job_b}"
    log(f"  TIES merging {spec_a} + {spec_b} → {out_dir}")
    r = subprocess.run(
        [PY_BIN, TIES_MERGE,
         "--adapters", spec_a, spec_b,
         "--out", str(out_dir),
         "--trim-ratio", "0.2"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        log(f"  ties_merge stdout (last 600): {r.stdout[-600:]}")
        log(f"  ties_merge stderr (last 600): {r.stderr[-600:]}")
        raise RuntimeError(f"ties_merge failed for {collection} r={rank}")
    log(f"  merged adapter ready: {out_dir}")
    return out_dir


def run_lane(name: str, queue: list[tuple[str, int, str]],
             completed: dict[tuple[str, int, str], str]) -> None:
    """Run one lane sequentially: submit job N, wait until terminal, submit N+1, ..."""
    log(f"=== {name} lane starting ({len(queue)} jobs) ===")
    for spec in queue:
        jid = submit_one(*spec)
        final = wait_terminal(jid)
        completed[spec] = jid
        if final == "failed":
            log(f"  WARN: {spec} failed — TIES merge will skip this pair")
    log(f"=== {name} lane complete ===")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    open(LOG, "w").close()  # truncate log for a clean run
    log("=== MoE Task 4.6 lane orchestrator starting (post-reboot pickup) ===")
    log(f"r=16 lane ({len(LANE_R16)} jobs): {LANE_R16}")
    log(f"r=32 lane ({len(LANE_R32)} jobs): {LANE_R32}")
    log(f"Expected wall-clock: ~12h (driven by r=32 lane)")

    # Two lanes run in parallel via subprocess.Popen pattern? Simpler: use threads.
    import threading
    completed: dict[tuple[str, int, str], str] = {}
    lane_threads = [
        threading.Thread(target=run_lane, args=("r=16", LANE_R16, completed), name="r16"),
        threading.Thread(target=run_lane, args=("r=32", LANE_R32, completed), name="r32"),
    ]
    for t in lane_threads:
        t.start()
    for t in lane_threads:
        t.join()
    log("=== Both lanes complete ===")
    log(f"Completed {len(completed)} jobs total")

    # TIES merges
    log("=== Starting TIES merges ===")
    for collection, rank in MERGES_TO_DO:
        spec_a = (collection, rank, "a")
        spec_b = (collection, rank, "b")
        a_jid = completed.get(spec_a)
        b_jid = completed.get(spec_b)
        if a_jid and b_jid:
            try:
                ties_merge_pair(collection, rank, a_jid, b_jid)
            except Exception as e:
                log(f"  {collection} r={rank} merge FAILED: {e!r}")
        else:
            log(f"  {collection} r={rank} merge SKIPPED: a_jid={a_jid}, b_jid={b_jid}")

    log("=== ALL MOE TASK 4.6 WORK COMPLETE ===")
    log("Merged adapter inventory:")
    for coll_short in ("nim", "nemo-usvcs"):
        for rank in (16, 32):
            log(f"  /mnt/nvme2/peft/checkpoints/lora/lora-{coll_short}-nemotron-nano-30b-r{rank}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
