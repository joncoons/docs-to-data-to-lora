#!/usr/bin/env python3
"""Dual-lane (r=16 / r=32) resume orchestrator — preserved for future use.

**CAVEAT (2026-05-27)**: this script was NOT used to produce the final Stage 3
MoE adapters. **r=32 never succeeded for the Nemotron-Nano-30B-A3B MoE model**
— it failed early via divergence in every attempt (loss → 9, grad_norm → 1632
at step 18 of `cust-U9RnGyiJTxYzGS1Kxrv4bD`, α=rank=32, lr=1e-4, warmup=20).
r=32 was abandoned for this MoE model; the final Stage 3 plan is r=16-only
with fully-serial submission via `moe_serial_r16_orchestrator.py`. Use that
script for the canonical Nemotron-Nano-30B-A3B MoE workflow.

**This caveat applies only to the Nemotron Nano MoE base model.** r=32 trains
stably for the dense Llama 3.2-1B / 3.2-3B / 3.1-8B adapters in Stage 3
(see `evals/training_session.log`); the dual-lane interleave pattern this
file encodes (one GPU per rank, exploit per-step time differential to stagger
NFS writes — see `INTERLEAVED_GPU_SCHEDULING.md`) is the right approach for
those dense runs and for any **future MoE recipe** that resolves the Nano
r=32 divergence — e.g., longer warmup, lower lr (5e-5), or different LoRA
target_modules. Adjust `LANE_R16` / `LANE_R32` and the `preexisting_r16`
wait-on-existing-job logic as needed.

Original purpose: resume Stage 3 MoE Task 4.6 after the 2026-05-27
warmup-validation failure. r=16 lane waited on a preexisting in-flight shard-a
job, r=32 lane re-ran all 4 jobs after the warmup fix landed in
train_adapter_moe.py. After all 6 reach terminal state, TIES-merge 3 pairs
(nim r=32, nemo-usvcs r=16, nemo-usvcs r=32).

Status log: /tmp/moe-lane-resume-status.log
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY_BIN = "<USER_HOME>/anaconda3/envs/nat/bin/python3"
TRAIN_MOE = str(REPO / "scripts" / "stage3" / "train_adapter_moe.py")
TIES_MERGE = str(REPO / "scripts" / "stage3" / "ties_merge.py")

CUSTOMIZER = "http://10.43.167.101:8000"
TERMINAL = {"completed", "failed", "cancelled"}
MERGE_ROOT = Path("<ARTIFACT_ROOT>/checkpoints/lora")
LOG = "/tmp/moe-lane-resume-status.log"

# Round-2 resume (after first attempt failed warmup validation):
# both lanes start fresh — both prior submissions were cancelled
PREEXISTING_R16_A_JID = None
PREEXISTING_R16_A_SPEC = None

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
                raise RuntimeError(f"Job {jid} failed 10 consecutive polls; aborting")
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


def run_lane(name: str,
             queue: list[tuple[str, int, str]],
             completed: dict[tuple[str, int, str], str],
             preexisting: tuple[str, tuple[str, int, str]] | None = None) -> None:
    """Run one lane sequentially. If `preexisting` is set, wait for that job
    to terminate FIRST before processing the queue, and record its jid in completed."""
    log(f"=== {name} lane starting ({len(queue)} new jobs"
        f"{', after waiting on preexisting job' if preexisting else ''}) ===")
    if preexisting is not None:
        pre_jid, pre_spec = preexisting
        log(f"  waiting on preexisting {pre_spec} → {pre_jid}")
        final = wait_terminal(pre_jid)
        completed[pre_spec] = pre_jid
        if final == "failed":
            log(f"  WARN: preexisting {pre_spec} failed — TIES merge will skip this pair")
    for spec in queue:
        jid = submit_one(*spec)
        final = wait_terminal(jid)
        completed[spec] = jid
        if final == "failed":
            log(f"  WARN: {spec} failed — TIES merge will skip this pair")
    log(f"=== {name} lane complete ===")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    open(LOG, "w").close()
    log("=== MoE Task 4.6 RESUME orchestrator starting ===")
    log(f"  r=16 preexisting: {PREEXISTING_R16_A_SPEC} → {PREEXISTING_R16_A_JID}")
    log(f"  r=16 queue ({len(LANE_R16)} jobs): {LANE_R16}")
    log(f"  r=32 queue ({len(LANE_R32)} jobs): {LANE_R32}")

    import threading
    completed: dict[tuple[str, int, str], str] = {}
    preexisting_r16 = (
        (PREEXISTING_R16_A_JID, PREEXISTING_R16_A_SPEC)
        if PREEXISTING_R16_A_JID is not None else None
    )
    lane_threads = [
        threading.Thread(
            target=run_lane,
            args=("r=16", LANE_R16, completed, preexisting_r16),
            name="r16",
        ),
        threading.Thread(target=run_lane, args=("r=32", LANE_R32, completed), name="r32"),
    ]
    for t in lane_threads:
        t.start()
    for t in lane_threads:
        t.join()
    log("=== Both lanes complete ===")
    log(f"Completed {len(completed)} jobs total")

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
            log(f"  <ARTIFACT_ROOT>/checkpoints/lora/lora-{coll_short}-nemotron-nano-30b-r{rank}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
