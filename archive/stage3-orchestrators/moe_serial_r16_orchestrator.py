#!/usr/bin/env python3
"""Stage 3 MoE serial r=16 orchestrator (2026-05-27 final plan).

Direction change: skip r=32 entirely for MoE; r=16 is sufficient. Run shards
serially (one GPU at a time) instead of the dual-lane parallel pattern. This
sidesteps the r=32 divergence at end-of-warmup observed in cust-U9RnGyiJTxYzGS1Kxrv4bD
and removes the concurrent NFS write hazard since only one shard is in flight.

Workflow:
  1. submit nemo_usvcs_curated r=16 shard-a → wait terminal
  2. submit nemo_usvcs_curated r=16 shard-b → wait terminal
  3. TIES-merge a + b → <ARTIFACT_ROOT>/checkpoints/lora/lora-nemo-usvcs-nemotron-nano-30b-r16/

nim r=16 already exists from Round 1 at
  <ARTIFACT_ROOT>/checkpoints/lora/lora-nim-nemotron-nano-30b-r16/
so no nim work needed here.

Status log: /tmp/moe-serial-r16-status.log
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
LOG = "/tmp/moe-serial-r16-status.log"

QUEUE: list[tuple[str, int, str]] = [
    ("nemo_usvcs_curated", 16, "a"),
    ("nemo_usvcs_curated", 16, "b"),
]
MERGES_TO_DO: list[tuple[str, int]] = [
    ("nemo_usvcs_curated", 16),
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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    open(LOG, "w").close()
    log("=== MoE serial r=16 orchestrator starting ===")
    log(f"  queue ({len(QUEUE)} jobs): {QUEUE}")
    log("  expected wall-clock ~5.2h (2 serial shards × ~2.6h)")

    completed: dict[tuple[str, int, str], str] = {}
    for spec in QUEUE:
        jid = submit_one(*spec)
        final = wait_terminal(jid)
        completed[spec] = jid
        if final != "completed":
            log(f"  WARN: {spec} ended with status={final} — TIES merge will skip this pair if missing")
    log("=== Serial queue complete ===")

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

    log("=== ALL R=16 MOE WORK COMPLETE ===")
    log("Merged adapter inventory (r=16 only):")
    log("  <ARTIFACT_ROOT>/checkpoints/lora/lora-nim-nemotron-nano-30b-r16/        (from Round 1, pre-existing)")
    log("  <ARTIFACT_ROOT>/checkpoints/lora/lora-nemo-usvcs-nemotron-nano-30b-r16/ (this run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
