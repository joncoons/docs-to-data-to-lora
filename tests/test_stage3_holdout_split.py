"""Tests for Stage 3 KVP-based holdout split."""
import json
from pathlib import Path

from scripts.stage3.holdout_split import (
    kvp_uid, pick_test_kvp_uids, partition_kvps, run_split,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_kvp_uid_stable_and_unique():
    """Same (passage_id, question) → same uid. Different inputs → different uids."""
    u1 = kvp_uid("p0", "What is X?")
    u2 = kvp_uid("p0", "What is X?")
    u3 = kvp_uid("p0", "What is Y?")
    u4 = kvp_uid("p1", "What is X?")
    assert u1 == u2
    assert u1 != u3
    assert u1 != u4
    assert len(u1) == 16          # 16-hex-char prefix of SHA1


def test_pick_test_kvp_uids_stratified_by_stage():
    """10% sample per stage, deterministic."""
    kvps = (
        [{"kvp_uid": f"a{i:02d}", "stage": "1a"} for i in range(50)] +
        [{"kvp_uid": f"b{i:02d}", "stage": "1b"} for i in range(30)] +
        [{"kvp_uid": f"c{i:02d}", "stage": "1c"} for i in range(20)]
    )
    a = pick_test_kvp_uids(kvps, fraction=0.1, seed=42)
    b = pick_test_kvp_uids(kvps, fraction=0.1, seed=42)
    assert a == b                  # deterministic
    # ~10% per stratum: 5 + 3 + 2 = 10
    assert len(a) == 10
    by_stage = {"1a": 0, "1b": 0, "1c": 0}
    uid_to_stage = {k["kvp_uid"]: k["stage"] for k in kvps}
    for uid in a:
        by_stage[uid_to_stage[uid]] += 1
    assert by_stage == {"1a": 5, "1b": 3, "1c": 2}


def test_partition_kvps_by_uid():
    kvps = [
        {"kvp_uid": "u0", "passage_id": "p0", "stage": "1a", "question": "Q0", "answer": "A0"},
        {"kvp_uid": "u1", "passage_id": "p0", "stage": "1a", "question": "Q1", "answer": "A1"},
        {"kvp_uid": "u2", "passage_id": "p1", "stage": "1b", "question": "Q2", "answer": "A2"},
    ]
    test_set, remaining = partition_kvps(kvps, test_kvp_uids={"u1"})
    assert len(test_set) == 1
    assert test_set[0]["kvp_uid"] == "u1"
    assert len(remaining) == 2
    # Same passage (p0) shows up in BOTH test and remaining — confirms KVP-based
    p0_in_test = any(k["passage_id"] == "p0" for k in test_set)
    p0_in_train = any(k["passage_id"] == "p0" for k in remaining)
    assert p0_in_test and p0_in_train


def test_run_split_writes_all_outputs(tmp_path):
    coll_dir = tmp_path / "nim_curated"
    coll_dir.mkdir()
    # 20 passages with 2 KVPs each = 40 KVPs total, mixed stages
    kvps_train = [
        {"passage_id": f"p{i}", "stage": "1a", "question": f"Q{i}", "answer": f"A{i}"}
        for i in range(20)
    ]
    kvps_val = [
        {"passage_id": f"p{i}", "stage": "1b", "question": f"Qv{i}", "answer": f"Av{i}"}
        for i in range(20)
    ]
    _write_jsonl(coll_dir / "training.jsonl", kvps_train)
    _write_jsonl(coll_dir / "validation.jsonl", kvps_val)

    result = run_split(coll_dir, fraction=0.1, seed=42)

    assert (coll_dir / "test_kvp_uids.json").exists()
    assert (coll_dir / "adapter_train.jsonl").exists()
    assert (coll_dir / "adapter_val.jsonl").exists()
    assert (coll_dir / "test_set.jsonl").exists()

    manifest = json.loads((coll_dir / "test_kvp_uids.json").read_text())
    test_uids = manifest["test_kvp_uids"]
    assert manifest["seed"] == 42
    assert manifest["fraction"] == 0.1
    # 10% of 40 = 4 (2 per stage, since 20 KVPs per stage)
    assert len(test_uids) == 4

    test_set = [json.loads(l) for l in (coll_dir / "test_set.jsonl").read_text().splitlines() if l]
    train = [json.loads(l) for l in (coll_dir / "adapter_train.jsonl").read_text().splitlines() if l]
    val = [json.loads(l) for l in (coll_dir / "adapter_val.jsonl").read_text().splitlines() if l]
    assert len(test_set) == 4
    # Remaining: 36 KVPs, split 95/5 → 34 + 2 approximately
    assert len(train) + len(val) == 36
    # KVP-based property: every passage should appear in remaining (train+val)
    passages_in_remaining = {k["passage_id"] for k in train + val}
    assert len(passages_in_remaining) == 20  # all 20 passages still represented
    assert result["test_set_size"] == 4


def test_pick_test_kvp_uids_swap_target_with_pid_total_one():
    """Regression: swap candidates must not deplete their own passage.

    Construct a corpus where the natural swap target (sorted first in the
    stage pool) has pid_total=1. The fix requires the swap to skip that
    candidate and select the next one with pid_total > 1 instead.

    Corpus layout:
      p0: 1 KVP  (p0_q0) — can be sampled initially, triggering a swap
      p1: 1 KVP  (p1_q0) — sorted before p2 uids; buggy code picks this
      p2: 2 KVPs (p2_q0, p2_q1) — correct swap target

    With fraction=0.25 and n=1, if p0_q0 is initially sampled it needs a
    swap (pid_selected[p0]=1 == pid_total[p0]=1). The buggy code finds
    p1_q0 first (alphabetically); the fixed code skips p1 (pid_total=1)
    and picks p2_q0 instead.  seed=2 reliably selects p0_q0 initially.
    """
    kvps = [
        {"kvp_uid": "p0_q0", "passage_id": "p0", "stage": "1a"},
        {"kvp_uid": "p1_q0", "passage_id": "p1", "stage": "1a"},
        {"kvp_uid": "p2_q0", "passage_id": "p2", "stage": "1a"},
        {"kvp_uid": "p2_q1", "passage_id": "p2", "stage": "1a"},
    ]
    # seed=2 with fraction=0.25 → n=1, p0_q0 initially sampled → swap fires
    for seed in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        picked = pick_test_kvp_uids(kvps, fraction=0.25, seed=seed)
        assert "p1_q0" not in picked, (
            f"seed={seed}: p1_q0 selected — would fully deplete passage p1. "
            f"Swap candidate filter must skip single-KVP passages."
        )
