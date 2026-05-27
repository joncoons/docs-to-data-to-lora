"""Stage 3 pre-training: hold out 10% of KVPs (stratified by stage) as a test set.

Domain-expert adapters want maximum corpus coverage, so we hold out individual
KVPs rather than entire passages — every passage in the corpus still contributes
to adapter training. Reads Stage 2 outputs (training.jsonl, validation.jsonl)
and writes:
  - test_kvp_uids.json     manifest of held-out KVP UIDs + seed + fraction
  - adapter_train.jsonl    LoRA training input (KVPs not in test set)
  - adapter_val.jsonl      mid-training val signal (subset of above)
  - test_set.jsonl         held-out test KVPs

Source training.jsonl + validation.jsonl are preserved untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)


def kvp_uid(passage_id: str, question: str) -> str:
    """Stable deterministic 16-hex-char SHA1 of (passage_id || question)."""
    h = hashlib.sha1(f"{passage_id}||{question}".encode("utf-8")).hexdigest()
    return h[:16]


def pick_test_kvp_uids(kvps: list[dict], fraction: float, seed: int) -> list[str]:
    """Sample `fraction` of KVPs per stage (stratified), deterministic via seed.

    Each input KVP must have a `kvp_uid` and a `stage` field.

    Passage-coverage invariant: no passage is fully removed from the remainder.
    When a candidate UID would be the last KVP for a passage, it is swapped for
    the next available non-colliding UID from the same stage pool (preserving
    the per-stage count exactly). If no swap candidate exists, the UID is dropped.
    """
    # Build uid → passage_id map
    uid_to_pid: dict[str, str] = {k["kvp_uid"]: k.get("passage_id", "") for k in kvps}

    # Count total KVPs per passage (across all stages)
    pid_total: dict[str, int] = {}
    for k in kvps:
        pid = k.get("passage_id", "")
        pid_total[pid] = pid_total.get(pid, 0) + 1

    # Build per-stage sorted uid lists for swap candidates
    by_stage: dict[str, list[str]] = {}
    for k in kvps:
        by_stage.setdefault(k["stage"], []).append(k["kvp_uid"])
    by_stage_sorted: dict[str, list[str]] = {
        s: sorted(uids) for s, uids in by_stage.items()
    }
    uid_to_stage: dict[str, str] = {k["kvp_uid"]: k["stage"] for k in kvps}

    rng = random.Random(seed)
    # First pass: sample per stage
    per_stage_sampled: dict[str, list[str]] = {}
    for stage in sorted(by_stage_sorted.keys()):
        uids = by_stage_sorted[stage]
        n = max(1, int(len(uids) * fraction))
        per_stage_sampled[stage] = list(rng.sample(uids, min(n, len(uids))))

    # Aggregate candidates; track how many KVPs per passage would be selected
    all_candidates: list[str] = []
    for stage in sorted(per_stage_sampled.keys()):
        all_candidates.extend(per_stage_sampled[stage])

    pid_selected: dict[str, int] = {}
    for uid in all_candidates:
        pid = uid_to_pid[uid]
        pid_selected[pid] = pid_selected.get(pid, 0) + 1

    # Second pass: resolve passages that would be fully depleted by swapping
    # out the last-selected UID for an unselected UID from the same stage.
    picked_set: set[str] = set(all_candidates)
    final: list[str] = []
    for uid in all_candidates:
        pid = uid_to_pid[uid]
        if pid_selected.get(pid, 0) >= pid_total.get(pid, 1):
            # This uid would fully deplete the passage — find a swap from same stage
            stage = uid_to_stage[uid]
            swap: str | None = None
            for candidate_uid in by_stage_sorted[stage]:
                if (candidate_uid not in picked_set
                        and uid_to_pid[candidate_uid] != pid
                        and pid_total.get(uid_to_pid[candidate_uid], 1) > 1):
                    swap = candidate_uid
                    break
            if swap is not None:
                picked_set.discard(uid)
                picked_set.add(swap)
                pid_selected[pid] -= 1
                swap_pid = uid_to_pid[swap]
                pid_selected[swap_pid] = pid_selected.get(swap_pid, 0) + 1
                final.append(swap)
            else:
                # No valid swap available; drop to preserve invariant
                picked_set.discard(uid)
                pid_selected[pid] -= 1
        else:
            final.append(uid)

    return sorted(final)


def partition_kvps(
    kvps: list[dict], test_kvp_uids: set[str]
) -> tuple[list[dict], list[dict]]:
    """Split kvps into (test_set, remaining) by membership in test_kvp_uids."""
    test_set: list[dict] = []
    remaining: list[dict] = []
    for k in kvps:
        if k.get("kvp_uid") in test_kvp_uids:
            test_set.append(k)
        else:
            remaining.append(k)
    return test_set, remaining


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _attach_uids(kvps: list[dict]) -> None:
    """Mutate kvps in-place: add kvp_uid field derived from (passage_id, question)."""
    for k in kvps:
        if "kvp_uid" not in k:
            k["kvp_uid"] = kvp_uid(k.get("passage_id", ""), k.get("question", ""))


def _enrich_from_stage2_eval(kvps: list[dict], collection_dir: Path) -> None:
    """Mutate kvps in-place: attach stage/passage_id/question from stage2_eval.jsonl.

    training.jsonl and validation.jsonl are written in the simplified customizer
    format {prompt, completion, system} — they lack stage, passage_id, and question.
    stage2_eval.jsonl is the full-metadata union of all curated KVPs; we join on
    question == prompt to recover those fields.

    The original customizer dict is stashed in _customizer_row so output files can
    be re-serialised without metadata fields.

    Rows that already carry stage/passage_id (e.g. in tests) are left unchanged
    except for the _customizer_row stash, which falls back to a minimal dict built
    from question/answer if prompt/completion are absent.
    """
    # Separate rows that already have metadata vs those needing enrichment
    needs_enrich = [k for k in kvps if "stage" not in k]
    already_enriched = [k for k in kvps if "stage" in k]

    # Rows that already have metadata: stash a _customizer_row preserving all
    # original fields (test fixtures use question/answer/passage_id directly).
    for k in already_enriched:
        k["_customizer_row"] = {key: val for key, val in k.items()
                                if not key.startswith("_")}
        k.setdefault("question", k.get("prompt", ""))
        k.setdefault("passage_id", "")

    if not needs_enrich:
        return

    stage2_path = collection_dir / "stage2_eval.jsonl"
    if not stage2_path.exists():
        log.warning("stage2_eval.jsonl not found in %s; stage/passage_id will be 'unknown'",
                    collection_dir)
        for k in needs_enrich:
            k["stage"] = "unknown"
            k["passage_id"] = ""
            k["question"] = k.get("prompt", "")
            k["_customizer_row"] = {
                "prompt": k["prompt"],
                "completion": k["completion"],
                "system": k.get("system", ""),
            }
        return

    meta_by_question: dict[str, dict] = {}
    for raw in _read_jsonl(stage2_path):
        q = raw.get("question", "")
        if q:
            meta_by_question[q] = raw

    missing = 0
    for k in needs_enrich:
        # Stash the customizer-format row before enriching
        k["_customizer_row"] = {
            "prompt": k["prompt"],
            "completion": k["completion"],
            "system": k.get("system", ""),
        }
        prompt = k.get("prompt", "")
        meta = meta_by_question.get(prompt)
        if meta:
            k["stage"] = meta["stage"]
            k["passage_id"] = meta.get("passage_id", "")
            k["question"] = meta["question"]
        else:
            k["stage"] = "unknown"
            k["passage_id"] = ""
            k["question"] = prompt
            missing += 1

    if missing:
        log.warning("%d KVPs had no match in stage2_eval.jsonl; stage='unknown'", missing)


def _to_customizer_rows(kvps: list[dict]) -> list[dict]:
    """Return each KVP as its original {prompt, completion, system} customizer dict."""
    return [k["_customizer_row"] for k in kvps]


def run_split(
    collection_dir: Path,
    fraction: float = 0.10,
    seed: int = 42,
    val_fraction_of_remainder: float = 0.05,
) -> dict:
    """Execute the KVP-based holdout split."""
    train_in = _read_jsonl(collection_dir / "training.jsonl")
    val_in = _read_jsonl(collection_dir / "validation.jsonl")
    all_kvps = train_in + val_in
    if not all_kvps:
        raise FileNotFoundError(f"No training.jsonl / validation.jsonl in {collection_dir}")

    # Enrich with stage/passage_id metadata before uid assignment (needs question field)
    _enrich_from_stage2_eval(all_kvps, collection_dir)
    _attach_uids(all_kvps)
    log.info("Loaded %d KVPs total (train=%d, val=%d)",
             len(all_kvps), len(train_in), len(val_in))

    test_uids = pick_test_kvp_uids(all_kvps, fraction=fraction, seed=seed)
    test_uid_set = set(test_uids)
    log.info("Held-out %d KVPs (seed=%d, fraction=%.2f), stratified by stage",
             len(test_uids), seed, fraction)

    test_set, remainder = partition_kvps(all_kvps, test_uid_set)

    # 95/5 split of remainder, stratified by stage
    rng = random.Random(seed)
    by_stage: dict[str, list[dict]] = {}
    for k in remainder:
        by_stage.setdefault(k.get("stage", "unknown"), []).append(k)
    new_train: list[dict] = []
    new_val: list[dict] = []
    for stage, items in by_stage.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction_of_remainder))
        new_val.extend(items[:n_val])
        new_train.extend(items[n_val:])
    rng.shuffle(new_train)
    rng.shuffle(new_val)

    # adapter_train / adapter_val must be in customizer {prompt,completion,system} format
    _write_jsonl(collection_dir / "adapter_train.jsonl", _to_customizer_rows(new_train))
    _write_jsonl(collection_dir / "adapter_val.jsonl", _to_customizer_rows(new_val))
    # test_set retains full metadata for eval scripts
    _write_jsonl(collection_dir / "test_set.jsonl", [k["_customizer_row"] for k in test_set])
    (collection_dir / "test_kvp_uids.json").write_text(
        json.dumps({"seed": seed, "fraction": fraction,
                    "test_kvp_uids": test_uids}, indent=2)
    )

    # Sanity: count distinct passages in remaining vs original
    passages_in_remaining = {k.get("passage_id") for k in (new_train + new_val)}
    passages_total = {k.get("passage_id") for k in all_kvps}
    log.info("Wrote adapter_train (%d) / adapter_val (%d) / test_set (%d). "
             "Passages in remaining: %d / %d",
             len(new_train), len(new_val), len(test_set),
             len(passages_in_remaining), len(passages_total))
    return {
        "test_set_size": len(test_set),
        "adapter_train_size": len(new_train),
        "adapter_val_size": len(new_val),
        "held_out_kvps": len(test_uids),
        "passages_in_remaining": len(passages_in_remaining),
        "passages_total": len(passages_total),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-dir", type=Path, required=True)
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    result = run_split(args.collection_dir, args.fraction, args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
