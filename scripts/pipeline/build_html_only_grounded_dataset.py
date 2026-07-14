#!/usr/bin/env python3
"""Build an HTML-only grounded dataset from existing lineage artifacts.

This does not call an LLM. It reconstructs a new versioned dataset from:

- passages.jsonl, filtering to doc_kind == "html"
- stage2_eval.jsonl, filtering rows by retained passage_id

The final Customizer splits are regenerated from metadata-rich KVP rows so
lineage is not recovered through question-only joins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.config import Config  # noqa: E402
from scripts.pipeline.dataset_admission import (  # noqa: E402
    admitted_dataset_samples_from_kvp_rows,
)
from scripts.pipeline.finalize_dataset import finalize_dataset  # noqa: E402
from scripts.pipeline.models import KVPRow, Passage  # noqa: E402
from scripts.pipeline.provenance_io import write_jsonl  # noqa: E402
from scripts.pipeline.stage3_curator import (  # noqa: E402
    answer_subset_of_question_filter,
    exact_dedup,
    length_filter,
    minhash_dedup,
    to_customizer_format,
    train_val_split,
)

log = logging.getLogger(__name__)


def kvp_uid(passage_id: str, question: str) -> str:
    return hashlib.sha1(f"{passage_id}||{question}".encode("utf-8")).hexdigest()[:16]


def read_passages(path: Path) -> list[Passage]:
    return [
        Passage.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def read_kvps(path: Path) -> list[KVPRow]:
    return [
        KVPRow.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_customizer_jsonl(path: Path, rows: list[KVPRow], system_prompt: str) -> None:
    with path.open("w") as f:
        for row in to_customizer_format(rows, system_prompt):
            f.write(json.dumps(row) + "\n")


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def compute_local_bias_report(
    passages: list[Passage],
    kvps: list[KVPRow],
    *,
    threshold_factor: float,
) -> dict[str, Any]:
    chunk_counts = Counter(p.product_family or "unknown" for p in passages)
    kvp_counts = Counter(row.product_family or "unknown" for row in kvps)
    densities = [
        kvp_counts.get(domain_slice, 0) / chunk_count
        for domain_slice, chunk_count in chunk_counts.items()
        if chunk_count
    ]
    sorted_densities = sorted(densities)
    median_density = (
        sorted_densities[len(sorted_densities) // 2] if sorted_densities else 0.0
    )
    threshold = median_density * threshold_factor
    domain_slices = []
    legacy_products = []
    for domain_slice in sorted(chunk_counts):
        chunk_count = chunk_counts[domain_slice]
        kvp_count = kvp_counts.get(domain_slice, 0)
        density = kvp_count / chunk_count if chunk_count else 0.0
        row = {
            "domain_slice": domain_slice,
            "chunk_count": chunk_count,
            "kvp_count": kvp_count,
            "density": round(density, 4),
            "underrepresented": density < threshold,
        }
        domain_slices.append(row)
        legacy_products.append({"product_family": domain_slice, **{k: v for k, v in row.items() if k != "domain_slice"}})
    return {
        "median_density": round(median_density, 4),
        "threshold": round(threshold, 4),
        "threshold_factor": threshold_factor,
        "domain_slices": domain_slices,
        # Backward-compatible alias for older local artifacts and callers.
        "products": legacy_products,
    }


def filter_jsonl_by_passage_id(
    source_path: Path,
    output_path: Path,
    html_passage_ids: set[str],
) -> dict[str, int]:
    if not source_path.exists():
        return {"source_rows": 0, "output_rows": 0}
    source_rows = 0
    output_rows = 0
    with source_path.open() as src, output_path.open("w") as out:
        for line in src:
            if not line.strip():
                continue
            source_rows += 1
            row = json.loads(line)
            if row.get("passage_id") in html_passage_ids:
                out.write(json.dumps(row) + "\n")
                output_rows += 1
    return {"source_rows": source_rows, "output_rows": output_rows}


def pick_test_uids(rows: list[KVPRow], fraction: float, seed: int) -> list[str]:
    """Stratified deterministic holdout while preserving passage coverage."""
    row_dicts = [
        {
            "kvp_uid": kvp_uid(row.passage_id, row.question),
            "passage_id": row.passage_id,
            "stage": row.stage,
        }
        for row in rows
    ]
    uid_to_pid = {row["kvp_uid"]: row["passage_id"] for row in row_dicts}
    uid_to_stage = {row["kvp_uid"]: row["stage"] for row in row_dicts}
    pid_total = Counter(row["passage_id"] for row in row_dicts)

    by_stage: dict[str, list[str]] = defaultdict(list)
    for row in row_dicts:
        by_stage[row["stage"]].append(row["kvp_uid"])
    by_stage = {stage: sorted(uids) for stage, uids in by_stage.items()}

    rng = random.Random(seed)
    sampled_by_stage: dict[str, list[str]] = {}
    for stage, uids in sorted(by_stage.items()):
        n = max(1, int(len(uids) * fraction))
        sampled_by_stage[stage] = list(rng.sample(uids, min(n, len(uids))))

    candidates: list[str] = []
    for stage in sorted(sampled_by_stage):
        candidates.extend(sampled_by_stage[stage])

    pid_selected = Counter(uid_to_pid[uid] for uid in candidates)
    picked = set(candidates)
    final: list[str] = []
    for uid in candidates:
        pid = uid_to_pid[uid]
        if pid_selected[pid] < pid_total[pid]:
            final.append(uid)
            continue

        stage = uid_to_stage[uid]
        swap: str | None = None
        for candidate_uid in by_stage[stage]:
            candidate_pid = uid_to_pid[candidate_uid]
            if (
                candidate_uid not in picked
                and candidate_pid != pid
                and pid_total[candidate_pid] > 1
            ):
                swap = candidate_uid
                break
        if swap is None:
            picked.discard(uid)
            pid_selected[pid] -= 1
            continue

        picked.discard(uid)
        picked.add(swap)
        pid_selected[pid] -= 1
        pid_selected[uid_to_pid[swap]] += 1
        final.append(swap)

    return sorted(final)


def make_adapter_splits(
    rows: list[KVPRow],
    *,
    test_fraction: float,
    val_fraction_of_remainder: float,
    seed: int,
) -> tuple[list[KVPRow], list[KVPRow], list[KVPRow], list[str]]:
    test_uids = set(pick_test_uids(rows, fraction=test_fraction, seed=seed))
    test_rows: list[KVPRow] = []
    remainder: list[KVPRow] = []
    for row in rows:
        if kvp_uid(row.passage_id, row.question) in test_uids:
            test_rows.append(row)
        else:
            remainder.append(row)

    rng = random.Random(seed)
    by_stage: dict[str, list[KVPRow]] = defaultdict(list)
    for row in remainder:
        by_stage[row.stage].append(row)

    adapter_train: list[KVPRow] = []
    adapter_val: list[KVPRow] = []
    for stage, items in sorted(by_stage.items()):
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction_of_remainder))
        adapter_val.extend(items[:n_val])
        adapter_train.extend(items[n_val:])

    rng.shuffle(adapter_train)
    rng.shuffle(adapter_val)
    return adapter_train, adapter_val, test_rows, sorted(test_uids)


def stage_counts(rows: list[KVPRow]) -> dict[str, int]:
    return dict(sorted(Counter(row.stage for row in rows).items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=Path("<DATASET_ROOT>/nim_curated"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("<DATASET_ROOT>/nim_curated_html_only"),
    )
    ap.add_argument("--collection", default="nim_curated_html_only")
    ap.add_argument("--source-collection", default="nim_curated")
    ap.add_argument("--test-fraction", type=float, default=0.10)
    ap.add_argument("--adapter-val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config()
    system_prompt = (
        "You are a precise NVIDIA NIM technical assistant. "
        "Answer based on official documentation."
    )

    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "provenance").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifests").mkdir(parents=True, exist_ok=True)

    source_passages = read_passages(args.source_dir / "passages.jsonl")
    html_passages = [p for p in source_passages if p.doc_kind == "html"]
    html_passage_ids = {p.passage_id for p in html_passages}
    write_jsonl(args.output_dir / "passages.jsonl", [p.model_dump() for p in html_passages])

    source_stage2 = read_kvps(args.source_dir / "stage2_eval.jsonl")
    html_stage2 = [row for row in source_stage2 if row.passage_id in html_passage_ids]
    write_jsonl(args.output_dir / "stage2_eval.jsonl", [row.model_dump() for row in html_stage2])

    filtered_artifacts = {}
    for rel in (
        "stage1a_le.jsonl",
        "stage1b_synthesis.jsonl",
        "stage1c_instruction.jsonl",
        "stage1_5_gapfill.jsonl",
        "stage2_dropped.jsonl",
    ):
        filtered_artifacts[rel] = filter_jsonl_by_passage_id(
            args.source_dir / rel,
            args.output_dir / rel,
            html_passage_ids,
        )

    curated = exact_dedup(html_stage2)
    after_exact = len(curated)
    curated = minhash_dedup(curated, threshold=cfg.minhash_threshold)
    after_minhash = len(curated)
    curated = length_filter(
        curated,
        min_q_tokens=cfg.min_question_tokens,
        min_a_tokens=cfg.min_answer_tokens,
    )
    after_length = len(curated)
    curated = answer_subset_of_question_filter(curated)
    after_subset = len(curated)

    training_rows, validation_rows = train_val_split(curated, train_ratio=cfg.train_val_split)
    write_customizer_jsonl(args.output_dir / "training.jsonl", training_rows, system_prompt)
    write_customizer_jsonl(args.output_dir / "validation.jsonl", validation_rows, system_prompt)

    adapter_train, adapter_val, test_rows, test_uids = make_adapter_splits(
        training_rows + validation_rows,
        test_fraction=args.test_fraction,
        val_fraction_of_remainder=args.adapter_val_fraction,
        seed=args.seed,
    )
    write_customizer_jsonl(args.output_dir / "adapter_train.jsonl", adapter_train, system_prompt)
    write_customizer_jsonl(args.output_dir / "adapter_val.jsonl", adapter_val, system_prompt)
    write_customizer_jsonl(args.output_dir / "test_set.jsonl", test_rows, system_prompt)
    write_manifest(
        args.output_dir / "test_kvp_uids.json",
        {
            "seed": args.seed,
            "fraction": args.test_fraction,
            "test_kvp_uids": test_uids,
        },
    )

    bias_report = compute_local_bias_report(
        html_passages,
        html_stage2,
        threshold_factor=cfg.bias_threshold_factor,
    )
    write_manifest(args.output_dir / "bias_report.json", bias_report)
    write_manifest(args.output_dir / "progress.json", {"completed_stages": ["0", "2", "3", "html_filter"]})

    write_jsonl(
        args.output_dir / "provenance" / "dataset_samples.jsonl",
        admitted_dataset_samples_from_kvp_rows(
            html_stage2,
            dataset_dir=args.output_dir,
            system_prompt=system_prompt,
        ),
    )

    manifest = {
        "collection": args.collection,
        "source_collection": args.source_collection,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "filter": {"doc_kind": "html"},
        "source_passages": len(source_passages),
        "html_passages": len(html_passages),
        "source_stage2_rows": len(source_stage2),
        "html_stage2_rows": len(html_stage2),
        "curator": {
            "input_rows": len(html_stage2),
            "after_exact_dedup": after_exact,
            "after_minhash_dedup": after_minhash,
            "after_length_filter": after_length,
            "after_subset_filter": after_subset,
            "stage_counts": stage_counts(curated),
        },
        "splits": {
            "training_rows": len(training_rows),
            "validation_rows": len(validation_rows),
            "adapter_train_rows": len(adapter_train),
            "adapter_val_rows": len(adapter_val),
            "test_rows": len(test_rows),
            "test_fraction": args.test_fraction,
            "adapter_val_fraction": args.adapter_val_fraction,
            "seed": args.seed,
        },
        "filtered_artifacts": filtered_artifacts,
    }
    write_manifest(args.output_dir / "manifests" / "html_only_filter_manifest.json", manifest)

    final_manifest = finalize_dataset(
        args.output_dir,
        dataset_name=args.collection,
        system_prompt=system_prompt,
        observability_dir=None,
    )
    log.info("HTML-only dataset written: %s", args.output_dir)
    log.info("Rows: %s", json.dumps(manifest["splits"], sort_keys=True))
    log.info("Dataset version: %s", final_manifest.get("dataset_version_id"))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
