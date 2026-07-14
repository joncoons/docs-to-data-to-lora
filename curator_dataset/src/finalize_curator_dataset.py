#!/usr/bin/env python3
"""Deduplicate, balance, and source-group split the Curator QA dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_url_documents import require_experiment_path, sha256_file, write_json_atomic, write_jsonl_atomic


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalized_question(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("prompt") or "").casefold().split())


def dedupe_questions(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    selected: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: str(item.get("sample_id") or "")):
        selected.setdefault(normalized_question(row), row)
    deduped = sorted(selected.values(), key=lambda item: str(item.get("sample_id") or ""))
    return deduped, len(rows) - len(deduped)


def round_robin_cap(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        segment_id = str((row.get("lineage") or {}).get("segment_id"))
        by_segment[segment_id].append(row)
    queues = [
        sorted(segment_rows, key=lambda item: str(item.get("sample_id") or ""))
        for _, segment_rows in sorted(by_segment.items())
    ]
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for queue in queues:
            if offset < len(queue):
                selected.append(queue[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def balance_by_document(
    rows: list[dict[str, Any]], limit: int
) -> tuple[list[dict[str, Any]], int]:
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_document[str((row.get("lineage") or {}).get("document_id"))].append(row)
    balanced: list[dict[str, Any]] = []
    for document_id in sorted(by_document):
        balanced.extend(round_robin_cap(by_document[document_id], limit))
    return balanced, len(rows) - len(balanced)


def split_documents(
    rows: list[dict[str, Any]], validation_ratio: float, seed: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    document_ids = sorted(
        {str((row.get("lineage") or {}).get("document_id")) for row in rows},
        key=lambda value: hashlib.sha256(f"{seed}\n{value}".encode()).hexdigest(),
    )
    validation_count = max(1, round(len(document_ids) * validation_ratio))
    validation_ids = set(document_ids[:validation_count])
    training = [
        row
        for row in rows
        if str((row.get("lineage") or {}).get("document_id")) not in validation_ids
    ]
    validation = [
        row
        for row in rows
        if str((row.get("lineage") or {}).get("document_id")) in validation_ids
    ]
    training.sort(key=lambda item: str(item.get("sample_id") or ""))
    validation.sort(key=lambda item: str(item.get("sample_id") or ""))
    return training, validation, sorted(set(document_ids) - validation_ids), sorted(validation_ids)


def customizer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": row["prompt"],
            "completion": row["completion"],
            "system": row["system"],
        }
        for row in rows
    ]


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "rows": sum(1 for line in path.read_text().splitlines() if line.strip()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs-per-document", type=int, default=32)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", default="curator-nim-source-split-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = require_experiment_path(args.input)
    output_dir = require_experiment_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_rows = read_jsonl(input_path)
    deduped, duplicate_questions_dropped = dedupe_questions(source_rows)
    balanced, cap_dropped = balance_by_document(deduped, args.max_pairs_per_document)
    training, validation, training_ids, validation_ids = split_documents(
        balanced, args.validation_ratio, args.split_seed
    )
    if set(training_ids) & set(validation_ids):
        raise AssertionError("document leakage between training and validation")

    paths = {
        "qa_samples": output_dir / "qa_samples.jsonl",
        "training_samples": output_dir / "training_samples.jsonl",
        "validation_samples": output_dir / "validation_samples.jsonl",
        "training": output_dir / "training.jsonl",
        "validation": output_dir / "validation.jsonl",
    }
    write_jsonl_atomic(paths["qa_samples"], balanced)
    write_jsonl_atomic(paths["training_samples"], training)
    write_jsonl_atomic(paths["validation_samples"], validation)
    write_jsonl_atomic(paths["training"], customizer_rows(training))
    write_jsonl_atomic(paths["validation"], customizer_rows(validation))

    manifest = {
        "schema_version": "curator_dataset.final_dataset_manifest.v1",
        "created_at": utc_now(),
        "source": artifact(input_path),
        "policy": {
            "question_deduplication": "global casefolded whitespace-normalized exact match",
            "max_pairs_per_document": args.max_pairs_per_document,
            "cap_sampling": "deterministic round-robin across Curator segments",
            "validation_ratio": args.validation_ratio,
            "split_seed": args.split_seed,
            "split_group": "lineage.document_id",
        },
        "metrics": {
            "source_rows": len(source_rows),
            "duplicate_question_rows_dropped": duplicate_questions_dropped,
            "document_cap_rows_dropped": cap_dropped,
            "final_rows": len(balanced),
            "documents": len(training_ids) + len(validation_ids),
            "training_documents": len(training_ids),
            "validation_documents": len(validation_ids),
            "training_rows": len(training),
            "validation_rows": len(validation),
            "document_leakage": 0,
        },
        "outputs": {name: artifact(path) for name, path in paths.items()},
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
