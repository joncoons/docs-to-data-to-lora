#!/usr/bin/env python3
"""Normalize raw Curator DiverseQA responses into one SFT row per QA pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_url_documents import require_experiment_path, sha256_file, write_json_atomic, write_jsonl_atomic


PAIR_RE = re.compile(
    r"(?:^|\n)\s*(?:-\s*)?Question:\s*(.*?)\s+Answer:\s*(.*?)"
    r"(?=(?:\n\s*(?:-\s*)?Question:)|\Z)",
    re.DOTALL,
)
SYSTEM_PROMPT = (
    "You are a helpful technical assistant specializing in NVIDIA NIM documentation. "
    "Answer accurately and concisely."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_pairs(response: str) -> list[tuple[str, str]]:
    return [
        (clean(match.group(1)), clean(match.group(2)))
        for match in PAIR_RE.finditer(response or "")
        if clean(match.group(1)) and clean(match.group(2))
    ]


def read_raw_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.rglob("*.jsonl")):
        with path.open() as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def normalize_rows(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    parsed_segments = 0
    parse_failures = 0
    exact_duplicates = 0
    for raw in raw_rows:
        pairs = parse_pairs(str(raw.get("diverse_qa") or ""))
        if not pairs:
            parse_failures += 1
            continue
        parsed_segments += 1
        for pair_index, (question, answer) in enumerate(pairs):
            key = (question.casefold(), answer.casefold())
            if key in seen:
                exact_duplicates += 1
                continue
            seen.add(key)
            document_id = str(raw.get("document_id") or raw.get("id") or "")
            segment_id = raw.get("segment_id")
            sample_material = f"{document_id}\n{segment_id}\n{question}\n{answer}"
            sample_id = f"curator_qa_{hashlib.sha256(sample_material.encode()).hexdigest()[:24]}"
            normalized.append(
                {
                    "schema_version": "curator_dataset.qa_sample.v1",
                    "sample_id": sample_id,
                    "origin": "curator_diverse_qa",
                    "task_type": "qa",
                    "prompt": question,
                    "completion": answer,
                    "system": SYSTEM_PROMPT,
                    "context": raw.get("text"),
                    "lineage": {
                        "collection": raw.get("collection"),
                        "document_id": document_id,
                        "url": raw.get("url"),
                        "segment_id": segment_id,
                        "pair_index": pair_index,
                        "es_chunk_ids": raw.get("es_chunk_ids") or [],
                        "source_chunk_ids": raw.get("source_chunk_ids") or [],
                        "source_revision_ids": raw.get("source_revision_ids") or [],
                    },
                    "generation": {
                        "framework": "NeMo Curator",
                        "stage": "Nemotron-CC DiverseQAStage",
                        "model": "nvidia/nvidia/nemotron-3-super-v3",
                    },
                }
            )
    return normalized, {
        "raw_segments": len(raw_rows),
        "parsed_segments": parsed_segments,
        "parse_failures": parse_failures,
        "qa_pairs": len(normalized),
        "exact_duplicate_pairs_dropped": exact_duplicates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = require_experiment_path(args.raw_dir)
    output = require_experiment_path(args.output)
    manifest_path = require_experiment_path(args.manifest)
    samples, metrics = normalize_rows(read_raw_rows(raw_dir))
    if not samples:
        raise RuntimeError(f"no QA pairs parsed from {raw_dir}")
    write_jsonl_atomic(output, samples)
    manifest = {
        "schema_version": "curator_dataset.qa_dataset_manifest.v1",
        "created_at": utc_now(),
        "raw_dir": str(raw_dir),
        "metrics": metrics,
        "output": {
            "path": str(output),
            "rows": len(samples),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
