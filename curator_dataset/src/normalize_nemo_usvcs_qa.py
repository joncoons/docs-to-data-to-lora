#!/usr/bin/env python3
"""Normalize Curator QA output with a NeMo Microservices system identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_url_documents import require_experiment_path, sha256_file, write_json_atomic, write_jsonl_atomic
from normalize_curator_qa import normalize_rows, read_raw_rows, utc_now


SYSTEM_PROMPT = (
    "You are a helpful technical assistant specializing in NVIDIA NeMo "
    "Microservices documentation. Answer accurately and concisely."
)


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
    for sample in samples:
        sample["system"] = SYSTEM_PROMPT
        sample["lineage"]["collection"] = "nemo_usvcs_curated"
    write_jsonl_atomic(output, samples)
    manifest = {
        "schema_version": "curator_dataset.qa_dataset_manifest.v1",
        "created_at": utc_now(),
        "collection": "nemo_usvcs_curated",
        "raw_dir": str(raw_dir),
        "system_prompt": SYSTEM_PROMPT,
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
