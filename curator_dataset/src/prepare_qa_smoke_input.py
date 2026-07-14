#!/usr/bin/env python3
"""Select a deterministic, length-bounded smoke sample for Curator QA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_url_documents import EXPERIMENT_ROOT, require_experiment_path, write_jsonl_atomic


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_rank(row: dict[str, Any], seed: str) -> str:
    value = f"{seed}\n{row.get('document_id') or row.get('id')}"
    return hashlib.sha256(value.encode()).hexdigest()


def select_smoke_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    min_words: int,
    max_words: int,
    seed: str,
) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows if min_words <= int(row.get("word_count") or 0) <= max_words
    ]
    selected = sorted(eligible, key=lambda row: stable_rank(row, seed))[:count]
    if len(selected) != count:
        raise ValueError(f"requested {count} rows but only {len(selected)} are eligible")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            EXPERIMENT_ROOT / "data" / "nim_curated" / "url_documents.curator_ready.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "data" / "nim_curated" / "qa" / "smoke_input.jsonl",
    )
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=900)
    parser.add_argument("--seed", default="curator-diverse-qa-smoke-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = require_experiment_path(args.output)
    selected = select_smoke_rows(
        read_jsonl(args.input),
        count=args.count,
        min_words=args.min_words,
        max_words=args.max_words,
        seed=args.seed,
    )
    write_jsonl_atomic(output, selected)
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(selected),
                "document_ids": [row["document_id"] for row in selected],
                "word_counts": [row["word_count"] for row in selected],
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
