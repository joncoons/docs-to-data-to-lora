#!/usr/bin/env python3
"""Apply explicit URL exclusions to an already reconstructed document list."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_url_documents import require_experiment_path, sha256_file, write_json_atomic, write_jsonl_atomic


DEFAULT_EXCLUSION = re.compile(r"acknowledg|(?:^|/)eula(?:[./]|$)", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def filter_rows(
    rows: list[dict[str, Any]], pattern: re.Pattern[str] = DEFAULT_EXCLUSION
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        url = str(row.get("url") or "")
        if pattern.search(url):
            excluded.append(
                {
                    "document_id": row.get("document_id"),
                    "url": url,
                    "word_count": row.get("word_count"),
                    "reason": "excluded_url_pattern",
                }
            )
        else:
            retained.append(row)
    return retained, excluded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = require_experiment_path(args.input)
    output = require_experiment_path(args.output)
    manifest_path = require_experiment_path(args.manifest)
    rows = read_jsonl(input_path)
    retained, excluded = filter_rows(rows)
    write_jsonl_atomic(output, retained)
    manifest = {
        "schema_version": "curator_dataset.url_filter_manifest.v1",
        "created_at": utc_now(),
        "input": {
            "path": str(input_path),
            "rows": len(rows),
            "sha256": sha256_file(input_path),
        },
        "policy": {
            "excluded_url_regex": DEFAULT_EXCLUSION.pattern,
            "case_insensitive": True,
        },
        "counts": {
            "retained": len(retained),
            "excluded": len(excluded),
        },
        "excluded": excluded,
        "output": {
            "path": str(output),
            "rows": len(retained),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
