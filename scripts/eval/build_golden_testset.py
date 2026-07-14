#!/usr/bin/env python3
"""Build immutable HTML-only golden test sets for Stage 3 evaluation.

The builder starts from the prior held-out test sets, then removes rows that
would leak into the current experiment by exact prompt overlap against any
current LE or Curator train/validation file. It also enforces the current
HTML-only provenance rule by requiring docs.nvidia.com context and excluding
PDF/no-context/non-doc rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "curator_dataset/experiments/20260709-curator-vs-le"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_ROOT / "golden_eval/golden-v1"

CURRENT_TRAIN_VAL_PATHS = [
    REPO_ROOT / "curator_dataset/experiments/20260709-le-uncapped-batch/runs/nim_curated/super-v3/training.jsonl",
    REPO_ROOT / "curator_dataset/experiments/20260709-le-uncapped-batch/runs/nim_curated/super-v3/validation.jsonl",
    REPO_ROOT / "curator_dataset/data/nim_curated/qa/runs/full-20260629t2032z-retry1/dataset/final/training.jsonl",
    REPO_ROOT / "curator_dataset/data/nim_curated/qa/runs/full-20260629t2032z-retry1/dataset/final/validation.jsonl",
    REPO_ROOT / "curator_dataset/experiments/20260709-le-uncapped-batch/runs/nemo_usvcs_curated/super-v3/training.jsonl",
    REPO_ROOT / "curator_dataset/experiments/20260709-le-uncapped-batch/runs/nemo_usvcs_curated/super-v3/validation.jsonl",
    REPO_ROOT / "curator_dataset/data/nemo_usvcs_curated/qa/runs/full-20260629t2120z/dataset/final/training.jsonl",
    REPO_ROOT / "curator_dataset/data/nemo_usvcs_curated/qa/runs/full-20260629t2120z/dataset/final/validation.jsonl",
]

SOURCE_TESTSETS = {
    "nim_curated": {
        "plain": Path("/mnt/nvme2/peft/datasets/experiments/nim_curated_dd_deterministic_5x_combined_20260605/test_set.jsonl"),
        "with_context": Path("/mnt/nvme2/peft/datasets/experiments/nim_curated_dd_deterministic_5x_combined_20260605/test_set_with_context.jsonl"),
    },
    "nemo_usvcs_curated": {
        "plain": Path("/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_deterministic_5x_combined_20260605/test_set.jsonl"),
        "with_context": Path("/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_deterministic_5x_combined_20260605/test_set_with_context.jsonl"),
    },
}

PDF_RE = re.compile(r"(?i)\.pdf\b|\.pdf\s")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_prompt(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def load_current_prompt_set(paths: list[Path]) -> tuple[set[str], dict[str, Any]]:
    prompts: set[str] = set()
    files: list[dict[str, Any]] = []
    for path in paths:
        rows = 0
        if path.exists():
            for row in iter_jsonl(path):
                prompts.add(normalize_prompt(str(row.get("prompt") or "")))
                rows += 1
        files.append({
            "path": str(path),
            "exists": path.exists(),
            "rows": rows,
            "sha256": sha256_file(path) if path.exists() else None,
        })
    prompt_fingerprint = sha256_bytes("\n".join(sorted(prompts)).encode("utf-8"))
    return prompts, {
        "files": files,
        "unique_normalized_prompts": len(prompts),
        "normalized_prompt_set_sha256": prompt_fingerprint,
    }


def exclusion_reason(row: dict[str, Any], context_row: dict[str, Any], current_prompts: set[str]) -> str | None:
    prompt = normalize_prompt(str(row.get("prompt") or ""))
    context_prompt = str(context_row.get("prompt") or "")
    if prompt in current_prompts:
        return "current_train_or_validation_prompt_overlap"
    if PDF_RE.search(context_prompt):
        return "pdf_context"
    if "(no relevant context found)" in context_prompt.lower():
        return "no_relevant_context"
    if "https://docs.nvidia.com/" not in context_prompt:
        return "non_docs_nvidia_context"
    return None


def golden_id(corpus: str, source_index: int, row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "corpus": corpus,
            "source_index": source_index,
            "prompt": row.get("prompt"),
            "completion": row.get("completion"),
        },
        sort_keys=True,
    )
    return f"golden_{sha256_bytes(payload.encode('utf-8'))[:24]}"


def enrich(row: dict[str, Any], *, corpus: str, source_index: int, context_baked: bool) -> dict[str, Any]:
    out = dict(row)
    out["golden_id"] = golden_id(corpus, source_index, row)
    out["golden"] = {
        "schema_version": "stage3-golden-row/v1",
        "version": "golden-v1",
        "corpus": corpus,
        "source_row_index": source_index,
        "context_baked": context_baked,
        "html_only": True,
        "leakage_filter": "exact_normalized_prompt_not_in_current_train_or_validation",
    }
    return out


def build_one(corpus: str, paths: dict[str, Path], out_root: Path, current_prompts: set[str]) -> dict[str, Any]:
    plain_rows = read_jsonl(paths["plain"])
    context_rows = read_jsonl(paths["with_context"])
    if len(plain_rows) != len(context_rows):
        raise ValueError(f"{corpus}: source row mismatch: plain={len(plain_rows)} with_context={len(context_rows)}")

    kept_plain: list[dict[str, Any]] = []
    kept_context: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    for idx, (row, context_row) in enumerate(zip(plain_rows, context_rows)):
        reason = exclusion_reason(row, context_row, current_prompts)
        if reason:
            reasons[reason] += 1
            excluded.append({
                "source_row_index": idx,
                "reason": reason,
                "prompt_sha256": sha256_bytes(str(row.get("prompt") or "").encode("utf-8")),
            })
            continue
        kept_plain.append(enrich(row, corpus=corpus, source_index=idx, context_baked=False))
        kept_context.append(enrich(context_row, corpus=corpus, source_index=idx, context_baked=True))

    corpus_dir = out_root / corpus
    plain_out = corpus_dir / "test_set.jsonl"
    context_out = corpus_dir / "test_set_with_context.jsonl"
    excluded_out = corpus_dir / "excluded_rows.jsonl"
    write_jsonl(plain_out, kept_plain)
    write_jsonl(context_out, kept_context)
    write_jsonl(excluded_out, excluded)

    manifest = {
        "schema_version": "stage3-golden-corpus-manifest/v1",
        "version": "golden-v1",
        "corpus": corpus,
        "source": {
            "plain": {"path": str(paths["plain"]), "rows": len(plain_rows), "sha256": sha256_file(paths["plain"])},
            "with_context": {
                "path": str(paths["with_context"]),
                "rows": len(context_rows),
                "sha256": sha256_file(paths["with_context"]),
            },
        },
        "filters": {
            "html_only": "Requires docs.nvidia.com context and excludes .pdf/no-context/non-doc rows.",
            "leakage": "Excludes exact normalized prompt matches from current LE and Curator train/validation files.",
        },
        "counts": {
            "source_rows": len(plain_rows),
            "kept_rows": len(kept_plain),
            "excluded_rows": len(excluded),
            "excluded_by_reason": dict(sorted(reasons.items())),
        },
        "outputs": {
            "test_set_jsonl": {"path": str(plain_out), "rows": len(kept_plain), "sha256": sha256_file(plain_out)},
            "test_set_with_context_jsonl": {
                "path": str(context_out),
                "rows": len(kept_context),
                "sha256": sha256_file(context_out),
            },
            "excluded_rows_jsonl": {"path": str(excluded_out), "rows": len(excluded), "sha256": sha256_file(excluded_out)},
        },
    }
    write_json(corpus_dir / "manifest.json", manifest)
    return manifest


def write_checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    lines = []
    for path in files:
        rel = path.relative_to(root)
        lines.append(f"{sha256_file(path)}  {rel.as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(output_dir: Path) -> dict[str, Any]:
    current_prompts, current_manifest = load_current_prompt_set(CURRENT_TRAIN_VAL_PATHS)
    corpus_manifests = {
        corpus: build_one(corpus, paths, output_dir, current_prompts)
        for corpus, paths in SOURCE_TESTSETS.items()
    }
    manifest = {
        "schema_version": "stage3-golden-manifest/v1",
        "version": "golden-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Immutable HTML-only golden test sets for LE vs Curator and Llama 3.3 70B "
            "reference evaluation."
        ),
        "current_train_validation_prompt_set": current_manifest,
        "corpora": corpus_manifests,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_checksums(output_dir)
    manifest["checksums_sha256"] = str(output_dir / "checksums.sha256")
    write_json(output_dir / "manifest.json", manifest)
    write_checksums(output_dir)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    manifest = build(args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "corpora": {
            corpus: data["counts"]
            for corpus, data in manifest["corpora"].items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
