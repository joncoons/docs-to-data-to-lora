#!/usr/bin/env python3
"""Build immutable golden test sets for Stage 3 evaluation.

The builder starts from held-out test rows, removes rows that would leak into
current train/validation data by exact normalized prompt overlap, and writes a
versioned test-set manifest with checksums. Source paths are supplied through a
small JSON manifest so the public tool is reusable for any domain corpus rather
than tied to one local experiment tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(os.getenv("GOLDEN_OUTPUT_DIR", "artifacts/evaluation/golden-v1"))
DEFAULT_SOURCE_MANIFEST = os.getenv("GOLDEN_SOURCE_MANIFEST")

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


def resolve_manifest_path(value: str | Path, *, manifest_dir: Path, dataset_root: Path | None) -> Path:
    raw = str(value)
    if dataset_root is not None:
        raw = raw.replace("<DATASET_ROOT>", str(dataset_root))
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if dataset_root is not None:
        candidate = dataset_root / path
        if candidate.exists():
            return candidate
    return manifest_dir / path


def load_source_manifest(path: Path) -> tuple[dict[str, dict[str, Path]], list[Path], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_dir = path.parent
    dataset_root_value = payload.get("dataset_root")
    dataset_root = None
    if dataset_root_value:
        dataset_root = resolve_manifest_path(dataset_root_value, manifest_dir=manifest_dir, dataset_root=None)

    source_testsets: dict[str, dict[str, Path]] = {}
    for corpus, spec in (payload.get("corpora") or {}).items():
        if not isinstance(spec, dict):
            raise ValueError(f"corpus {corpus!r} must map to a dict")
        try:
            plain = spec["plain"]
            with_context = spec["with_context"]
        except KeyError as exc:
            raise ValueError(f"corpus {corpus!r} requires plain and with_context paths") from exc
        source_testsets[corpus] = {
            "plain": resolve_manifest_path(plain, manifest_dir=manifest_dir, dataset_root=dataset_root),
            "with_context": resolve_manifest_path(with_context, manifest_dir=manifest_dir, dataset_root=dataset_root),
        }
    if not source_testsets:
        raise ValueError("source manifest must define at least one corpus under 'corpora'")

    current_paths = [
        resolve_manifest_path(item, manifest_dir=manifest_dir, dataset_root=dataset_root)
        for item in payload.get("current_train_validation_paths", [])
    ]
    return source_testsets, current_paths, payload


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


def exclusion_reason(
    row: dict[str, Any],
    context_row: dict[str, Any],
    current_prompts: set[str],
    *,
    exclude_pdf_context: bool,
    exclude_no_context: bool,
    required_context_substring: str | None,
) -> str | None:
    prompt = normalize_prompt(str(row.get("prompt") or ""))
    context_prompt = str(context_row.get("prompt") or "")
    if prompt in current_prompts:
        return "current_train_or_validation_prompt_overlap"
    if exclude_pdf_context and PDF_RE.search(context_prompt):
        return "pdf_context"
    if exclude_no_context and "(no relevant context found)" in context_prompt.lower():
        return "no_relevant_context"
    if required_context_substring and required_context_substring not in context_prompt:
        return "missing_required_context_substring"
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


def enrich(row: dict[str, Any], *, corpus: str, source_index: int, context_baked: bool, filters: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["golden_id"] = golden_id(corpus, source_index, row)
    out["golden"] = {
        "schema_version": "stage3-golden-row/v1",
        "version": "golden-v1",
        "corpus": corpus,
        "source_row_index": source_index,
        "context_baked": context_baked,
        "filters": filters,
    }
    return out


def build_one(
    corpus: str,
    paths: dict[str, Path],
    out_root: Path,
    current_prompts: set[str],
    *,
    exclude_pdf_context: bool,
    exclude_no_context: bool,
    required_context_substring: str | None,
) -> dict[str, Any]:
    plain_rows = read_jsonl(paths["plain"])
    context_rows = read_jsonl(paths["with_context"])
    if len(plain_rows) != len(context_rows):
        raise ValueError(f"{corpus}: source row mismatch: plain={len(plain_rows)} with_context={len(context_rows)}")

    filter_settings = {
        "exclude_pdf_context": exclude_pdf_context,
        "exclude_no_context": exclude_no_context,
        "required_context_substring": required_context_substring,
        "leakage_filter": "exact_normalized_prompt_not_in_current_train_or_validation",
    }
    kept_plain: list[dict[str, Any]] = []
    kept_context: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    for idx, (row, context_row) in enumerate(zip(plain_rows, context_rows)):
        reason = exclusion_reason(
            row,
            context_row,
            current_prompts,
            exclude_pdf_context=exclude_pdf_context,
            exclude_no_context=exclude_no_context,
            required_context_substring=required_context_substring,
        )
        if reason:
            reasons[reason] += 1
            excluded.append({
                "source_row_index": idx,
                "reason": reason,
                "prompt_sha256": sha256_bytes(str(row.get("prompt") or "").encode("utf-8")),
            })
            continue
        kept_plain.append(enrich(row, corpus=corpus, source_index=idx, context_baked=False, filters=filter_settings))
        kept_context.append(enrich(context_row, corpus=corpus, source_index=idx, context_baked=True, filters=filter_settings))

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
        "filters": filter_settings,
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


def build(
    output_dir: Path,
    *,
    source_testsets: dict[str, dict[str, Path]],
    current_train_val_paths: list[Path],
    source_manifest_path: Path | None,
    exclude_pdf_context: bool,
    exclude_no_context: bool,
    required_context_substring: str | None,
) -> dict[str, Any]:
    current_prompts, current_manifest = load_current_prompt_set(current_train_val_paths)
    corpus_manifests = {
        corpus: build_one(
            corpus,
            paths,
            output_dir,
            current_prompts,
            exclude_pdf_context=exclude_pdf_context,
            exclude_no_context=exclude_no_context,
            required_context_substring=required_context_substring,
        )
        for corpus, paths in source_testsets.items()
    }
    manifest = {
        "schema_version": "stage3-golden-manifest/v1",
        "version": "golden-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Immutable golden test sets for LE-vs-Curator and reference-model evaluation.",
        "source_manifest": str(source_manifest_path) if source_manifest_path else None,
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
    ap.add_argument("--source-manifest", type=Path, default=Path(DEFAULT_SOURCE_MANIFEST) if DEFAULT_SOURCE_MANIFEST else None)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--current-train-val", type=Path, action="append", default=[])
    ap.add_argument("--required-context-substring", default=None)
    ap.add_argument("--allow-pdf-context", action="store_true")
    ap.add_argument("--allow-no-context", action="store_true")
    args = ap.parse_args()

    if args.source_manifest is None:
        ap.error(
            "--source-manifest is required. Provide JSON with corpora.<name>.plain, "
            "corpora.<name>.with_context, and optional current_train_validation_paths."
        )

    source_testsets, current_paths, _manifest_payload = load_source_manifest(args.source_manifest)
    current_paths.extend(args.current_train_val)
    manifest = build(
        args.output_dir,
        source_testsets=source_testsets,
        current_train_val_paths=current_paths,
        source_manifest_path=args.source_manifest,
        exclude_pdf_context=not args.allow_pdf_context,
        exclude_no_context=not args.allow_no_context,
        required_context_substring=args.required_context_substring,
    )
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
