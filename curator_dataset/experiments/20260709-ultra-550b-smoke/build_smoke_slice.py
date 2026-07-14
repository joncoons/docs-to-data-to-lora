#!/usr/bin/env python3
"""Freeze a source-fixed smoke slice for Super-vs-Ultra dataset generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover - optional local convenience
    tiktoken = None


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "inputs"
DEFAULT_SEED = "20260709-ultra-550b-smoke-v1"
DEFAULT_DOCS_PER_CORPUS = 48

CORPORA = {
    "nim_curated": {
        "source": REPO_ROOT
        / "curator_dataset/data/nim_curated/url_documents.curator_ready.jsonl",
        "manifest": REPO_ROOT
        / "curator_dataset/data/nim_curated/url_documents.curator_ready.manifest.json",
        "product_family": "NIM",
        "product_name": "NVIDIA NIM",
    },
    "nemo_usvcs_curated": {
        "source": REPO_ROOT
        / "curator_dataset/data/nemo_usvcs_curated/url_documents.curator_ready.jsonl",
        "manifest": REPO_ROOT
        / "curator_dataset/data/nemo_usvcs_curated/url_documents.curator_ready.manifest.json",
        "product_family": "NeMo Microservices",
        "product_name": "NVIDIA NeMo Microservices",
    },
}

LENGTH_BUCKETS = (
    ("short", 30, 249),
    ("medium", 250, 749),
    ("long", 750, 1999),
    ("very_long", 2000, None),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as stream:
        for index, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line_number"] = index
            rows.append(row)
    return rows


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def word_count(row: dict[str, Any]) -> int:
    raw = row.get("word_count")
    if isinstance(raw, int):
        return raw
    return len(str(row.get("text") or "").split())


def text_hash(row: dict[str, Any]) -> str:
    return str(row.get("text_sha256") or sha256_text(str(row.get("text") or "")))


def token_count(text: str) -> int:
    if tiktoken is not None:
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    return len(text.split())


def length_bucket(row: dict[str, Any]) -> str:
    count = word_count(row)
    for name, minimum, maximum in LENGTH_BUCKETS:
        if count >= minimum and (maximum is None or count <= maximum):
            return name
    return "too_short"


def selection_key(row: dict[str, Any], *, collection: str, seed: str) -> str:
    material = "\n".join(
        [
            seed,
            collection,
            str(row.get("document_id") or row.get("id") or ""),
            str(row.get("url") or ""),
            str(row.get("text_sha256") or sha256_text(str(row.get("text") or ""))),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def select_rows(
    rows: list[dict[str, Any]],
    *,
    collection: str,
    docs_per_corpus: int,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible_with_duplicates = [row for row in rows if length_bucket(row) != "too_short"]
    deduped_by_text: dict[str, dict[str, Any]] = {}
    for row in sorted(
        eligible_with_duplicates,
        key=lambda candidate: selection_key(candidate, collection=collection, seed=seed),
    ):
        deduped_by_text.setdefault(text_hash(row), row)
    eligible = sorted(
        deduped_by_text.values(),
        key=lambda candidate: int(candidate["_source_line_number"]),
    )
    target_per_bucket = docs_per_corpus // len(LENGTH_BUCKETS)
    remainder = docs_per_corpus % len(LENGTH_BUCKETS)
    selected: dict[str, dict[str, Any]] = {}
    bucket_counts_before = Counter(length_bucket(row) for row in rows)
    bucket_targets: dict[str, int] = {}

    for bucket_index, (bucket, _, _) in enumerate(LENGTH_BUCKETS):
        target = target_per_bucket + (1 if bucket_index < remainder else 0)
        bucket_targets[bucket] = target
        bucket_rows = [row for row in eligible if length_bucket(row) == bucket]
        bucket_rows.sort(key=lambda row: selection_key(row, collection=collection, seed=seed))
        for row in bucket_rows[:target]:
            selected[str(row.get("document_id") or row.get("id"))] = row

    if len(selected) < docs_per_corpus:
        fill_rows = [
            row
            for row in eligible
            if str(row.get("document_id") or row.get("id")) not in selected
        ]
        fill_rows.sort(key=lambda row: selection_key(row, collection=collection, seed=seed))
        for row in fill_rows[: docs_per_corpus - len(selected)]:
            selected[str(row.get("document_id") or row.get("id"))] = row

    selected_rows = sorted(selected.values(), key=lambda row: int(row["_source_line_number"]))
    selected_counts = Counter(length_bucket(row) for row in selected_rows)
    return selected_rows, {
        "source_rows": len(rows),
        "eligible_rows_before_text_deduplication": len(eligible_with_duplicates),
        "eligible_rows": len(eligible),
        "exact_duplicate_text_rows_excluded": len(eligible_with_duplicates) - len(eligible),
        "selected_rows": len(selected_rows),
        "bucket_counts_before_selection": dict(sorted(bucket_counts_before.items())),
        "bucket_targets": bucket_targets,
        "selected_bucket_counts": dict(sorted(selected_counts.items())),
    }


def curator_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    cleaned.pop("_source_line_number", None)
    return cleaned


def passage_row(row: dict[str, Any], *, fallback: dict[str, str]) -> dict[str, Any]:
    text = str(row.get("text") or "")
    return {
        "passage_id": str(row.get("document_id") or row.get("id")),
        "url": str(row.get("url") or ""),
        "text": text,
        "token_count": token_count(text),
        "chunk_ids": list(row.get("es_chunk_ids") or row.get("source_chunk_ids") or []),
        "product_family": str(row.get("product_family") or fallback["product_family"]),
        "product_name": str(row.get("product_name") or fallback["product_name"]),
        "doc_kind": str(row.get("doc_kind") or "html"),
        "source_revision_id": None,
        "source_chunk_ids": list(row.get("source_chunk_ids") or []),
        "source_systems": list(row.get("source_systems") or []),
        "source_kinds": list(row.get("source_kinds") or []),
        "modalities": list(row.get("modalities") or []),
    }


def artifact_manifest(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "rows": sum(1 for line in path.read_text().splitlines() if line.strip()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_slice(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    manifest: dict[str, Any] = {
        "schema_version": "ultra_550b_smoke.slice_manifest.v1",
        "created_at": utc_now(),
        "experiment": "20260709-ultra-550b-smoke",
        "seed": args.seed,
        "docs_per_corpus": args.docs_per_corpus,
        "selection_policy": {
            "minimum_words": 30,
            "length_buckets": [
                {"name": name, "minimum_words": minimum, "maximum_words": maximum}
                for name, minimum, maximum in LENGTH_BUCKETS
            ],
            "bucket_sampling": "deterministic SHA-256 sort by seed, collection, document_id, url, and text hash",
            "output_order": "original source JSONL line order",
        },
        "corpora": {},
        "selected_documents": [],
    }

    for collection, config in CORPORA.items():
        source_path = config["source"].resolve()
        rows = read_jsonl(source_path)
        selected, metrics = select_rows(
            rows,
            collection=collection,
            docs_per_corpus=args.docs_per_corpus,
            seed=args.seed,
        )

        curator_output = output_root / f"{collection}.curator_input.jsonl"
        passage_output = output_root / f"{collection}.passages.jsonl"
        write_jsonl_atomic(curator_output, [curator_row(row) for row in selected])
        write_jsonl_atomic(
            passage_output,
            [passage_row(row, fallback=config) for row in selected],
        )

        manifest["corpora"][collection] = {
            "source": {
                "path": relative(source_path),
                "rows": len(rows),
                "bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
                "manifest_path": relative(config["manifest"]),
            },
            "selection": metrics,
            "outputs": {
                "curator_input": artifact_manifest(curator_output),
                "le_passages": artifact_manifest(passage_output),
            },
        }
        manifest["selected_documents"].extend(
            {
                "collection": collection,
                "source_line_number": int(row["_source_line_number"]),
                "document_id": str(row.get("document_id") or row.get("id")),
                "url": str(row.get("url") or ""),
                "word_count": word_count(row),
                "length_bucket": length_bucket(row),
                "text_sha256": text_hash(row),
            }
            for row in selected
        )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-per-corpus", type=int, default=DEFAULT_DOCS_PER_CORPUS)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT_ROOT / "slice_manifest.json",
    )
    args = parser.parse_args()
    if args.docs_per_corpus < len(LENGTH_BUCKETS):
        parser.error(f"--docs-per-corpus must be at least {len(LENGTH_BUCKETS)}")
    return args


def main() -> int:
    args = parse_args()
    manifest = build_slice(args)
    write_json_atomic(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
