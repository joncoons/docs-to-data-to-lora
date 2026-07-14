#!/usr/bin/env python3
"""Build an augmented training dataset from accepted Data Designer samples.

The grounded dataset remains the source of truth. This script appends accepted
synthetic rows to the train/adapter-train splits only, copies validation and
test artifacts unchanged, and writes manifests that connect the augmented
training data back to the Data Designer job and source dataset checksums.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.provenance import SCHEMA_VERSION, utc_now  # noqa: E402

DEFAULT_SOURCE_DATASET_DIR = Path(os.getenv("SOURCE_DATASET_DIR", "<DATASET_ROOT>/nim_curated"))
DEFAULT_EXPERIMENT_DIR = Path(
    os.getenv(
        "DATA_DESIGNER_AUGMENTATION_DIR",
        "<DATASET_ROOT>/experiments/nim_curated_dd_llm_1b",
    )
)
DEFAULT_SYSTEM_PROMPT = "You are a precise NVIDIA NIM technical assistant. Answer based on official documentation."
COPY_FILES = (
    "validation.jsonl",
    "test_set.jsonl",
    "test_set_with_context.jsonl",
    "adapter_val.jsonl",
    "validation_report.json",
    "bias_report.json",
    "test_kvp_uids.json",
)


@dataclass(frozen=True)
class AugmentedDatasetConfig:
    source_dataset_dir: Path
    experiment_dir: Path
    output_dir: Path
    accepted_samples_path: Path
    collection: str
    dataset_name: str
    system_prompt: str
    synthetic_limit: int | None = None
    overwrite: bool = False


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact(path: Path, *, base_dir: Path, kind: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "relative_path": str(path.relative_to(base_dir)),
        "artifact_kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        item["rows"] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return item


def compact_train_row(row: dict[str, Any], *, system_prompt: str) -> dict[str, str]:
    prompt = str(row.get("prompt") or row.get("question") or "").strip()
    completion = str(row.get("completion") or row.get("answer") or "").strip()
    system = str(row.get("system") or system_prompt).strip()
    if not prompt or not completion:
        raise ValueError("accepted synthetic row is missing prompt/completion")
    return {"prompt": prompt, "completion": completion, "system": system}


def load_synthetic_rows(config: AugmentedDatasetConfig) -> list[dict[str, str]]:
    records = read_jsonl(config.accepted_samples_path)
    if config.synthetic_limit is not None:
        records = records[: config.synthetic_limit]
    return [compact_train_row(row, system_prompt=config.system_prompt) for row in records]


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output without --overwrite: {path}")


def write_augmented_split(source_path: Path, output_path: Path, synthetic_rows: list[dict[str, str]], overwrite: bool) -> dict[str, Any]:
    ensure_can_write(output_path, overwrite)
    source_rows = read_jsonl(source_path)
    write_jsonl(output_path, source_rows + synthetic_rows)
    return {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "source_rows": len(source_rows),
        "synthetic_rows_appended": len(synthetic_rows),
        "output_rows": len(source_rows) + len(synthetic_rows),
        "source_sha256": sha256_file(source_path),
        "output_sha256": sha256_file(output_path),
    }


def copy_unchanged_file(source_path: Path, output_path: Path, overwrite: bool) -> dict[str, Any] | None:
    if not source_path.exists():
        return None
    ensure_can_write(output_path, overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, output_path)
    return {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "source_sha256": sha256_file(source_path),
        "output_sha256": sha256_file(output_path),
        "unchanged": sha256_file(source_path) == sha256_file(output_path),
    }


def build_augmented_dataset(config: AugmentedDatasetConfig) -> dict[str, Any]:
    synthetic_rows = load_synthetic_rows(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    split_manifests = {
        "training": write_augmented_split(
            config.source_dataset_dir / "training.jsonl",
            config.output_dir / "training.jsonl",
            synthetic_rows,
            config.overwrite,
        ),
        "adapter_train": write_augmented_split(
            config.source_dataset_dir / "adapter_train.jsonl",
            config.output_dir / "adapter_train.jsonl",
            synthetic_rows,
            config.overwrite,
        ),
    }
    copied: dict[str, Any] = {}
    for filename in COPY_FILES:
        copied_item = copy_unchanged_file(
            config.source_dataset_dir / filename,
            config.output_dir / filename,
            config.overwrite,
        )
        if copied_item:
            copied[filename] = copied_item

    synthetic_train_rows = config.output_dir / "data_designer" / "synthetic_train_rows.jsonl"
    ensure_can_write(synthetic_train_rows, config.overwrite)
    write_jsonl(synthetic_train_rows, synthetic_rows)

    result_manifest = read_json(config.experiment_dir / "data_designer" / "result_manifest.json")
    source_manifest = read_json(config.experiment_dir / "source_snapshot" / "manifest.json")
    dataset_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset_name": config.dataset_name,
        "collection": config.collection,
        "source_dataset_dir": str(config.source_dataset_dir),
        "output_dir": str(config.output_dir),
        "policy": {
            "grounded_dataset_is_source_of_truth": True,
            "synthetic_rows_appended_to": ["training.jsonl", "adapter_train.jsonl"],
            "validation_and_test_unchanged": True,
            "source_support_check": result_manifest.get("source_support_check", "not_run"),
            "curator_job_id": result_manifest.get("curator_job_id"),
        },
        "data_designer": {
            "job_id": result_manifest.get("data_designer_job_id"),
            "model_provider": result_manifest.get("model_provider"),
            "model": result_manifest.get("model"),
            "accepted_synthetic_pair_count": result_manifest.get("accepted_synthetic_pair_count", len(synthetic_rows)),
            "synthetic_rows_used": len(synthetic_rows),
            "side_effect_columns_excluded": result_manifest.get("side_effect_columns_excluded", []),
            "token_metrics": result_manifest.get("token_metrics", {}),
        },
        "splits": split_manifests,
        "copied_unchanged_files": copied,
        "artifacts": [],
        "source_snapshot": source_manifest,
    }
    artifact_paths = [
        (config.output_dir / "training.jsonl", "augmented_training_split"),
        (config.output_dir / "adapter_train.jsonl", "augmented_adapter_train_split"),
        (synthetic_train_rows, "synthetic_train_rows"),
    ]
    artifact_paths.extend((config.output_dir / filename, "unchanged_source_artifact") for filename in copied)
    dataset_manifest["artifacts"] = [
        artifact(path, base_dir=config.output_dir, kind=kind)
        for path, kind in artifact_paths
        if path.exists()
    ]

    merge_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": dataset_manifest["created_at"],
        "dataset_name": config.dataset_name,
        "collection": config.collection,
        "source_dataset_dir": str(config.source_dataset_dir),
        "accepted_samples_path": str(config.accepted_samples_path),
        "data_designer_job_id": result_manifest.get("data_designer_job_id"),
        "synthetic_rows_appended": len(synthetic_rows),
        "training_output_rows": split_manifests["training"]["output_rows"],
        "adapter_train_output_rows": split_manifests["adapter_train"]["output_rows"],
        "validation_and_test_unchanged": True,
    }
    write_json(config.output_dir / "manifests" / "dataset_version_manifest.json", dataset_manifest)
    write_json(config.output_dir / "provenance" / "merge_manifest.json", merge_manifest)
    return dataset_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dataset-dir", type=Path, default=DEFAULT_SOURCE_DATASET_DIR)
    ap.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--accepted-samples-path", type=Path, default=None)
    ap.add_argument("--collection", default="nim_curated")
    ap.add_argument("--dataset-name", default="nim_curated_dd_llm_1b")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--synthetic-limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_dir = args.experiment_dir
    output_dir = args.output_dir or experiment_dir
    accepted_samples = args.accepted_samples_path or experiment_dir / "data_designer" / "accepted_samples.jsonl"
    config = AugmentedDatasetConfig(
        source_dataset_dir=args.source_dataset_dir,
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        accepted_samples_path=accepted_samples,
        collection=args.collection,
        dataset_name=args.dataset_name,
        system_prompt=args.system_prompt,
        synthetic_limit=args.synthetic_limit,
        overwrite=args.overwrite,
    )
    result = build_augmented_dataset(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
