#!/usr/bin/env python3
"""Prepare and collect native NeMo Curator curation work.

Stage 3's showcase path should use NeMo Curator for generic quality filtering
and deduplication. This helper keeps repository-specific duties small:

- prepare `provenance/dataset_samples.jsonl` as Curator-readable JSONL;
- copy/hash the Curator config used by the native job;
- collect retained/removed Curator records back into repository artifacts;
- write Customizer-compatible `training.jsonl` and `validation.jsonl`;
- emit MLflow-ready observability files.

The actual curation job runs in the NeMo Curator container or cluster. Local
Python execution remains useful for handoff validation and result collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.provenance import SCHEMA_VERSION, utc_now  # noqa: E402

DEFAULT_DATASET_DIR = Path(os.getenv("DATASET_DIR", "/datasets/nim_curated"))
DEFAULT_OBSERVABILITY_DIR = Path(
    os.getenv("OBSERVABILITY_DIR", "/observability/curator/nim_curated")
)
DEFAULT_CONFIG_FILE = _REPO_ROOT / "configs" / "curator" / "sft-dedup-quality.yaml"
DEFAULT_TRAIN_RATIO = float(os.getenv("CURATOR_TRAIN_RATIO", "0.9"))
DEFAULT_SEED = int(os.getenv("CURATOR_SPLIT_SEED", "42"))


@dataclass(frozen=True)
class CuratorHandoffConfig:
    dataset_dir: Path
    collection: str
    mode: str
    observability_dir: Path
    config_file: Path
    train_ratio: float
    split_seed: int
    curator_job_id: str | None = None
    accepted_jsonl: Path | None = None
    accepted_dir: Path | None = None
    rejected_jsonl: Path | None = None
    rejected_dir: Path | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def artifact_manifest(path: Path, dataset_dir: Path, artifact_kind: str) -> dict[str, Any]:
    try:
        artifact_path = str(path.relative_to(dataset_dir))
    except ValueError:
        artifact_path = path.name
    manifest: dict[str, Any] = {
        "path": str(path),
        "artifact_path": artifact_path,
        "artifact_kind": artifact_kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        manifest["rows"] = count_jsonl_rows(path)
    return manifest


def load_dataset_samples(dataset_dir: Path) -> list[dict[str, Any]]:
    samples_path = dataset_dir / "provenance" / "dataset_samples.jsonl"
    samples = read_jsonl(samples_path)
    if not samples:
        raise FileNotFoundError(f"no dataset samples found at {samples_path}")
    return samples


def sample_text(sample: dict[str, Any]) -> str:
    prompt = str(sample.get("prompt") or "").strip()
    completion = str(sample.get("completion") or "").strip()
    return f"Question: {prompt}\nAnswer: {completion}".strip()


def normalize_sample_for_curator(sample: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id") or sample.get("id") or "")
    if not sample_id:
        raise ValueError("dataset sample is missing sample_id")
    lineage = sample.get("lineage") if isinstance(sample.get("lineage"), dict) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    quality = sample.get("quality") if isinstance(sample.get("quality"), dict) else {}
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "text": sample_text(sample),
        "prompt": sample.get("prompt"),
        "completion": sample.get("completion"),
        "system": sample.get("system"),
        "origin": sample.get("origin"),
        "task_type": sample.get("task_type"),
        "source_systems": lineage.get("source_systems") or metadata.get("source_systems") or [],
        "source_kinds": lineage.get("source_kinds") or metadata.get("source_kinds") or [],
        "modalities": lineage.get("modalities") or metadata.get("modalities") or [],
        "lineage": lineage,
        "quality": quality,
        "metadata": metadata,
    }


def prepare_curator_input(config: CuratorHandoffConfig) -> dict[str, Any]:
    samples = load_dataset_samples(config.dataset_dir)
    curator_dir = config.dataset_dir / "curator"
    input_dir = curator_dir / "input"
    input_path = input_dir / "dataset_samples.jsonl"
    config_copy = curator_dir / "curator_config.yaml"
    input_records = [normalize_sample_for_curator(sample) for sample in samples]

    input_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(input_path, input_records)
    if not config.config_file.exists():
        raise FileNotFoundError(f"Curator config file not found: {config.config_file}")
    config_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.config_file, config_copy)
    config_hash = sha256_file(config_copy)

    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "collection": config.collection,
        "native_service": "NeMo Curator",
        "status": "prepared",
        "input": {
            "dataset_samples_uri": "provenance/dataset_samples.jsonl",
            "curator_input_uri": "curator/input/dataset_samples.jsonl",
            "document_count": len(input_records),
            "text_field": "text",
            "id_field": "id",
        },
        "config": {
            "source_path": str(config.config_file),
            "dataset_uri": "curator/curator_config.yaml",
            "sha256": config_hash,
        },
        "outputs": {
            "retained_dir": "curator/retained",
            "removed_dir": "curator/removed",
            "scores_dir": "curator/scores",
        },
        "native_execution": {
            "container": "nvcr.io/nvidia/nemo-curator:latest",
            "notes": (
                "Run NeMo Curator in the official container or Dask/Ray cluster "
                "against curator/input/dataset_samples.jsonl, then run collect."
            ),
        },
    }
    write_json(curator_dir / "submission_plan.json", plan)
    return {
        "input_records": len(input_records),
        "config_hash": config_hash,
        "plan": plan,
    }


def _jsonl_files_in_dir(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix in {".jsonl", ".json"}
    )


def load_records(path: Path | None, directory: Path | None) -> list[dict[str, Any]]:
    if path:
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict) and isinstance(data.get("rows"), list):
                return [item for item in data["rows"] if isinstance(item, dict)]
            return []
        return read_jsonl(path)
    if not directory:
        return []
    rows: list[dict[str, Any]] = []
    for item in _jsonl_files_in_dir(directory):
        rows.extend(load_records(item, None))
    return rows


def sample_id_from_record(record: dict[str, Any]) -> str | None:
    value = record.get("sample_id") or record.get("id")
    return str(value) if value else None


def _curation_metadata(record: dict[str, Any], status: str) -> dict[str, Any]:
    metadata = record.get("curator") if isinstance(record.get("curator"), dict) else {}
    reason = (
        record.get("rejection_reason")
        or record.get("removed_by")
        or record.get("filter_name")
        or metadata.get("rejection_reason")
    )
    return {
        "curation_status": status,
        "rejection_reason": str(reason) if reason else None,
        "scores": record.get("scores") if isinstance(record.get("scores"), dict) else {},
    }


def apply_curation_status(
    records: list[dict[str, Any]],
    samples_by_id: dict[str, dict[str, Any]],
    *,
    status: str,
    curator_job_id: str | None,
    curator_config_hash: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        sample_id = sample_id_from_record(record)
        if not sample_id or sample_id not in samples_by_id:
            continue
        sample = json.loads(json.dumps(samples_by_id[sample_id]))
        quality = sample.get("quality") if isinstance(sample.get("quality"), dict) else {}
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        quality.update({
            "curator_job_id": curator_job_id,
            "curator_config_hash": curator_config_hash,
        })
        metadata["curation"] = _curation_metadata(record, status)
        sample["quality"] = quality
        sample["metadata"] = metadata
        samples.append(sample)
    return samples


def train_val_split_samples(
    samples: list[dict[str, Any]],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = f"{sample.get('origin') or 'unknown'}::{sample.get('task_type') or 'unknown'}"
        buckets[key].append(sample)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for rows in buckets.values():
        rng.shuffle(rows)
        n_train = int(len(rows) * train_ratio)
        if len(rows) > 1 and n_train == len(rows):
            n_train = len(rows) - 1
        train.extend(rows[:n_train])
        validation.extend(rows[n_train:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def to_customizer_format(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": sample.get("prompt"),
            "completion": sample.get("completion"),
            "system": sample.get("system"),
        }
        for sample in samples
    ]


def build_rejection_report(rejected_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in rejected_samples:
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        curation = metadata.get("curation") if isinstance(metadata.get("curation"), dict) else {}
        rows.append({
            "sample_id": sample.get("sample_id"),
            "origin": sample.get("origin"),
            "task_type": sample.get("task_type"),
            "rejection_reason": curation.get("rejection_reason"),
            "scores": curation.get("scores") or {},
        })
    return rows


def collect_curator_results(config: CuratorHandoffConfig, config_hash: str | None = None) -> dict[str, Any]:
    samples = load_dataset_samples(config.dataset_dir)
    samples_by_id = {
        str(sample["sample_id"]): sample
        for sample in samples
        if sample.get("sample_id")
    }
    curator_dir = config.dataset_dir / "curator"
    config_copy = curator_dir / "curator_config.yaml"
    config_hash = config_hash or (sha256_file(config_copy) if config_copy.exists() else "")
    if not config_hash:
        raise FileNotFoundError("curator/curator_config.yaml is missing; run prepare first")

    accepted_records = load_records(
        config.accepted_jsonl,
        config.accepted_dir or curator_dir / "retained",
    )
    if not accepted_records:
        raise FileNotFoundError(
            "no Curator retained records found; provide --accepted-jsonl or --accepted-dir"
        )
    rejected_records = load_records(
        config.rejected_jsonl,
        config.rejected_dir or curator_dir / "removed",
    )

    accepted_samples = apply_curation_status(
        accepted_records,
        samples_by_id,
        status="accepted",
        curator_job_id=config.curator_job_id,
        curator_config_hash=config_hash,
    )
    rejected_samples = apply_curation_status(
        rejected_records,
        samples_by_id,
        status="rejected",
        curator_job_id=config.curator_job_id,
        curator_config_hash=config_hash,
    )
    accepted_ids = {sample.get("sample_id") for sample in accepted_samples}
    rejected_samples = [
        sample for sample in rejected_samples if sample.get("sample_id") not in accepted_ids
    ]

    train_samples, validation_samples = train_val_split_samples(
        accepted_samples,
        train_ratio=config.train_ratio,
        seed=config.split_seed,
    )
    write_jsonl(curator_dir / "accepted_samples.jsonl", accepted_samples)
    write_jsonl(curator_dir / "rejected_samples.jsonl", rejected_samples)
    write_jsonl(curator_dir / "rejection_report.jsonl", build_rejection_report(rejected_samples))
    write_jsonl(config.dataset_dir / "training.jsonl", to_customizer_format(train_samples))
    write_jsonl(config.dataset_dir / "validation.jsonl", to_customizer_format(validation_samples))

    reason_counts = Counter(
        (sample.get("metadata") or {}).get("curation", {}).get("rejection_reason") or "unknown"
        for sample in rejected_samples
    )
    metrics = {
        "input_samples": len(samples),
        "accepted_samples": len(accepted_samples),
        "rejected_samples": len(rejected_samples),
        "acceptance_rate": round(len(accepted_samples) / len(samples), 6) if samples else 0,
        "training_rows": len(train_samples),
        "validation_rows": len(validation_samples),
        "rejection_reasons": dict(sorted(reason_counts.items())),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "collection": config.collection,
        "native_service": "NeMo Curator",
        "curator_job_id": config.curator_job_id,
        "curator_config_hash": config_hash,
        "status": "collected",
        "metrics": metrics,
        "outputs": {
            "accepted_samples_uri": "curator/accepted_samples.jsonl",
            "rejected_samples_uri": "curator/rejected_samples.jsonl",
            "rejection_report_uri": "curator/rejection_report.jsonl",
            "training_uri": "training.jsonl",
            "validation_uri": "validation.jsonl",
        },
        "split": {
            "method": "curator-retained-stratified-random",
            "train_ratio": config.train_ratio,
            "seed": config.split_seed,
        },
    }
    write_json(curator_dir / "curation_manifest.json", manifest)
    return {
        "config_hash": config_hash,
        "metrics": metrics,
        "manifest": manifest,
    }


def collect_artifact_paths(dataset_dir: Path, mode: str) -> list[tuple[Path, str]]:
    candidates = [
        (dataset_dir / "curator" / "input" / "dataset_samples.jsonl", "curator_input"),
        (dataset_dir / "curator" / "curator_config.yaml", "curator_config"),
        (dataset_dir / "curator" / "submission_plan.json", "curator_submission_plan"),
    ]
    if mode in {"collect", "prepare-and-collect"}:
        candidates.extend([
            (dataset_dir / "curator" / "curation_manifest.json", "curator_manifest"),
            (dataset_dir / "curator" / "accepted_samples.jsonl", "curator_accepted_samples"),
            (dataset_dir / "curator" / "rejected_samples.jsonl", "curator_rejected_samples"),
            (dataset_dir / "curator" / "rejection_report.jsonl", "curator_rejection_report"),
            (dataset_dir / "training.jsonl", "training_split"),
            (dataset_dir / "validation.jsonl", "validation_split"),
        ])
    return [(path, kind) for path, kind in candidates if path.exists()]


def build_observability_documents(
    config: CuratorHandoffConfig,
    *,
    config_hash: str | None,
    input_count: int,
    metrics: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    metrics = metrics or {}
    obs_metrics: dict[str, int | float] = {
        "curator.samples.input": input_count,
        "curator.samples.accepted": int(metrics.get("accepted_samples", 0)),
        "curator.samples.rejected": int(metrics.get("rejected_samples", 0)),
        "curator.acceptance_rate": float(metrics.get("acceptance_rate", 0)),
        "curator.rows.training": int(metrics.get("training_rows", 0)),
        "curator.rows.validation": int(metrics.get("validation_rows", 0)),
    }
    for reason, count in metrics.get("rejection_reasons", {}).items():
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(reason)).strip("_")
        obs_metrics[f"curator.rejection_reason.{safe or 'unknown'}.count"] = count

    artifacts = [
        artifact_manifest(path, config.dataset_dir, kind)
        for path, kind in collect_artifact_paths(config.dataset_dir, config.mode)
    ]
    run_context = {
        "schema_version": "observability.v1",
        "pipeline_stage": "curator",
        "created_at": utc_now(),
        "collection": config.collection,
        "dataset_dir": str(config.dataset_dir),
        "mode": config.mode,
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
        "mlflow": {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
            "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME"),
            "parent_run_id": os.getenv("MLFLOW_PARENT_RUN_ID"),
        },
    }
    service_refs = {
        "schema_version": "observability.v1",
        "curator": {
            "native_service": "NeMo Curator",
            "job_id": config.curator_job_id,
            "config_hash": config_hash,
            "config_uri": "curator/curator_config.yaml",
            "input_uri": "curator/input/dataset_samples.jsonl",
            "manifest_uri": "curator/curation_manifest.json",
        },
    }
    return {
        "run_context.json": run_context,
        "metrics.json": obs_metrics,
        "artifacts_manifest.json": {"schema_version": "observability.v1", "artifacts": artifacts},
        "service_refs.json": service_refs,
    }


def write_observability_documents(out_dir: Path, documents: dict[str, dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in documents.items():
        write_json(out_dir / name, payload)


def run(config: CuratorHandoffConfig) -> dict[str, Any]:
    input_count = 0
    config_hash: str | None = None
    metrics: dict[str, Any] | None = None
    if config.mode in {"prepare", "prepare-and-collect"}:
        prepared = prepare_curator_input(config)
        input_count = int(prepared["input_records"])
        config_hash = str(prepared["config_hash"])
    else:
        input_count = count_jsonl_rows(config.dataset_dir / "curator" / "input" / "dataset_samples.jsonl")
        config_copy = config.dataset_dir / "curator" / "curator_config.yaml"
        config_hash = sha256_file(config_copy) if config_copy.exists() else None

    if config.mode in {"collect", "prepare-and-collect"}:
        collected = collect_curator_results(config, config_hash=config_hash)
        config_hash = str(collected["config_hash"])
        metrics = collected["metrics"]
        input_count = int(metrics["input_samples"])

    docs = build_observability_documents(
        config,
        config_hash=config_hash,
        input_count=input_count,
        metrics=metrics,
    )
    write_observability_documents(config.observability_dir, docs)
    return {
        "mode": config.mode,
        "collection": config.collection,
        "config_hash": config_hash,
        "input_samples": input_count,
        "metrics": metrics or {},
        "observability_dir": str(config.observability_dir),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--collection", default=os.getenv("COLLECTION"))
    ap.add_argument(
        "--mode",
        choices=["prepare", "collect", "prepare-and-collect"],
        default=os.getenv("CURATOR_MODE", "prepare"),
    )
    ap.add_argument("--observability-dir", type=Path, default=DEFAULT_OBSERVABILITY_DIR)
    ap.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_FILE)
    ap.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    ap.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--curator-job-id", default=os.getenv("CURATOR_JOB_ID"))
    ap.add_argument("--accepted-jsonl", type=Path, default=None)
    ap.add_argument("--accepted-dir", type=Path, default=None)
    ap.add_argument("--rejected-jsonl", type=Path, default=None)
    ap.add_argument("--rejected-dir", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    collection = args.collection or args.dataset_dir.name
    config = CuratorHandoffConfig(
        dataset_dir=args.dataset_dir,
        collection=collection,
        mode=args.mode,
        observability_dir=args.observability_dir,
        config_file=args.config_file,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        curator_job_id=args.curator_job_id,
        accepted_jsonl=args.accepted_jsonl,
        accepted_dir=args.accepted_dir,
        rejected_jsonl=args.rejected_jsonl,
        rejected_dir=args.rejected_dir,
    )
    print(json.dumps(run(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
