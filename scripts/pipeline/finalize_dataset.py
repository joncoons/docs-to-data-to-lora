"""Finalize a dataset build with a deterministic version manifest.

This step closes the local lineage loop before NeMo Data Store / Entity Store
registration. It reads final split files plus provenance sidecars, materializes
`manifests/dataset_version_manifest.json`, and writes MLflow-ready observability
JSON without requiring an MLflow client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.models import KVPRow  # noqa: E402
from scripts.pipeline.provenance import (  # noqa: E402
    dataset_samples_from_kvp_rows,
    stable_id,
    utc_now,
)
from scripts.pipeline.provenance_io import write_json, write_jsonl  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = Path(os.getenv("DATASET_DIR", "/datasets/nim_curated"))
DEFAULT_OBSERVABILITY_DIR = Path(
    os.getenv("OBSERVABILITY_DIR", "/observability/dataset-finalization/nim_curated")
)
DEFAULT_DATASET_ROLE = os.getenv("DATASET_ROLE")

SPLIT_FILES = {
    "training": "training.jsonl",
    "validation": "validation.jsonl",
    "test": "test_set.jsonl",
}

ARTIFACT_CANDIDATES = (
    "stage2_eval.jsonl",
    "training.jsonl",
    "validation.jsonl",
    "test_set.jsonl",
    "test_set_with_context.jsonl",
    "adapter_train.jsonl",
    "adapter_val.jsonl",
    "test_kvp_uids.json",
    "manifests/crawl_run.json",
    "provenance/source_revisions.jsonl",
    "provenance/source_chunks.jsonl",
    "provenance/entailments.jsonl",
    "provenance/dataset_samples.jsonl",
    "provenance/delta_manifest.json",
    "provenance/gap_manifest.json",
    "bias_report.json",
    "validation_report.json",
)


def safe_metric_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in cleaned.split("_") if part) or "unknown"


def relative_artifact_path(path: Path, dataset_dir: Path) -> str:
    try:
        return str(path.relative_to(dataset_dir))
    except ValueError:
        return path.name


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def file_manifest(path: Path, dataset_dir: Path, artifact_kind: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "path": str(path),
        "artifact_path": relative_artifact_path(path, dataset_dir),
        "artifact_kind": artifact_kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        manifest["rows"] = count_jsonl_rows(path)
    return manifest


def list_existing_artifacts(dataset_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for rel_path in ARTIFACT_CANDIDATES:
        path = dataset_dir / rel_path
        if path.exists():
            artifacts.append(file_manifest(path, dataset_dir, artifact_kind="dataset_input"))
    return artifacts


def _list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _lineage_values(sample: dict[str, Any], key: str) -> list[str]:
    lineage = sample.get("lineage") if isinstance(sample.get("lineage"), dict) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    return _list_values(lineage.get(key)) or _list_values(metadata.get(key))


def ensure_dataset_samples(
    dataset_dir: Path,
    *,
    system_prompt: str | None = None,
    generate_missing: bool = True,
) -> bool:
    samples_path = dataset_dir / "provenance" / "dataset_samples.jsonl"
    if samples_path.exists() or not generate_missing:
        return False

    stage2_path = dataset_dir / "stage2_eval.jsonl"
    if not stage2_path.exists():
        raise FileNotFoundError(
            "provenance/dataset_samples.jsonl is missing and stage2_eval.jsonl "
            "is unavailable for backfill"
        )

    rows = [
        KVPRow.model_validate_json(line)
        for line in stage2_path.read_text().splitlines()
        if line.strip()
    ]
    write_jsonl(samples_path, dataset_samples_from_kvp_rows(rows, system_prompt=system_prompt))
    log.info("Dataset finalization: backfilled %d dataset samples", len(rows))
    return True


def build_source_composition(samples: list[dict[str, Any]]) -> dict[str, Any]:
    source_urls: set[str] = set()
    source_revision_ids: set[str] = set()
    source_chunk_ids: set[str] = set()
    entailment_ids: set[str] = set()
    gap_ids: set[str] = set()
    data_designer_job_ids: set[str] = set()
    origin_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    source_system_counts: Counter[str] = Counter()
    source_kind_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()

    for sample in samples:
        origin_counts[str(sample.get("origin") or "unknown")] += 1
        task_type_counts[str(sample.get("task_type") or "unknown")] += 1
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        source_url = metadata.get("source_url")
        if source_url:
            source_urls.add(str(source_url))
        source_revision_ids.update(_lineage_values(sample, "source_revision_ids"))
        source_chunk_ids.update(_lineage_values(sample, "source_chunk_ids"))
        entailment_ids.update(_lineage_values(sample, "entailment_ids"))
        gap_ids.update(_lineage_values(sample, "gap_id"))
        data_designer_job_ids.update(_lineage_values(sample, "data_designer_job_id"))
        source_system_counts.update(_lineage_values(sample, "source_systems") or ["unknown"])
        source_kind_counts.update(_lineage_values(sample, "source_kinds") or ["unknown"])
        modality_counts.update(_lineage_values(sample, "modalities") or ["unknown"])

    total = len(samples)
    synthetic_count = origin_counts.get("synthetic_gapfill", 0)
    grounded_count = origin_counts.get("source_entailed", 0)
    return {
        "sample_count": total,
        "source_url_count": len(source_urls),
        "source_revision_count": len(source_revision_ids),
        "source_chunk_count": len(source_chunk_ids),
        "entailment_count": len(entailment_ids),
        "gap_count": len(gap_ids),
        "data_designer_job_count": len(data_designer_job_ids),
        "synthetic_sample_count": synthetic_count,
        "grounded_sample_count": grounded_count,
        "synthetic_ratio": round(synthetic_count / total, 6) if total else 0,
        "origins": dict(sorted(origin_counts.items())),
        "task_types": dict(sorted(task_type_counts.items())),
        "source_systems": dict(sorted(source_system_counts.items())),
        "source_kinds": dict(sorted(source_kind_counts.items())),
        "modalities": dict(sorted(modality_counts.items())),
        "source_revision_ids": sorted(source_revision_ids),
        "source_chunk_ids": sorted(source_chunk_ids),
        "entailment_ids": sorted(entailment_ids),
        "gap_ids": sorted(gap_ids),
        "data_designer_job_ids": sorted(data_designer_job_ids),
    }


def infer_dataset_role(explicit_role: str | None, composition: dict[str, Any]) -> str:
    if explicit_role:
        return explicit_role
    synthetic = composition.get("synthetic_sample_count", 0)
    grounded = composition.get("grounded_sample_count", 0)
    if synthetic and grounded:
        return "blended"
    if synthetic:
        return "synthetic_gapfill"
    return "source_entailed"


def split_metadata(dataset_dir: Path) -> dict[str, Any]:
    test_manifest = read_json_if_exists(dataset_dir / "test_kvp_uids.json")
    return {
        "method": "stage3-holdout" if test_manifest else "file-based",
        "seed": test_manifest.get("seed"),
        "train_count": count_jsonl_rows(dataset_dir / SPLIT_FILES["training"]),
        "validation_count": count_jsonl_rows(dataset_dir / SPLIT_FILES["validation"]),
        "test_count": count_jsonl_rows(dataset_dir / SPLIT_FILES["test"]),
    }


def input_ids(dataset_dir: Path, composition: dict[str, Any]) -> dict[str, list[str]]:
    crawl_run = read_json_if_exists(dataset_dir / "manifests" / "crawl_run.json")
    delta_manifest = read_json_if_exists(dataset_dir / "provenance" / "delta_manifest.json")
    gap_manifest = read_json_if_exists(dataset_dir / "provenance" / "gap_manifest.json")
    crawl_run_ids = [crawl_run["crawl_run_id"]] if crawl_run.get("crawl_run_id") else []
    delta_ids = _list_values(delta_manifest.get("delta_manifest_id"))
    gap_ids = _list_values(gap_manifest.get("gap_manifest_id")) or composition.get("gap_ids", [])
    return {
        "crawl_run_ids": crawl_run_ids,
        "delta_manifest_ids": delta_ids,
        "gap_manifest_ids": gap_ids,
        "data_designer_job_ids": composition.get("data_designer_job_ids", []),
        "parent_dataset_version_ids": [],
    }


def build_outputs(dataset_dir: Path, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    artifact_hashes = {
        artifact["artifact_path"]: artifact["sha256"]
        for artifact in artifacts
        if artifact.get("sha256")
    }
    return {
        "samples_uri": "provenance/dataset_samples.jsonl",
        "training_uri": (
            SPLIT_FILES["training"]
            if (dataset_dir / SPLIT_FILES["training"]).exists()
            else None
        ),
        "validation_uri": (
            SPLIT_FILES["validation"]
            if (dataset_dir / SPLIT_FILES["validation"]).exists()
            else None
        ),
        "test_uri": (
            SPLIT_FILES["test"] if (dataset_dir / SPLIT_FILES["test"]).exists() else None
        ),
        "hashes": artifact_hashes,
    }


def build_manifest_metrics(
    *,
    split: dict[str, Any],
    composition: dict[str, Any],
    artifact_count: int,
) -> dict[str, Any]:
    return {
        "rows": {
            "training": split.get("train_count") or 0,
            "validation": split.get("validation_count") or 0,
            "test": split.get("test_count") or 0,
        },
        "samples": {
            "provenance": composition.get("sample_count", 0),
            "synthetic": composition.get("synthetic_sample_count", 0),
            "grounded": composition.get("grounded_sample_count", 0),
            "synthetic_ratio": composition.get("synthetic_ratio", 0),
        },
        "sources": {
            "urls": composition.get("source_url_count", 0),
            "source_revisions": composition.get("source_revision_count", 0),
            "source_chunks": composition.get("source_chunk_count", 0),
            "entailments": composition.get("entailment_count", 0),
            "gaps": composition.get("gap_count", 0),
            "data_designer_jobs": composition.get("data_designer_job_count", 0),
        },
        "artifacts": {
            "count": artifact_count,
        },
    }


def build_dataset_version_manifest(
    dataset_dir: Path,
    *,
    dataset_name: str,
    dataset_role: str | None = None,
    created_at: str | None = None,
    parent_dataset_version_ids: list[str] | None = None,
    curator_job_id: str | None = None,
    curator_config_hash: str | None = None,
    nemo_namespace: str | None = None,
    nemo_dataset_name: str | None = None,
    nemo_files_url: str | None = None,
    nemo_version_id: str | None = None,
    nemo_version_tags: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = read_jsonl_if_exists(dataset_dir / "provenance" / "dataset_samples.jsonl")
    artifacts = list_existing_artifacts(dataset_dir)
    composition = build_source_composition(samples)
    role = infer_dataset_role(dataset_role, composition)
    split = split_metadata(dataset_dir)
    inputs = input_ids(dataset_dir, composition)
    inputs["parent_dataset_version_ids"] = parent_dataset_version_ids or []
    outputs = build_outputs(dataset_dir, artifacts)
    metrics = build_manifest_metrics(
        split=split,
        composition=composition,
        artifact_count=len(artifacts),
    )
    dataset_version_id = stable_id(
        "dsv",
        dataset_name,
        role,
        inputs,
        split,
        outputs.get("hashes", {}),
        metrics,
    )
    manifest = {
        "schema_version": "provenance.v1",
        "dataset_version_id": dataset_version_id,
        "dataset_name": dataset_name,
        "dataset_role": role,
        "created_at": created_at or utc_now(),
        "inputs": inputs,
        "curation": {
            "curator_job_id": curator_job_id,
            "curator_config_hash": curator_config_hash,
        },
        "split": split,
        "outputs": outputs,
        "metrics": metrics,
        "source_composition": composition,
        "artifacts": artifacts,
        "nemo_registration": {
            "namespace": nemo_namespace,
            "dataset_name": nemo_dataset_name,
            "files_url": nemo_files_url,
            "version_id": nemo_version_id,
            "version_tags": nemo_version_tags or [],
        },
    }
    return manifest, artifacts


def build_observability_documents(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    artifacts: list[dict[str, Any]],
    pipeline_run_id: str | None,
    mlflow_tracking_uri: str | None,
    mlflow_experiment_name: str | None,
    mlflow_parent_run_id: str | None,
) -> dict[str, dict[str, Any]]:
    metrics_data = manifest["metrics"]
    composition = manifest["source_composition"]
    metrics: dict[str, int | float] = {
        "dataset.rows.training": metrics_data["rows"]["training"],
        "dataset.rows.validation": metrics_data["rows"]["validation"],
        "dataset.rows.test": metrics_data["rows"]["test"],
        "dataset.samples.provenance.count": metrics_data["samples"]["provenance"],
        "dataset.samples.synthetic.count": metrics_data["samples"]["synthetic"],
        "dataset.samples.grounded.count": metrics_data["samples"]["grounded"],
        "dataset.synthetic_ratio": metrics_data["samples"]["synthetic_ratio"],
        "dataset.sources.count": metrics_data["sources"]["urls"],
        "dataset.source_revisions.count": metrics_data["sources"]["source_revisions"],
        "dataset.source_chunks.count": metrics_data["sources"]["source_chunks"],
        "dataset.entailments.count": metrics_data["sources"]["entailments"],
        "dataset.gap_count": metrics_data["sources"]["gaps"],
        "dataset.artifacts.count": metrics_data["artifacts"]["count"],
    }
    for source_system, count in composition.get("source_systems", {}).items():
        metrics[f"dataset.source_system.{safe_metric_name(source_system)}.samples"] = count
    for source_kind, count in composition.get("source_kinds", {}).items():
        metrics[f"dataset.source_kind.{safe_metric_name(source_kind)}.samples"] = count
    for modality, count in composition.get("modalities", {}).items():
        metrics[f"dataset.modality.{safe_metric_name(modality)}.samples"] = count

    artifact_entries = [file_manifest(manifest_path, dataset_dir, "dataset_version_manifest")]
    artifact_entries.extend(artifacts)
    run_context = {
        "schema_version": "observability.v1",
        "pipeline_stage": "dataset-finalization",
        "created_at": utc_now(),
        "pipeline_run_id": pipeline_run_id,
        "dataset_dir": str(dataset_dir),
        "dataset_name": manifest["dataset_name"],
        "dataset_role": manifest["dataset_role"],
        "dataset_version_id": manifest["dataset_version_id"],
        "mlflow": {
            "tracking_uri": mlflow_tracking_uri,
            "experiment_name": mlflow_experiment_name,
            "parent_run_id": mlflow_parent_run_id,
        },
    }
    service_refs = {
        "schema_version": "observability.v1",
        "services": {},
        "dataset_version": {
            "dataset_version_id": manifest["dataset_version_id"],
            "manifest_path": str(manifest_path),
            "dataset_name": manifest["dataset_name"],
            "dataset_role": manifest["dataset_role"],
        },
        "outputs": {
            "dataset_version_manifest": str(manifest_path),
        },
    }
    artifacts_manifest = {
        "schema_version": "observability.v1",
        "artifacts": artifact_entries,
    }
    return {
        "run_context.json": run_context,
        "metrics.json": metrics,
        "artifacts_manifest.json": artifacts_manifest,
        "service_refs.json": service_refs,
    }


def write_observability_documents(out_dir: Path, documents: dict[str, dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in documents.items():
        (out_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def finalize_dataset(
    dataset_dir: Path,
    *,
    dataset_name: str,
    dataset_role: str | None = None,
    observability_dir: Path | None = None,
    system_prompt: str | None = None,
    generate_missing_samples: bool = True,
    pipeline_run_id: str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_parent_run_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    ensure_dataset_samples(
        dataset_dir,
        system_prompt=system_prompt,
        generate_missing=generate_missing_samples,
    )
    manifest, artifacts = build_dataset_version_manifest(
        dataset_dir,
        dataset_name=dataset_name,
        dataset_role=dataset_role,
        created_at=created_at,
    )
    manifest_path = dataset_dir / "manifests" / "dataset_version_manifest.json"
    write_json(manifest_path, manifest)
    if observability_dir is not None:
        docs = build_observability_documents(
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            artifacts=artifacts,
            pipeline_run_id=pipeline_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_parent_run_id=mlflow_parent_run_id,
        )
        write_observability_documents(observability_dir, docs)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--dataset-name", default=os.getenv("DATASET_NAME"))
    ap.add_argument("--dataset-role", default=DEFAULT_DATASET_ROLE)
    ap.add_argument("--observability-dir", type=Path, default=DEFAULT_OBSERVABILITY_DIR)
    ap.add_argument("--system-prompt", default=os.getenv("DATASET_SYSTEM_PROMPT"))
    ap.add_argument("--no-generate-samples", action="store_true")
    ap.add_argument("--pipeline-run-id", default=os.getenv("PIPELINE_RUN_ID"))
    ap.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    ap.add_argument("--mlflow-experiment-name", default=os.getenv("MLFLOW_EXPERIMENT_NAME"))
    ap.add_argument("--mlflow-parent-run-id", default=os.getenv("MLFLOW_PARENT_RUN_ID"))
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args(argv)
    dataset_name = args.dataset_name or args.dataset_dir.name
    manifest = finalize_dataset(
        args.dataset_dir,
        dataset_name=dataset_name,
        dataset_role=args.dataset_role,
        observability_dir=args.observability_dir,
        system_prompt=args.system_prompt,
        generate_missing_samples=not args.no_generate_samples,
        pipeline_run_id=args.pipeline_run_id,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_parent_run_id=args.mlflow_parent_run_id,
    )
    log.info(
        "Dataset finalization complete: %s",
        args.dataset_dir / "manifests" / "dataset_version_manifest.json",
    )
    print(json.dumps({"dataset_version_id": manifest["dataset_version_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
