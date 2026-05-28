"""Register Stage 3 datasets in NeMo Data Store and Entity Store.

The original version of this script was a one-off test-set uploader with
hard-coded NodePorts and embedded Data Store credentials. This version is
Kubernetes-friendly while remaining usable locally:

- Service URLs come from CLI args or environment variables.
- Data Store credentials come from environment variables or Kubernetes Secrets.
- Training, bare test, and context-baked test datasets can be registered.
- MLflow-ready observability JSON files are emitted to an output directory.

The script does not log directly to MLflow. A later MLflow export Job can upload
its structured observability files along with Customizer/Evaluator exports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx


DEFAULT_NAMESPACE = os.getenv("DATASET_NAMESPACE", "default")
DEFAULT_ENTITY_STORE_URL = os.getenv("ENTITY_STORE_URL", "http://nemo-entity-store:8000")
DEFAULT_DATA_STORE_URL = os.getenv("DATA_STORE_URL", "http://nemo-data-store:3000")
DEFAULT_DATA_STORE_GIT_BASE = os.getenv("DATA_STORE_GIT_BASE", DEFAULT_DATA_STORE_URL)
DEFAULT_DATA_STORE_HF_ENDPOINT = os.getenv(
    "DATA_STORE_HF_ENDPOINT",
    "http://nemo-data-store:3000/v1/hf",
)
DEFAULT_BASE_DIR = Path(os.getenv("DATASET_BASE_DIR", "/mnt/nvme2/peft/datasets/v2"))
DEFAULT_OBSERVABILITY_DIR = Path(
    os.getenv("OBSERVABILITY_DIR", "/outputs/observability/dataset-registration")
)
DEFAULT_COLLECTIONS = ("nim_curated", "nemo_usvcs_curated")


@dataclass(frozen=True)
class DatasetFile:
    source: Path
    repo_path: str
    split: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    collection: str
    dataset_role: str
    source_dir: Path
    files: tuple[DatasetFile, ...]
    context_baked: bool = False

    @property
    def entity_ref(self) -> str:
        return f"{DEFAULT_NAMESPACE}/{self.name}"


@dataclass(frozen=True)
class ServiceConfig:
    namespace: str
    entity_store_url: str
    data_store_url: str
    data_store_git_base: str
    data_store_hf_endpoint: str
    data_store_user: str | None
    data_store_password: str | None
    git_user_email: str = "dataset-registration@nemo-peft.local"
    git_user_name: str = "dataset-registration"

    @property
    def data_store_auth(self) -> tuple[str, str] | None:
        if not self.data_store_user:
            return None
        return (self.data_store_user, self.data_store_password or "")

    @property
    def authenticated_git_base(self) -> str:
        return inject_basic_auth(
            self.data_store_git_base,
            self.data_store_user,
            self.data_store_password,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def collection_slug(collection: str) -> str:
    return collection.replace("_", "-")


def safe_metric_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def inject_basic_auth(base_url: str, user: str | None, password: str | None) -> str:
    """Return base_url with URL-encoded basic auth if credentials are supplied."""
    base_url = base_url.rstrip("/")
    if not user:
        return base_url
    parts = urlsplit(base_url)
    if parts.username:
        return base_url
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"Data Store Git base must be an absolute URL, got: {base_url!r}")
    auth = quote(user, safe="")
    if password is not None:
        auth = f"{auth}:{quote(password, safe='')}"
    return urlunsplit(
        (
            parts.scheme,
            f"{auth}@{parts.netloc}",
            parts.path.rstrip("/"),
            parts.query,
            parts.fragment,
        )
    )


def count_jsonl_rows(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def file_manifest(path: Path, repo_path: str | None = None) -> dict[str, Any]:
    manifest = {
        "path": str(path),
        "repo_path": repo_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        manifest["rows"] = count_jsonl_rows(path)
    return manifest


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def default_dataset_specs(
    base_dir: Path,
    collections: list[str],
    include_train: bool,
    include_test: bool,
    include_context_test: bool,
) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for collection in collections:
        coll_dir = base_dir / collection
        slug = collection_slug(collection)
        if include_train:
            specs.append(
                DatasetSpec(
                    name=f"stage3-{slug}",
                    collection=collection,
                    dataset_role="source_entailed",
                    source_dir=coll_dir,
                    files=(
                        DatasetFile(coll_dir / "training.jsonl", "training.jsonl", "training"),
                        DatasetFile(
                            coll_dir / "validation.jsonl", "validation.jsonl", "validation"
                        ),
                    ),
                )
            )
        if include_test:
            specs.append(
                DatasetSpec(
                    name=f"stage3-{slug}-test",
                    collection=collection,
                    dataset_role="evaluation",
                    source_dir=coll_dir,
                    files=(DatasetFile(coll_dir / "test_set.jsonl", "test.jsonl", "test"),),
                )
            )
        if include_context_test:
            specs.append(
                DatasetSpec(
                    name=f"stage3-{slug}-test-with-context",
                    collection=collection,
                    dataset_role="evaluation",
                    source_dir=coll_dir,
                    files=(
                        DatasetFile(
                            coll_dir / "test_set_with_context.jsonl",
                            "test.jsonl",
                            "test",
                        ),
                    ),
                    context_baked=True,
                )
            )
    return specs


def discover_provenance_artifacts(source_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        source_dir / "manifests" / "crawl_run.json",
        source_dir / "manifests" / "dataset_version_manifest.json",
        source_dir / "provenance" / "dataset_version_manifest.json",
        source_dir / "provenance" / "source_revisions.jsonl",
        source_dir / "provenance" / "source_chunks.jsonl",
        source_dir / "provenance" / "entailments.jsonl",
        source_dir / "provenance" / "dataset_samples.jsonl",
        source_dir / "provenance" / "delta_manifest.json",
        source_dir / "provenance" / "gap_manifest.json",
        source_dir / "bias_report.json",
        source_dir / "validation_report.json",
    ]
    artifacts = []
    for item in candidates:
        if item.exists():
            rel = item.relative_to(source_dir)
            manifest = file_manifest(item, repo_path=str(rel))
            manifest["artifact_path"] = str(rel)
            artifacts.append(manifest)
    return artifacts


def load_dataset_version_manifest(source_dir: Path) -> dict[str, Any]:
    for candidate in (
        source_dir / "manifests" / "dataset_version_manifest.json",
        source_dir / "provenance" / "dataset_version_manifest.json",
    ):
        data = load_json_if_exists(candidate)
        if data:
            return data
    return {}


def build_description(
    spec: DatasetSpec,
    row_counts: dict[str, int],
    dataset_version: dict[str, Any],
    pipeline_run_id: str | None,
    mlflow_parent_run_id: str | None,
) -> str:
    row_text = ", ".join(f"{split}={count}" for split, count in sorted(row_counts.items()))
    if spec.context_baked:
        desc = (
            f"Stage 3 context-baked evaluation dataset for {spec.collection} ({row_text}). "
            "Prompts include retrieved context produced by bake_context_into_testset.py."
        )
    elif spec.dataset_role == "evaluation":
        desc = f"Stage 3 held-out evaluation dataset for {spec.collection} ({row_text})."
    else:
        desc = f"Stage 3 source-grounded training dataset for {spec.collection} ({row_text})."

    lineage = []
    dataset_version_id = dataset_version.get("dataset_version_id")
    if dataset_version_id:
        lineage.append(f"dataset_version_id={dataset_version_id}")
    if pipeline_run_id:
        lineage.append(f"pipeline_run_id={pipeline_run_id}")
    if mlflow_parent_run_id:
        lineage.append(f"mlflow_parent_run_id={mlflow_parent_run_id}")
    if lineage:
        desc += " Lineage: " + ", ".join(lineage) + "."
    return desc


def build_entity_store_payload(
    spec: DatasetSpec,
    config: ServiceConfig,
    row_counts: dict[str, int],
    dataset_version: dict[str, Any] | None = None,
    pipeline_run_id: str | None = None,
    mlflow_parent_run_id: str | None = None,
) -> dict[str, Any]:
    dataset_version = dataset_version or {}
    return {
        "name": spec.name,
        "namespace": config.namespace,
        "description": build_description(
            spec,
            row_counts,
            dataset_version,
            pipeline_run_id,
            mlflow_parent_run_id,
        ),
        "format": "hf",
        "files_url": f"hf://datasets/{config.namespace}/{spec.name}",
        "hf_endpoint": config.data_store_hf_endpoint,
    }


def validate_spec_files(spec: DatasetSpec) -> list[Path]:
    return [item.source for item in spec.files if not item.source.exists()]


def run_cmd(
    args: list[str],
    cwd: Path | None = None,
    allow_fail: bool = False,
) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=cwd, capture_output=True)
    if result.returncode != 0 and not allow_fail:
        sys.stderr.write(result.stderr.decode())
        raise subprocess.CalledProcessError(result.returncode, args)
    return result


def create_dataset_repo(spec: DatasetSpec, config: ServiceConfig) -> str:
    response = httpx.post(
        f"{config.data_store_url.rstrip('/')}/v1/hf/api/repos/create",
        auth=config.data_store_auth,
        json={
            "type": "dataset",
            "name": spec.name,
            "organization": config.namespace,
            "private": False,
        },
        timeout=15,
    )
    if response.status_code in (200, 201):
        return "created"
    if response.status_code == 409 or "already created" in response.text:
        return "exists"
    raise RuntimeError(
        f"create repo failed for {spec.name}: {response.status_code} {response.text}"
    )


def push_dataset_files(spec: DatasetSpec, config: ServiceConfig) -> dict[str, Any]:
    repo_id = f"{config.namespace}/{spec.name}"
    clone_url = f"{config.authenticated_git_base.rstrip('/')}/{repo_id}.git"
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        run_cmd(["git", "clone", clone_url, str(td_path / "repo")])
        repo = td_path / "repo"
        for item in spec.files:
            dst = repo / item.repo_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item.source, dst)
        run_cmd(["git", "config", "user.email", config.git_user_email], cwd=repo)
        run_cmd(["git", "config", "user.name", config.git_user_name], cwd=repo)
        run_cmd(["git", "add", *[item.repo_path for item in spec.files]], cwd=repo)
        row_counts = {item.split: count_jsonl_rows(item.source) for item in spec.files}
        row_text = ", ".join(f"{split}={count}" for split, count in sorted(row_counts.items()))
        commit = run_cmd(
            ["git", "commit", "-m", f"register {spec.name}: {row_text}"],
            cwd=repo,
            allow_fail=True,
        )
        if commit.returncode != 0:
            if b"nothing to commit" in commit.stdout or b"nothing to commit" in commit.stderr:
                return {"git_status": "unchanged", "row_counts": row_counts}
            sys.stderr.write(commit.stderr.decode())
            raise subprocess.CalledProcessError(commit.returncode, commit.args)
        run_cmd(["git", "push", "origin", "main"], cwd=repo)
    return {"git_status": "pushed", "row_counts": row_counts}


def register_in_entity_store(payload: dict[str, Any], config: ServiceConfig) -> str:
    url = f"{config.entity_store_url.rstrip('/')}/v1/datasets"
    response = httpx.post(url, json=payload, timeout=30)
    if response.status_code in (200, 201):
        return "created"
    if response.status_code == 409:
        patch_payload = {
            "description": payload["description"],
            "format": payload["format"],
            "files_url": payload["files_url"],
            "hf_endpoint": payload["hf_endpoint"],
        }
        patch_url = f"{url}/{payload['namespace']}/{payload['name']}"
        patch = httpx.patch(patch_url, json=patch_payload, timeout=15)
        if patch.status_code in (200, 201, 204):
            return "patched"
        raise RuntimeError(
            f"patch entity failed for {payload['name']}: {patch.status_code} {patch.text}"
        )
    raise RuntimeError(
        f"register entity failed for {payload['name']}: {response.status_code} {response.text}"
    )


def build_observability_documents(
    specs: list[DatasetSpec],
    results: list[dict[str, Any]],
    config: ServiceConfig,
    base_dir: Path,
    pipeline_run_id: str | None,
    mlflow_tracking_uri: str | None,
    mlflow_experiment_name: str | None,
    mlflow_parent_run_id: str | None,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, int | float] = {"datasets.count": len(results)}
    artifact_entries: list[dict[str, Any]] = []
    dataset_refs: list[dict[str, Any]] = []

    for spec, result in zip(specs, results):
        metric_prefix = f"dataset.{safe_metric_name(spec.name)}"
        for item in spec.files:
            manifest = file_manifest(item.source, repo_path=item.repo_path)
            manifest.update({
                "dataset": spec.name,
                "collection": spec.collection,
                "split": item.split,
                "artifact_kind": "dataset_file",
            })
            artifact_entries.append(manifest)
            if "rows" in manifest:
                metrics[f"{metric_prefix}.rows.{item.split}"] = manifest["rows"]
            metrics[f"{metric_prefix}.bytes.{safe_metric_name(item.repo_path)}"] = manifest["bytes"]

        provenance = discover_provenance_artifacts(spec.source_dir)
        for artifact in provenance:
            artifact.update({
                "dataset": spec.name,
                "collection": spec.collection,
                "artifact_kind": "provenance_or_report",
            })
            artifact_entries.append(artifact)
            if "rows" in artifact:
                metric_name = f"{metric_prefix}.{safe_metric_name(artifact['artifact_path'])}.rows"
                metrics[metric_name] = artifact["rows"]

        dataset_version = load_dataset_version_manifest(spec.source_dir)
        dataset_refs.append({
            "dataset": spec.name,
            "collection": spec.collection,
            "dataset_role": spec.dataset_role,
            "entity_ref": f"{config.namespace}/{spec.name}",
            "files_url": f"hf://datasets/{config.namespace}/{spec.name}",
            "hf_endpoint": config.data_store_hf_endpoint,
            "dataset_version_id": dataset_version.get("dataset_version_id"),
            "repo_status": result.get("repo_status"),
            "git_status": result.get("git_status"),
            "entity_status": result.get("entity_status"),
        })

    run_context = {
        "schema_version": "observability.v1",
        "pipeline_stage": "dataset-registration",
        "created_at": utc_now(),
        "pipeline_run_id": pipeline_run_id,
        "base_dir": str(base_dir),
        "collections": sorted({spec.collection for spec in specs}),
        "mlflow": {
            "tracking_uri": mlflow_tracking_uri,
            "experiment_name": mlflow_experiment_name,
            "parent_run_id": mlflow_parent_run_id,
        },
    }
    service_refs = {
        "schema_version": "observability.v1",
        "services": {
            "data_store_url": config.data_store_url,
            "data_store_git_base": config.data_store_git_base,
            "data_store_hf_endpoint": config.data_store_hf_endpoint,
            "entity_store_url": config.entity_store_url,
            "namespace": config.namespace,
        },
        "datasets": dataset_refs,
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


def register_dataset_spec(
    spec: DatasetSpec,
    config: ServiceConfig,
    pipeline_run_id: str | None,
    mlflow_parent_run_id: str | None,
) -> dict[str, Any]:
    missing = validate_spec_files(spec)
    if missing:
        missing_text = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"{spec.name} missing source files: {missing_text}")
    repo_status = create_dataset_repo(spec, config)
    push_result = push_dataset_files(spec, config)
    dataset_version = load_dataset_version_manifest(spec.source_dir)
    payload = build_entity_store_payload(
        spec,
        config,
        push_result["row_counts"],
        dataset_version=dataset_version,
        pipeline_run_id=pipeline_run_id,
        mlflow_parent_run_id=mlflow_parent_run_id,
    )
    entity_status = register_in_entity_store(payload, config)
    return {
        "dataset": spec.name,
        "collection": spec.collection,
        "repo_status": repo_status,
        "git_status": push_result["git_status"],
        "entity_status": entity_status,
        "row_counts": push_result["row_counts"],
        "entity_payload": payload,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    ap.add_argument("--collections", nargs="+", default=list(DEFAULT_COLLECTIONS))
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    ap.add_argument("--entity-store-url", default=DEFAULT_ENTITY_STORE_URL)
    ap.add_argument("--data-store-url", default=DEFAULT_DATA_STORE_URL)
    ap.add_argument("--data-store-git-base", default=DEFAULT_DATA_STORE_GIT_BASE)
    ap.add_argument("--data-store-hf-endpoint", default=DEFAULT_DATA_STORE_HF_ENDPOINT)
    ap.add_argument("--data-store-user", default=os.getenv("DATA_STORE_USER"))
    ap.add_argument("--data-store-password", default=os.getenv("DATA_STORE_PASSWORD"))
    ap.add_argument("--include-train", action="store_true")
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--include-context-test", action="store_true")
    ap.add_argument("--observability-dir", type=Path, default=DEFAULT_OBSERVABILITY_DIR)
    ap.add_argument("--pipeline-run-id", default=os.getenv("PIPELINE_RUN_ID"))
    ap.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    ap.add_argument("--mlflow-experiment-name", default=os.getenv("MLFLOW_EXPERIMENT_NAME"))
    ap.add_argument("--mlflow-parent-run-id", default=os.getenv("MLFLOW_PARENT_RUN_ID"))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build specs and observability only; do not call NeMo services",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    include_train = args.include_train
    include_test = args.include_test
    include_context_test = args.include_context_test
    if not any((include_train, include_test, include_context_test)):
        include_train = include_test = include_context_test = True

    config = ServiceConfig(
        namespace=args.namespace,
        entity_store_url=args.entity_store_url,
        data_store_url=args.data_store_url,
        data_store_git_base=args.data_store_git_base,
        data_store_hf_endpoint=args.data_store_hf_endpoint,
        data_store_user=args.data_store_user,
        data_store_password=args.data_store_password,
    )
    specs = default_dataset_specs(
        args.base_dir,
        args.collections,
        include_train=include_train,
        include_test=include_test,
        include_context_test=include_context_test,
    )

    results: list[dict[str, Any]] = []
    for spec in specs:
        print(f"=== {spec.name} ({spec.collection}, {spec.dataset_role}) ===")
        missing = validate_spec_files(spec)
        if missing:
            print(f"FATAL: missing source files: {', '.join(str(p) for p in missing)}")
            return 1
        if args.dry_run:
            row_counts = {item.split: count_jsonl_rows(item.source) for item in spec.files}
            results.append({
                "dataset": spec.name,
                "collection": spec.collection,
                "repo_status": "dry-run",
                "git_status": "dry-run",
                "entity_status": "dry-run",
                "row_counts": row_counts,
            })
            print(f"  dry-run row counts: {row_counts}")
            continue
        result = register_dataset_spec(
            spec,
            config,
            pipeline_run_id=args.pipeline_run_id,
            mlflow_parent_run_id=args.mlflow_parent_run_id,
        )
        results.append(result)
        print(
            f"  repo={result['repo_status']} git={result['git_status']} "
            f"entity={result['entity_status']} rows={result['row_counts']}"
        )

    docs = build_observability_documents(
        specs,
        results,
        config,
        args.base_dir,
        pipeline_run_id=args.pipeline_run_id,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_parent_run_id=args.mlflow_parent_run_id,
    )
    write_observability_documents(args.observability_dir, docs)
    print(f"\nObservability written to: {args.observability_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
