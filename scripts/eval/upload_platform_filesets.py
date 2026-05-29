"""Upload finalized Stage 3 datasets to NeMo Platform FileSets.

This is the Platform-native handoff between dataset registration/finalization
and Customizer. It mirrors the dataset specs used by upload_test_datasets.py,
then uploads each training/evaluation dataset plus lineage sidecars into a
FileSet with the same dataset name. Customizer can then reference the dataset
as `fileset://<workspace>/<dataset-name>`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.upload_test_datasets import (  # noqa: E402
    DEFAULT_BASE_DIR,
    DEFAULT_COLLECTIONS,
    DatasetSpec,
    default_dataset_specs,
    discover_lineage_files,
    file_manifest,
    validate_spec_files,
)
from scripts.nemo_platform import default_nmp_base_url, default_nmp_workspace  # noqa: E402
from scripts.pipeline.provenance import utc_now  # noqa: E402
from scripts.stage3.platform_customizer import fileset_uri_from_ref  # noqa: E402

DEFAULT_OBSERVABILITY_DIR = Path(
    os.getenv("OBSERVABILITY_DIR", "/outputs/observability/platform-filesets")
)


@dataclass(frozen=True)
class PlatformFileSetConfig:
    base_dir: Path
    collections: list[str]
    workspace: str
    nmp_base_url: str
    observability_dir: Path
    include_train: bool
    include_test: bool
    include_context_test: bool
    include_lineage: bool
    dry_run: bool


def build_fileset_plan(
    spec: DatasetSpec,
    *,
    workspace: str,
    include_lineage: bool = True,
) -> dict[str, Any]:
    """Build an auditable upload plan for one dataset FileSet."""
    files: list[dict[str, Any]] = []
    for item in spec.files:
        manifest = file_manifest(item.source, repo_path=item.repo_path)
        manifest.update(
            {
                "local_path": str(item.source),
                "remote_path": item.repo_path,
                "artifact_kind": "dataset_file",
                "split": item.split,
                "required": True,
            }
        )
        files.append(manifest)
    if include_lineage:
        for item in discover_lineage_files(spec.source_dir):
            manifest = file_manifest(item.source, repo_path=item.repo_path)
            manifest.update(
                {
                    "local_path": str(item.source),
                    "remote_path": item.repo_path,
                    "artifact_kind": item.artifact_kind,
                    "required": item.required,
                }
            )
            files.append(manifest)

    fileset_ref = f"{workspace}/{spec.name}"
    return {
        "schema_version": "platform_fileset_upload.v1",
        "created_at": utc_now(),
        "dataset": spec.name,
        "collection": spec.collection,
        "dataset_role": spec.dataset_role,
        "context_baked": spec.context_baked,
        "workspace": workspace,
        "fileset_name": spec.name,
        "fileset_ref": fileset_ref,
        "fileset_uri": fileset_uri_from_ref(fileset_ref),
        "source_dir": str(spec.source_dir),
        "files": files,
        "metrics": {
            "files.count": len(files),
            "bytes.total": sum(int(item.get("bytes", 0)) for item in files),
            "rows.total": sum(int(item.get("rows", 0)) for item in files),
        },
    }


def upload_fileset_plan(plan: dict[str, Any], *, nmp_base_url: str) -> dict[str, Any]:
    """Upload all files in a plan through the NeMo Platform Python SDK."""
    try:
        from nemo_platform import NeMoPlatform  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "NeMo Platform SDK is not installed. Install nemo-platform or run "
            "with --dry-run to generate the FileSet upload manifest only."
        ) from exc

    sdk = NeMoPlatform(base_url=nmp_base_url, workspace=plan["workspace"])
    uploaded: list[dict[str, Any]] = []
    for entry in plan["files"]:
        fileset = sdk.files.upload(
            local_path=entry["local_path"],
            remote_path=entry["remote_path"],
            fileset=plan["fileset_name"],
            workspace=plan["workspace"],
            fileset_auto_create=True,
        )
        uploaded.append(
            {
                "local_path": entry["local_path"],
                "remote_path": entry["remote_path"],
                "fileset_name": getattr(fileset, "name", plan["fileset_name"]),
            }
        )
    return {**plan, "status": "uploaded", "uploaded_files": uploaded}


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "platform_filesets_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def build_manifest(config: PlatformFileSetConfig) -> dict[str, Any]:
    specs = default_dataset_specs(
        config.base_dir,
        config.collections,
        include_train=config.include_train,
        include_test=config.include_test,
        include_context_test=config.include_context_test,
    )
    plans = [
        build_fileset_plan(
            spec,
            workspace=config.workspace,
            include_lineage=config.include_lineage,
        )
        for spec in specs
    ]
    return {
        "schema_version": "platform_fileset_manifest.v1",
        "created_at": utc_now(),
        "workspace": config.workspace,
        "nmp_base_url": config.nmp_base_url,
        "base_dir": str(config.base_dir),
        "collections": config.collections,
        "dry_run": config.dry_run,
        "status": "planned" if config.dry_run else "pending-upload",
        "filesets": plans,
        "metrics": {
            "filesets.count": len(plans),
            "files.count": sum(len(plan["files"]) for plan in plans),
            "bytes.total": sum(plan["metrics"]["bytes.total"] for plan in plans),
            "rows.total": sum(plan["metrics"]["rows.total"] for plan in plans),
        },
    }


def run(config: PlatformFileSetConfig) -> dict[str, Any]:
    specs = default_dataset_specs(
        config.base_dir,
        config.collections,
        include_train=config.include_train,
        include_test=config.include_test,
        include_context_test=config.include_context_test,
    )
    for spec in specs:
        missing = validate_spec_files(spec, require_lineage=config.include_lineage)
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"{spec.name} missing source files: {missing_text}")

    manifest = build_manifest(config)
    if not config.dry_run:
        manifest["filesets"] = [
            upload_fileset_plan(plan, nmp_base_url=config.nmp_base_url)
            for plan in manifest["filesets"]
        ]
        manifest["status"] = "uploaded"
    manifest_path = write_manifest(config.observability_dir, manifest)
    return {
        "status": manifest["status"],
        "filesets": len(manifest["filesets"]),
        "manifest_path": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    ap.add_argument("--collections", nargs="+", default=list(DEFAULT_COLLECTIONS))
    ap.add_argument("--workspace", default=default_nmp_workspace())
    ap.add_argument("--nmp-base-url", default=default_nmp_base_url())
    ap.add_argument("--observability-dir", type=Path, default=DEFAULT_OBSERVABILITY_DIR)
    ap.add_argument("--include-train", action="store_true")
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--include-context-test", action="store_true")
    ap.add_argument("--no-lineage", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    include_train = args.include_train
    include_test = args.include_test
    include_context_test = args.include_context_test
    if not any((include_train, include_test, include_context_test)):
        include_train = include_test = include_context_test = True

    config = PlatformFileSetConfig(
        base_dir=args.base_dir,
        collections=args.collections,
        workspace=args.workspace,
        nmp_base_url=args.nmp_base_url,
        observability_dir=args.observability_dir,
        include_train=include_train,
        include_test=include_test,
        include_context_test=include_context_test,
        include_lineage=not args.no_lineage,
        dry_run=args.dry_run,
    )
    result = run(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
