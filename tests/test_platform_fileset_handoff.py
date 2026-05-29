"""Tests for NeMo Platform FileSet dataset handoff."""

import json

from scripts.eval.upload_platform_filesets import (
    PlatformFileSetConfig,
    build_fileset_plan,
    run,
)
from scripts.eval.upload_test_datasets import default_dataset_specs


def _write_dataset_tree(base_dir):
    coll = base_dir / "nim_curated"
    (coll / "manifests").mkdir(parents=True)
    (coll / "provenance").mkdir()
    (coll / "training.jsonl").write_text('{"prompt":"Q","completion":"A"}' + "\n")
    (coll / "validation.jsonl").write_text('{"prompt":"V","completion":"A"}' + "\n")
    (coll / "test_set.jsonl").write_text('{"prompt":"T","completion":"A"}' + "\n")
    (coll / "test_set_with_context.jsonl").write_text(
        '{"prompt":"C","completion":"A"}' + "\n"
    )
    (coll / "manifests" / "dataset_version_manifest.json").write_text(json.dumps({
        "dataset_version_id": "dsv_test",
    }) + "\n")
    (coll / "provenance" / "source_chunks.jsonl").write_text(
        '{"source_chunk_id":"chunk"}' + "\n"
    )
    return coll


def test_build_fileset_plan_preserves_dataset_and_lineage_paths(tmp_path):
    _write_dataset_tree(tmp_path)
    spec = default_dataset_specs(
        tmp_path,
        ["nim_curated"],
        include_train=True,
        include_test=False,
        include_context_test=False,
    )[0]

    plan = build_fileset_plan(spec, workspace="default", include_lineage=True)

    assert plan["fileset_uri"] == "fileset://default/stage3-nim-curated"
    remote_paths = {item["remote_path"] for item in plan["files"]}
    assert {"training.jsonl", "validation.jsonl"}.issubset(remote_paths)
    assert "manifests/dataset_version_manifest.json" in remote_paths
    assert "provenance/source_chunks.jsonl" in remote_paths
    assert plan["metrics"]["files.count"] == len(plan["files"])
    assert plan["metrics"]["rows.total"] >= 3


def test_run_dry_run_writes_platform_fileset_manifest(tmp_path):
    _write_dataset_tree(tmp_path)
    out_dir = tmp_path / "observability"
    config = PlatformFileSetConfig(
        base_dir=tmp_path,
        collections=["nim_curated"],
        workspace="default",
        nmp_base_url="http://nemo-platform-api:8080",
        observability_dir=out_dir,
        include_train=True,
        include_test=False,
        include_context_test=False,
        include_lineage=True,
        dry_run=True,
    )

    result = run(config)

    assert result["status"] == "planned"
    manifest = json.loads((out_dir / "platform_filesets_manifest.json").read_text())
    assert manifest["dry_run"] is True
    assert manifest["filesets"][0]["fileset_uri"] == "fileset://default/stage3-nim-curated"
