import json
import subprocess
from pathlib import Path

from scripts.eval import upload_test_datasets as upload
from scripts.eval.upload_test_datasets import (
    ServiceConfig,
    build_entity_store_payload,
    build_observability_documents,
    default_dataset_specs,
    discover_lineage_files,
    inject_basic_auth,
    push_dataset_files,
    validate_spec_files,
    write_observability_documents,
)


def _config():
    return ServiceConfig(
        namespace="default",
        entity_store_url="http://entity-store:8000",
        data_store_url="http://data-store:3000",
        data_store_git_base="http://data-store:3000",
        data_store_hf_endpoint="http://data-store:3000/v1/hf",
        data_store_user=None,
        data_store_password=None,
    )


def test_default_dataset_specs_include_training_and_eval_variants(tmp_path):
    specs = default_dataset_specs(
        tmp_path,
        ["nim_curated"],
        include_train=True,
        include_test=True,
        include_context_test=True,
    )

    assert [spec.name for spec in specs] == [
        "stage3-nim-curated",
        "stage3-nim-curated-test",
        "stage3-nim-curated-test-with-context",
    ]
    assert specs[0].files[0].repo_path == "training.jsonl"
    assert specs[0].files[1].repo_path == "validation.jsonl"
    assert specs[1].files[0].source == tmp_path / "nim_curated" / "test_set.jsonl"
    assert specs[2].context_baked is True


def test_inject_basic_auth_url_encodes_credentials():
    url = inject_basic_auth("http://data-store:3000", "user@example.com", "p@ ss")

    assert url == "http://user%40example.com:p%40%20ss@data-store:3000"


def test_build_entity_store_payload_includes_lineage_in_description(tmp_path):
    spec = default_dataset_specs(
        tmp_path,
        ["nim_curated"],
        include_train=True,
        include_test=False,
        include_context_test=False,
    )[0]

    payload = build_entity_store_payload(
        spec,
        _config(),
        {"training": 10, "validation": 2},
        dataset_version={"dataset_version_id": "dsv_nim_001"},
        pipeline_run_id="pipe-123",
        mlflow_parent_run_id="mlflow-run-abc",
    )

    assert payload["name"] == "stage3-nim-curated"
    assert payload["namespace"] == "default"
    assert payload["files_url"] == "hf://datasets/default/stage3-nim-curated"
    assert payload["hf_endpoint"] == "http://data-store:3000/v1/hf"
    assert "dataset_version_id=dsv_nim_001" in payload["description"]
    assert "mlflow_parent_run_id=mlflow-run-abc" in payload["description"]


def test_validate_spec_files_requires_dataset_version_manifest(tmp_path):
    coll = tmp_path / "nim_curated"
    coll.mkdir()
    (coll / "training.jsonl").write_text('{"a": 1}\n')
    (coll / "validation.jsonl").write_text('{"a": 2}\n')
    spec = default_dataset_specs(
        tmp_path,
        ["nim_curated"],
        include_train=True,
        include_test=False,
        include_context_test=False,
    )[0]

    assert validate_spec_files(spec, require_lineage=False) == []

    missing = validate_spec_files(spec, require_lineage=True)

    assert missing == [coll / "manifests" / "dataset_version_manifest.json"]


def test_discover_lineage_files_finds_finalized_manifest_and_sidecars(tmp_path):
    coll = tmp_path / "nim_curated"
    (coll / "manifests").mkdir(parents=True)
    (coll / "provenance").mkdir()
    (coll / "manifests" / "dataset_version_manifest.json").write_text("{}")
    (coll / "provenance" / "source_chunks.jsonl").write_text('{"id": "c1"}\n')
    (coll / "provenance" / "dataset_samples.jsonl").write_text('{"id": "s1"}\n')

    lineage_files = discover_lineage_files(coll)

    assert [item.repo_path for item in lineage_files] == [
        "manifests/dataset_version_manifest.json",
        "provenance/source_chunks.jsonl",
        "provenance/dataset_samples.jsonl",
    ]


def test_push_dataset_files_includes_lineage_artifacts(tmp_path, monkeypatch):
    coll = tmp_path / "nim_curated"
    (coll / "manifests").mkdir(parents=True)
    (coll / "provenance").mkdir()
    (coll / "training.jsonl").write_text('{"a": 1}\n')
    (coll / "validation.jsonl").write_text('{"a": 2}\n')
    (coll / "manifests" / "dataset_version_manifest.json").write_text("{}")
    (coll / "provenance" / "dataset_samples.jsonl").write_text('{"id": "s1"}\n')
    spec = default_dataset_specs(
        tmp_path,
        ["nim_curated"],
        include_train=True,
        include_test=False,
        include_context_test=False,
    )[0]
    commands = []

    def fake_run_cmd(args, cwd=None, allow_fail=False):
        commands.append(args)
        if args[:2] == ["git", "clone"]:
            requested_repo = Path(args[-1])
            requested_repo.mkdir(parents=True)
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(upload, "run_cmd", fake_run_cmd)

    result = push_dataset_files(spec, _config())

    assert result["lineage_files"] == [
        "manifests/dataset_version_manifest.json",
        "provenance/dataset_samples.jsonl",
    ]
    git_add = [args for args in commands if args[:2] == ["git", "add"]][0]
    assert "training.jsonl" in git_add
    assert "validation.jsonl" in git_add
    assert "manifests/dataset_version_manifest.json" in git_add
    assert "provenance/dataset_samples.jsonl" in git_add


def test_build_observability_documents_records_metrics_and_refs(tmp_path):
    coll = tmp_path / "nim_curated"
    coll.mkdir()
    (coll / "training.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (coll / "validation.jsonl").write_text('{"a": 3}\n')
    manifests = coll / "manifests"
    manifests.mkdir()
    provenance = coll / "provenance"
    provenance.mkdir()
    (manifests / "dataset_version_manifest.json").write_text(json.dumps({
        "dataset_version_id": "dsv_nim_001",
    }))
    dataset_samples = [
        {
            "origin": "source_entailed",
            "lineage": {
                "entailment_ids": ["ent_web"],
                "source_revision_ids": ["srcrev_web"],
                "source_chunk_ids": ["chunk_web"],
                "source_systems": ["web_crawl"],
                "source_kinds": ["web_page"],
                "modalities": ["text"],
            },
            "metadata": {
                "source_url": "https://docs.example.com/web",
            },
        },
        {
            "origin": "source_entailed",
            "lineage": {
                "entailment_ids": ["ent_image"],
                "source_revision_ids": ["srcrev_image"],
                "source_chunk_ids": ["chunk_image"],
                "source_systems": ["image_dense_caption"],
                "source_kinds": ["dense_caption"],
                "modalities": ["image"],
            },
            "metadata": {
                "source_url": "s3://captures/image.png",
            },
        },
        {
            "origin": "synthetic_gapfill",
            "lineage": {
                "source_systems": ["data_designer"],
                "source_kinds": ["synthetic_gapfill"],
                "modalities": ["text"],
            },
            "metadata": {
                "source_url": "<synthetic>",
            },
        },
    ]
    (provenance / "dataset_samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dataset_samples)
    )
    specs = default_dataset_specs(
        tmp_path,
        ["nim_curated"],
        include_train=True,
        include_test=False,
        include_context_test=False,
    )
    results = [{
        "repo_status": "created",
        "git_status": "pushed",
        "entity_status": "created",
        "lineage_files": [
            "manifests/dataset_version_manifest.json",
            "provenance/dataset_samples.jsonl",
        ],
    }]

    docs = build_observability_documents(
        specs,
        results,
        _config(),
        tmp_path,
        pipeline_run_id="pipe-123",
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="docs-to-data-to-lora",
        mlflow_parent_run_id="run-abc",
    )

    metrics = docs["metrics.json"]
    service_refs = docs["service_refs.json"]
    assert metrics["datasets.count"] == 1
    assert metrics["dataset.stage3_nim_curated.rows.training"] == 2
    assert metrics["dataset.stage3_nim_curated.rows.validation"] == 1
    assert metrics["dataset.stage3_nim_curated.lineage_files.uploaded.count"] == 2
    assert metrics["dataset.stage3_nim_curated.samples.provenance.count"] == 3
    assert metrics["dataset.stage3_nim_curated.sources.count"] == 3
    assert metrics["dataset.stage3_nim_curated.source_revisions.count"] == 2
    assert metrics["dataset.stage3_nim_curated.source_chunks.count"] == 2
    assert metrics["dataset.stage3_nim_curated.entailments.count"] == 2
    assert metrics["dataset.stage3_nim_curated.samples.synthetic.count"] == 1
    assert metrics["dataset.stage3_nim_curated.samples.grounded.count"] == 2
    assert metrics["dataset.stage3_nim_curated.synthetic_ratio"] == 0.333333
    assert metrics["dataset.stage3_nim_curated.source_system.web_crawl.samples"] == 1
    assert metrics["dataset.stage3_nim_curated.source_system.image_dense_caption.samples"] == 1
    assert metrics["dataset.stage3_nim_curated.source_system.data_designer.samples"] == 1
    assert metrics["dataset.stage3_nim_curated.source_kind.web_page.samples"] == 1
    assert metrics["dataset.stage3_nim_curated.modality.text.samples"] == 2
    assert metrics["dataset.stage3_nim_curated.modality.image.samples"] == 1
    artifacts = {
        item.get("artifact_path") or item.get("repo_path"): item
        for item in docs["artifacts_manifest.json"]["artifacts"]
    }
    assert artifacts["training.jsonl"]["artifact_kind"] == "uploaded_dataset_file"
    assert artifacts["training.jsonl"]["uploaded_to_data_store"] is True
    assert (
        artifacts["manifests/dataset_version_manifest.json"]["artifact_kind"]
        == "uploaded_lineage_file"
    )
    assert artifacts["provenance/dataset_samples.jsonl"]["uploaded_to_data_store"] is True
    assert service_refs["datasets"][0]["dataset_version_id"] == "dsv_nim_001"
    assert service_refs["datasets"][0]["entity_ref"] == "default/stage3-nim-curated"
    assert service_refs["datasets"][0]["lineage_files"] == [
        "manifests/dataset_version_manifest.json",
        "provenance/dataset_samples.jsonl",
    ]
    assert service_refs["datasets"][0]["source_composition"]["source_revision_count"] == 2


def test_write_observability_documents_creates_json_files(tmp_path):
    out = tmp_path / "observability"

    write_observability_documents(out, {"metrics.json": {"datasets.count": 1}})

    assert json.loads((out / "metrics.json").read_text()) == {"datasets.count": 1}
