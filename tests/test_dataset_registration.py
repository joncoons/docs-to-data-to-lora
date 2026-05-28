import json

from scripts.eval.upload_test_datasets import (
    ServiceConfig,
    build_entity_store_payload,
    build_observability_documents,
    default_dataset_specs,
    inject_basic_auth,
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


def test_build_observability_documents_records_metrics_and_refs(tmp_path):
    coll = tmp_path / "nim_curated"
    coll.mkdir()
    (coll / "training.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (coll / "validation.jsonl").write_text('{"a": 3}\n')
    provenance = coll / "provenance"
    provenance.mkdir()
    (provenance / "dataset_version_manifest.json").write_text(json.dumps({
        "dataset_version_id": "dsv_nim_001",
    }))
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
    assert service_refs["datasets"][0]["dataset_version_id"] == "dsv_nim_001"
    assert service_refs["datasets"][0]["entity_ref"] == "default/stage3-nim-curated"


def test_write_observability_documents_creates_json_files(tmp_path):
    out = tmp_path / "observability"

    write_observability_documents(out, {"metrics.json": {"datasets.count": 1}})

    assert json.loads((out / "metrics.json").read_text()) == {"datasets.count": 1}
