"""Tests for NeMo Curator handoff preparation and collection."""

import json

from scripts.pipeline.curator_handoff import (
    CuratorHandoffConfig,
    normalize_sample_for_curator,
    run,
)


def _sample(sample_id, prompt="Question?", completion="Answer.", origin="source_entailed"):
    return {
        "schema_version": "provenance.v1",
        "sample_id": sample_id,
        "origin": origin,
        "task_type": "qa",
        "prompt": prompt,
        "completion": completion,
        "system": "System.",
        "lineage": {
            "entailment_ids": ["ent_1"] if origin == "source_entailed" else [],
            "source_revision_ids": ["srcrev_1"] if origin == "source_entailed" else [],
            "source_chunk_ids": ["chunk_1"],
            "source_systems": ["web_crawl"] if origin == "source_entailed" else ["data_designer"],
            "source_kinds": ["web_page"] if origin == "source_entailed" else ["synthetic_gapfill"],
            "modalities": ["text"],
        },
        "quality": {},
        "metadata": {
            "source_url": "https://docs.example.com/source",
            "product_family": "nim",
        },
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _config(tmp_path, mode="prepare", accepted_jsonl=None, rejected_jsonl=None):
    config_file = tmp_path / "curator.yaml"
    config_file.write_text("name: test-curator\ntext_field: text\n")
    return CuratorHandoffConfig(
        dataset_dir=tmp_path / "nim_curated",
        collection="nim_curated",
        mode=mode,
        observability_dir=tmp_path / "observability",
        config_file=config_file,
        train_ratio=0.5,
        split_seed=7,
        curator_job_id="curator_job_1" if mode != "prepare" else None,
        accepted_jsonl=accepted_jsonl,
        rejected_jsonl=rejected_jsonl,
    )


def _write_samples(dataset_dir):
    _write_jsonl(
        dataset_dir / "provenance" / "dataset_samples.jsonl",
        [
            _sample("sample_1", "Q1?", "A1."),
            _sample("sample_2", "Q2?", "A2.", origin="synthetic_gapfill"),
            _sample("sample_3", "Q3?", "A3."),
        ],
    )


def test_normalize_sample_for_curator_uses_text_field_and_lineage():
    record = normalize_sample_for_curator(_sample("sample_1", "What?", "This."))

    assert record["id"] == "sample_1"
    assert record["sample_id"] == "sample_1"
    assert record["text"] == "Question: What?\nAnswer: This."
    assert record["source_systems"] == ["web_crawl"]


def test_run_prepare_writes_curator_input_plan_and_observability(tmp_path):
    config = _config(tmp_path)
    _write_samples(config.dataset_dir)

    result = run(config)

    assert result["mode"] == "prepare"
    assert result["input_samples"] == 3
    assert (config.dataset_dir / "curator" / "input" / "dataset_samples.jsonl").exists()
    assert (config.dataset_dir / "curator" / "curator_config.yaml").exists()
    plan = json.loads((config.dataset_dir / "curator" / "submission_plan.json").read_text())
    assert plan["native_service"] == "NeMo Curator"
    assert plan["input"]["document_count"] == 3
    metrics = json.loads((config.observability_dir / "metrics.json").read_text())
    assert metrics["curator.samples.input"] == 3
    assert metrics["curator.samples.accepted"] == 0


def test_run_collect_writes_curated_splits_manifest_and_observability(tmp_path):
    accepted_jsonl = tmp_path / "accepted.jsonl"
    rejected_jsonl = tmp_path / "rejected.jsonl"
    _write_jsonl(accepted_jsonl, [{"id": "sample_1"}, {"sample_id": "sample_2"}])
    _write_jsonl(rejected_jsonl, [{"id": "sample_3", "rejection_reason": "short_answer"}])
    config = _config(
        tmp_path,
        mode="prepare-and-collect",
        accepted_jsonl=accepted_jsonl,
        rejected_jsonl=rejected_jsonl,
    )
    _write_samples(config.dataset_dir)

    result = run(config)

    assert result["metrics"]["accepted_samples"] == 2
    assert result["metrics"]["rejected_samples"] == 1
    assert (config.dataset_dir / "training.jsonl").exists()
    assert (config.dataset_dir / "validation.jsonl").exists()
    accepted = [
        json.loads(line)
        for line in (config.dataset_dir / "curator" / "accepted_samples.jsonl").read_text().splitlines()
    ]
    rejected = [
        json.loads(line)
        for line in (config.dataset_dir / "curator" / "rejected_samples.jsonl").read_text().splitlines()
    ]
    assert accepted[0]["quality"]["curator_job_id"] == "curator_job_1"
    assert accepted[0]["quality"]["curator_config_hash"].startswith("sha256:")
    assert rejected[0]["metadata"]["curation"]["rejection_reason"] == "short_answer"

    manifest = json.loads((config.dataset_dir / "curator" / "curation_manifest.json").read_text())
    assert manifest["curator_job_id"] == "curator_job_1"
    assert manifest["metrics"]["training_rows"] + manifest["metrics"]["validation_rows"] == 2
    metrics = json.loads((config.observability_dir / "metrics.json").read_text())
    service_refs = json.loads((config.observability_dir / "service_refs.json").read_text())
    assert metrics["curator.samples.accepted"] == 2
    assert metrics["curator.rejection_reason.short_answer.count"] == 1
    assert service_refs["curator"]["job_id"] == "curator_job_1"
