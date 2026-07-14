"""Tests for building augmented datasets from Data Designer synthetic rows."""

import json

from scripts.pipeline.build_augmented_dataset_from_synthetic import (
    AugmentedDatasetConfig,
    build_augmented_dataset,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_augmented_dataset_appends_synthetic_only_to_train_splits(tmp_path):
    source = tmp_path / "source"
    experiment = tmp_path / "experiment"
    output = tmp_path / "augmented"
    source.mkdir()
    _write_jsonl(source / "training.jsonl", [
        {"prompt": "Grounded train?", "completion": "Grounded answer.", "system": "system"},
    ])
    _write_jsonl(source / "adapter_train.jsonl", [
        {"prompt": "Adapter train?", "completion": "Adapter answer.", "system": "system"},
    ])
    _write_jsonl(source / "validation.jsonl", [
        {"prompt": "Validation?", "completion": "Validation answer.", "system": "system"},
    ])
    _write_jsonl(source / "test_set.jsonl", [
        {"prompt": "Test?", "completion": "Test answer.", "system": "system"},
    ])
    _write_jsonl(source / "test_set_with_context.jsonl", [
        {"prompt": "Test context?", "completion": "Context answer.", "system": "system"},
    ])
    _write_jsonl(source / "adapter_val.jsonl", [
        {"prompt": "Adapter val?", "completion": "Adapter val answer.", "system": "system"},
    ])
    accepted = [
        {"prompt": "Synthetic one?", "completion": "Synthetic answer one.", "system": "synthetic system"},
        {"prompt": "Synthetic two?", "completion": "Synthetic answer two."},
    ]
    _write_jsonl(experiment / "data_designer" / "accepted_samples.jsonl", accepted)
    (experiment / "data_designer" / "result_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (experiment / "data_designer" / "result_manifest.json").write_text(json.dumps({
        "data_designer_job_id": "job_123",
        "model_provider": "kimi-k2",
        "model": "kimi-k2-6",
        "accepted_synthetic_pair_count": 2,
        "side_effect_columns_excluded": ["qa_pairs_json__reasoning_trace"],
        "source_support_check": "not_run",
    }))
    (experiment / "source_snapshot").mkdir(parents=True, exist_ok=True)
    (experiment / "source_snapshot" / "manifest.json").write_text(json.dumps({"source": "snapshot"}))

    result = build_augmented_dataset(AugmentedDatasetConfig(
        source_dataset_dir=source,
        experiment_dir=experiment,
        output_dir=output,
        accepted_samples_path=experiment / "data_designer" / "accepted_samples.jsonl",
        collection="nim_curated",
        dataset_name="nim_curated_dd_kimi_1b",
        system_prompt="default system",
    ))

    training_rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    adapter_rows = [json.loads(line) for line in (output / "adapter_train.jsonl").read_text().splitlines()]
    validation_rows = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]

    assert len(training_rows) == 3
    assert len(adapter_rows) == 3
    assert len(validation_rows) == 1
    assert training_rows[-1]["system"] == "default system"
    assert result["splits"]["training"]["synthetic_rows_appended"] == 2
    assert result["data_designer"]["job_id"] == "job_123"
    assert result["policy"]["validation_and_test_unchanged"] is True
    assert (output / "manifests" / "dataset_version_manifest.json").exists()
    assert (output / "provenance" / "merge_manifest.json").exists()
