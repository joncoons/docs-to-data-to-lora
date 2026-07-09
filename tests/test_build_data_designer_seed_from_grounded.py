"""Tests for grounded-training seed preparation for NeMo Data Designer."""

import csv
import json

import pytest

from scripts.pipeline.build_data_designer_seed_from_grounded import (
    SeedConfig,
    load_candidate_rows,
    parse_json_object,
    run,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _config(tmp_path, *, mode="prepare", allow_external_kimi=False):
    return SeedConfig(
        dataset_dir=tmp_path / "nim_curated",
        output_dir=tmp_path / "experiment",
        collection="nim_curated",
        source_dataset_name="nim_curated",
        mode=mode,
        seed_count=2,
        pairs_per_seed=3,
        random_seed=7,
        judge_api_url="https://maas.example.test/kimi/v1",
        judge_model="kimi-k2-6",
        judge_api_key_env="KIMI_KEY",
        allow_external_kimi=allow_external_kimi,
        temperature=0.0,
        max_tokens=8192,
        timeout_s=30.0,
    )


def _write_dataset(dataset_dir):
    training = [
        {
            "prompt": "Which endpoint does NIM expose for metrics?",
            "completion": "NIM exposes metrics from the status endpoint.",
            "system": "system",
        },
        {
            "prompt": "How do you deploy NIM with Helm?",
            "completion": "Install the chart after creating the required secrets and storage.",
            "system": "system",
        },
        {
            "prompt": "Holdout prompt must not seed generation.",
            "completion": "Holdout answer must be excluded.",
            "system": "system",
        },
    ]
    stage2 = [
        {
            "passage_id": "p0",
            "source_url": "https://docs.nvidia.com/nim/status.html",
            "product_family": "NIM",
            "stage": "1a",
            "question": training[0]["prompt"],
            "answer": training[0]["completion"],
            "context": "NIM exposes operational status and metrics from a status endpoint.",
            "qa_type": None,
            "instr_type": None,
        },
        {
            "passage_id": "p1",
            "source_url": "https://docs.nvidia.com/nim/helm.html",
            "product_family": "NIM",
            "stage": "1c",
            "question": training[1]["prompt"],
            "answer": training[1]["completion"],
            "context": "The Helm chart requires secrets and persistent storage for model cache.",
            "qa_type": None,
            "instr_type": "procedural",
        },
        {
            "passage_id": "p2",
            "source_url": "https://docs.nvidia.com/nim/holdout.html",
            "product_family": "NIM",
            "stage": "1a",
            "question": training[2]["prompt"],
            "answer": training[2]["completion"],
            "context": "This row should be excluded because it is in validation.",
        },
    ]
    _write_jsonl(dataset_dir / "training.jsonl", training)
    _write_jsonl(dataset_dir / "stage2_eval.jsonl", stage2)
    _write_jsonl(dataset_dir / "validation.jsonl", [training[2]])
    _write_jsonl(dataset_dir / "test_set.jsonl", [])
    _write_jsonl(dataset_dir / "test_set_with_context.jsonl", [])


def test_load_candidate_rows_uses_stage2_context_and_excludes_holdouts(tmp_path):
    config = _config(tmp_path)
    _write_dataset(config.dataset_dir)

    candidates, metrics = load_candidate_rows(config)

    assert metrics["eligible_training_pairs"] == 2
    assert metrics["candidate_rows"] == 2
    assert {candidate["candidate_source"] for candidate in candidates} == {"stage2_eval"}
    assert {candidate["source_url"] for candidate in candidates} == {
        "https://docs.nvidia.com/nim/status.html",
        "https://docs.nvidia.com/nim/helm.html",
    }
    assert all("Holdout" not in candidate["prompt"] for candidate in candidates)


def test_load_candidate_rows_includes_training_rows_missing_from_stage2(tmp_path):
    config = _config(tmp_path)
    _write_dataset(config.dataset_dir)
    extra = {
        "prompt": "Which row is missing from stage2?",
        "completion": "This grounded training row still seeds generation.",
        "system": "system",
    }
    with (config.dataset_dir / "training.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(extra) + "\n")

    candidates, metrics = load_candidate_rows(config)

    assert metrics["eligible_training_pairs"] == 3
    assert metrics["candidate_rows"] == 3
    assert metrics["candidate_source_counts"] == {"stage2_eval": 2, "training": 1}
    fallback = [candidate for candidate in candidates if candidate["candidate_source"] == "training"][0]
    assert fallback["prompt"] == extra["prompt"]
    assert fallback["source_url"] is None


def test_run_prepare_writes_seed_csv_plan_and_observability(tmp_path):
    config = _config(tmp_path)
    _write_dataset(config.dataset_dir)

    result = run(config)

    assert result["seed_records"] == 2
    assert result["synthetic_pairs_requested"] == 6
    seed_requests = config.output_dir / "data_designer" / "kimi_seed_requests.jsonl"
    seed_csv = config.output_dir / "data_designer" / "seed_dataset.csv"
    submission_plan = config.output_dir / "data_designer" / "submission_plan.json"
    assert seed_requests.exists()
    assert seed_csv.exists()
    assert submission_plan.exists()

    seeds = [json.loads(line) for line in seed_requests.read_text().splitlines()]
    assert {seed["seed_author"] for seed in seeds} == {"heuristic"}
    assert all(seed["source_sample_ids"] for seed in seeds)
    assert all(seed["retrieved_chunks"] for seed in seeds)
    assert all(seed["kimi"]["prompt_sha256"].startswith("sha256:") for seed in seeds)
    assert all(seed["kimi"]["response_sha256"] is None for seed in seeds)

    with seed_csv.open(newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 2
    assert json.loads(csv_rows[0]["constraints_json"])
    assert json.loads(csv_rows[0]["retrieved_urls"])

    plan = json.loads(submission_plan.read_text())
    assert plan["data_designer"]["prompt_column"] == "qa_pairs_json"
    assert plan["data_designer"]["num_records"] == 2
    assert plan["metrics"]["synthetic_pairs_requested"] == 6
    assert (config.output_dir / "observability" / "data-designer-grounded-seeds" / "metrics.json").exists()


def test_kimi_mode_requires_explicit_external_permission(tmp_path):
    config = _config(tmp_path, mode="kimi", allow_external_kimi=False)
    _write_dataset(config.dataset_dir)

    with pytest.raises(RuntimeError, match="allow-external-kimi"):
        run(config)



def test_parse_json_object_strips_inline_think_tags():
    parsed = parse_json_object(
        '<think>{"not": "the answer"}</think>\n```json\n{"coverage_axis": "api"}\n```'
    )

    assert parsed == {"coverage_axis": "api"}
