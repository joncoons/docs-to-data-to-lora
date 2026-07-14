"""Tests for Data Designer gap-fill submission and result collection."""

import csv
import json

from scripts.pipeline.data_designer_gapfill import (
    DataDesignerConfig,
    build_submission_plan,
    extract_pairs,
    load_generated_records,
    normalize_generated_records,
    run,
    write_seed_csv,
)


def _request():
    return {
        "gap_id": "gap_nim",
        "gap_manifest_id": "gapmanifest_123",
        "dataset_version_id": "dsv_123",
        "recipe_name": "nim-gapfill",
        "num_records": 2,
        "pairs_needed": 7,
        "pairs_per_record": 5,
        "input": {
            "gap_id": "gap_nim",
            "product_family": "nim",
            "pairs_count": 5,
            "seed_styles": "- What is NIM?",
            "retrieved_chunks": "NIM exposes OpenAI-compatible endpoints.",
            "generation_brief": "Generate grounded NIM QA pairs.",
        },
        "retrieved_urls": ["https://docs.example.com/nim"],
        "seed_entailment_ids": ["ent_nim"],
        "seed_chunk_ids": ["chunk_nim"],
    }


def _config(tmp_path, mode="prepare", results_jsonl=None):
    return DataDesignerConfig(
        dataset_dir=tmp_path / "nim_curated",
        collection="nim_curated",
        observability_dir=tmp_path / "observability",
        mode=mode,
        data_designer_url="http://nemo-data-designer:8080",
        datastore_endpoint="http://nemo-data-store:3000/v1/hf",
        seed_repo_id="default/nim-gapfill-seeds",
        seed_filename="gapfill_requests.csv",
        model="nvidia/nemotron-3-super-120b-a12b",
        model_alias="gapfill_model",
        model_provider="system/nvidia-build",
        temperature=0.3,
        top_p=1.0,
        max_tokens=2048,
        sampling_strategy="ordered",
        wait=False,
        job_id="dd_job_123" if mode == "collect" else None,
        results_jsonl=results_jsonl,
    )


def _write_inputs(dataset_dir):
    (dataset_dir / "provenance").mkdir(parents=True)
    (dataset_dir / "data_designer").mkdir()
    (dataset_dir / "provenance" / "gap_manifest.json").write_text(json.dumps({
        "schema_version": "provenance.v1",
        "gap_manifest_id": "gapmanifest_123",
        "dataset_version_id": "dsv_123",
        "analysis": {
            "analyzer_name": "stage1_5",
            "analyzer_version": "test",
            "config_hash": "sha256:" + "a" * 64,
        },
        "gaps": [{"gap_id": "gap_nim"}],
    }))
    (dataset_dir / "data_designer" / "gapfill_requests.jsonl").write_text(
        json.dumps(_request()) + "\n"
    )


def test_write_seed_csv_flattens_gapfill_request(tmp_path):
    out = tmp_path / "seed.csv"

    write_seed_csv([_request()], out)

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["gap_id"] == "gap_nim"
    assert rows[0]["product_family"] == "nim"
    assert rows[0]["pairs_count"] == "5"
    assert json.loads(rows[0]["retrieved_urls"]) == ["https://docs.example.com/nim"]


def test_build_submission_plan_records_native_data_designer_settings(tmp_path):
    config = _config(tmp_path)
    seed_csv = tmp_path / "seed.csv"
    gap_manifest = {"gap_manifest_id": "gapmanifest_123", "dataset_version_id": "dsv_123"}

    plan = build_submission_plan(config, gap_manifest, [_request()], seed_csv)

    assert plan["gap_manifest_id"] == "gapmanifest_123"
    assert plan["seed_dataset"]["repo_id"] == "default/nim-gapfill-seeds"
    assert plan["data_designer"]["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert plan["data_designer"]["num_records"] == 2
    assert plan["metrics"]["pairs_requested"] == 7


def test_extract_pairs_accepts_data_designer_json_column():
    record = {
        "qa_pairs_json": json.dumps({
            "pairs": [{"question": "Q?", "answer": "A."}],
        })
    }

    assert extract_pairs(record) == [{"question": "Q?", "answer": "A."}]


def test_normalize_generated_records_writes_stage_rows_and_provenance_samples():
    records = [{
        "gap_id": "gap_nim",
        "qa_pairs_json": json.dumps({
            "pairs": [{"question": "What does NIM expose?", "answer": "Endpoints."}],
        }),
    }]

    rows, samples = normalize_generated_records(
        records,
        [_request()],
        collection="nim_curated",
        data_designer_job_id="dd_job_123",
    )

    assert rows[0].stage == "1.5"
    assert rows[0].source_systems == ["data_designer"]
    assert rows[0].source_chunk_ids == ["chunk_nim"]
    assert samples[0]["origin"] == "synthetic_gapfill"
    assert samples[0]["lineage"]["gap_id"] == "gap_nim"
    assert samples[0]["lineage"]["data_designer_job_id"] == "dd_job_123"
    assert samples[0]["lineage"]["entailment_ids"] == ["ent_nim"]


def test_run_prepare_writes_seed_plan_and_observability(tmp_path):
    config = _config(tmp_path)
    _write_inputs(config.dataset_dir)

    result = run(config)

    assert result["mode"] == "prepare"
    assert (config.dataset_dir / "data_designer" / "seed_dataset.csv").exists()
    assert (config.dataset_dir / "data_designer" / "submission_plan.json").exists()
    metrics = json.loads((config.observability_dir / "metrics.json").read_text())
    assert metrics["gapfill.gaps.requested"] == 1
    assert metrics["gapfill.samples.generated"] == 0


def test_run_collect_normalizes_results_and_writes_manifests(tmp_path):
    results_jsonl = tmp_path / "results.jsonl"
    results_jsonl.write_text(json.dumps({
        "gap_id": "gap_nim",
        "qa_pairs_json": json.dumps({
            "pairs": [
                {"question": "Q1?", "answer": "A1."},
                {"question": "Q2?", "answer": "A2."},
            ],
        }),
    }) + "\n")
    config = _config(tmp_path, mode="collect", results_jsonl=results_jsonl)
    _write_inputs(config.dataset_dir)

    result = run(config)

    assert result["generated_samples"] == 2
    assert (config.dataset_dir / "stage1_5_gapfill.jsonl").exists()
    assert (config.dataset_dir / "data_designer" / "generated_samples.jsonl").exists()
    assert (config.dataset_dir / "provenance" / "data_designer_samples.jsonl").exists()
    manifest = json.loads((config.dataset_dir / "data_designer" / "result_manifest.json").read_text())
    assert manifest["data_designer_job_id"] == "dd_job_123"
    assert manifest["generated_sample_count"] == 2
    metrics = json.loads((config.observability_dir / "metrics.json").read_text())
    assert metrics["gapfill.samples.generated"] == 2


def test_load_generated_records_reads_csv(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "dataset.csv").write_text('gap_id,question,answer\ngap_nim,Q?,A.\n')

    records = load_generated_records(None, out)

    assert records == [{"gap_id": "gap_nim", "question": "Q?", "answer": "A."}]



def test_extract_pairs_strips_inline_think_tags():
    record = {
        "qa_pairs_json": '<think>{"pairs": []}</think>{"pairs": [{"question": "<think>draft</think>Q?", "answer": "<think>draft</think>A."}]}'
    }

    assert extract_pairs(record) == [{"question": "Q?", "answer": "A."}]
