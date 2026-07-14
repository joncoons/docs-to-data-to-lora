"""Tests for grounded Data Designer result collection."""

import json

from scripts.pipeline.collect_data_designer_grounded_results import (
    CollectConfig,
    extract_pairs,
    run,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_extract_pairs_accepts_nested_and_dotted_pairs():
    assert extract_pairs({
        "qa_pairs_json": {"pairs": [{"question": "Q?", "answer": "A."}]},
    }) == [{"question": "Q?", "answer": "A."}]
    assert extract_pairs({
        "qa_pairs_json.pairs": json.dumps([
            {"question": "<think>x</think>Q2?", "answer": "<think>y</think>A2."}
        ]),
    }) == [{"question": "Q2?", "answer": "A2."}]


def test_run_collects_grounded_results_and_rejects_duplicates(tmp_path):
    source = tmp_path / "source"
    experiment = tmp_path / "experiment"
    results = tmp_path / "results.jsonl"
    _write_jsonl(source / "training.jsonl", [
        {"prompt": "Existing?", "completion": "Already grounded."},
    ])
    _write_jsonl(source / "validation.jsonl", [
        {"prompt": "Holdout?", "completion": "Do not train."},
    ])
    _write_jsonl(experiment / "data_designer" / "kimi_seed_requests.jsonl", [
        {
            "seed_id": "seed_1",
            "gap_id": "seed_1",
            "collection": "nim_curated",
            "source_dataset": "nim_curated",
            "source_split": "training",
            "source_row_index": 7,
            "source_sample_id": "src_7",
            "source_url": "https://docs.example.com/nim",
            "passage_id": "p7",
            "product_family": "NIM",
            "retrieved_urls": ["https://docs.example.com/nim"],
            "seed_author": "kimi",
            "coverage_axis": "deployment",
            "pairs_count": 3,
        },
        {
            "seed_id": "seed_omitted",
            "gap_id": "seed_omitted",
            "source_dataset": "nim_curated",
            "source_split": "training",
        },
    ])
    (experiment / "data_designer" / "job_response.json").write_text(json.dumps({"id": "job_123"}))
    (experiment / "data_designer" / "job_request.json").write_text(json.dumps({
        "spec": {
            "config": {
                "model_configs": [{"provider": "kimi-k2", "model": "kimi-k2-6"}],
            }
        }
    }))
    _write_jsonl(results, [
        {
            "seed_id": "seed_1",
            "qa_pairs_json": json.dumps({
                "pairs": [
                    {"question": "Existing?", "answer": "Already grounded."},
                    {"question": "<think>draft</think>New?", "answer": "<think>draft</think>Answer."},
                    {"question": "New?", "answer": "Answer."},
                ]
            }),
            "qa_pairs_json__reasoning_trace": "hidden",
        }
    ])

    result = run(CollectConfig(
        source_dataset_dir=source,
        experiment_dir=experiment,
        collection="nim_curated",
        job_id=None,
        results_jsonl=results,
        results_dir=None,
        model_provider=None,
        model=None,
        system_prompt="system",
    ))

    assert result["raw_synthetic_pair_count"] == 3
    assert result["accepted_synthetic_pair_count"] == 1
    assert result["rejected_synthetic_pair_count"] == 2
    assert result["rejection_counts"] == {
        "duplicate_with_grounded_or_eval_dataset": 1,
        "duplicate_with_synthetic_dataset": 1,
    }
    assert result["seed_records_omitted"] == 1
    accepted = [
        json.loads(line)
        for line in (experiment / "data_designer" / "accepted_samples.jsonl").read_text().splitlines()
    ]
    assert accepted[0]["prompt"] == "New?"
    assert accepted[0]["completion"] == "Answer."
    assert accepted[0]["metadata"]["source"]["source_row_index"] == 7
    assert (experiment / "provenance" / "synthetic_samples.jsonl").exists()
    assert (experiment / "observability" / "data-designer-synthetic-generation" / "metrics.json").exists()
