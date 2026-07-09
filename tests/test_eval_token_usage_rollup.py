import json
from pathlib import Path

from scripts.eval.summarize_token_usage import (
    find_summaries,
    summarize_one,
    aggregate,
    SummaryRef,
    write_csv,
)


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_singleaxis_token_rollup(tmp_path: Path) -> None:
    eval_root = tmp_path / "evals"
    summary = eval_root / (
        "singleaxis-kimi/dataset-a/model-a/lora-a/r16/completion-run/eval-run/summary.json"
    )
    write_summary(
        summary,
        {
            "rows_scored": 2,
            "rows_failed": 1,
            "target_generation": {
                "prompt_tokens": 10,
                "completion_tokens_raw": 20,
                "completion_tokens_cleaned_est": 12,
                "think_tokens_est": 8,
                "total_tokens_raw": 30,
            },
            "judge_scoring": {
                "prompt_tokens": 100,
                "completion_tokens_raw": 200,
                "total_tokens_raw": 300,
            },
            "combined_total_tokens_raw": 330,
        },
    )

    row = summarize_one(eval_root, SummaryRef("singleaxis-kimi", summary))

    assert row["eval_run_id"] == "eval-run"
    assert row["dataset_slug"] == "dataset-a"
    assert row["target_slug"] == "lora-a"
    assert row["rank_slug"] == "r16"
    assert row["rows_scored_or_compared"] == 2
    assert row["target_generation"]["think_tokens_est"] == 8
    assert row["judge_scoring"]["total_tokens_raw"] == 300
    assert row["avg_judge_tokens_per_row"] == 150


def test_pairwise_token_rollup_uses_left_right_target_totals(tmp_path: Path) -> None:
    eval_root = tmp_path / "evals"
    summary = eval_root / "pairwise-kimi/dataset-a/model-a-vs-model-b/eval-run/summary.json"
    write_summary(
        summary,
        {
            "rows_compared": 3,
            "rows_failed": 0,
            "target_generation": {
                "left": {"total_tokens_raw": 30, "prompt_tokens": 10},
                "right": {"total_tokens_raw": 45, "prompt_tokens": 15},
                "combined_total_tokens_raw": 75,
            },
            "judge_scoring": {
                "prompt_tokens": 300,
                "completion_tokens_raw": 150,
                "total_tokens_raw": 450,
            },
        },
    )

    row = summarize_one(eval_root, SummaryRef("pairwise-kimi", summary))

    assert row["eval_run_id"] == "eval-run"
    assert row["pair_slug"] == "dataset-a/model-a-vs-model-b"
    assert row["rows_scored_or_compared"] == 3
    assert row["target_generation"]["prompt_tokens"] == 25
    assert row["target_generation"]["total_tokens_raw"] == 75
    assert row["combined_total_tokens_raw"] == 525


def test_find_summaries_filters_by_run_id(tmp_path: Path) -> None:
    eval_root = tmp_path / "evals"
    included = eval_root / "singleaxis-kimi/d/m/t/r/c/run-a/summary.json"
    excluded = eval_root / "singleaxis-kimi/d/m/t/r/c/run-b/summary.json"
    write_summary(included, {"rows_scored": 0})
    write_summary(excluded, {"rows_scored": 0})

    refs = find_summaries(eval_root, ["singleaxis-kimi"], "run-a")

    assert [ref.path for ref in refs] == [included]


def test_aggregate_and_csv_output(tmp_path: Path) -> None:
    rows = [
        {
            "eval_type": "singleaxis-kimi",
            "eval_run_id": "run-a",
            "dataset_slug": "d",
            "base_slug": "m",
            "target_slug": "base",
            "rank_slug": "base",
            "completion_run_id": "c",
            "relative_dir": "singleaxis-kimi/d/m/base/base/c/run-a",
            "rows_scored_or_compared": 2,
            "rows_failed": 1,
            "target_generation": {
                "prompt_tokens": 1,
                "completion_tokens_raw": 2,
                "completion_tokens_cleaned_est": 0,
                "think_tokens_est": 0,
                "total_tokens_raw": 3,
            },
            "judge_scoring": {
                "prompt_tokens": 10,
                "completion_tokens_raw": 20,
                "completion_tokens_cleaned_est": 0,
                "think_tokens_est": 0,
                "total_tokens_raw": 30,
            },
            "combined_total_tokens_raw": 33,
            "avg_judge_tokens_per_row": 15,
            "avg_combined_tokens_per_row": 16.5,
        }
    ]

    result = aggregate(rows)
    assert result["totals"]["rows_scored_or_compared"] == 2
    assert result["totals"]["judge_scoring"]["total_tokens_raw"] == 30
    assert result["totals"]["avg_combined_tokens_per_row"] == 16.5

    out = tmp_path / "tokens.csv"
    write_csv(out, rows)
    text = out.read_text(encoding="utf-8")
    assert "judge_total_tokens_raw" in text
    assert "singleaxis-kimi" in text
