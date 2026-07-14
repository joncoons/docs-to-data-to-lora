"""Tests for direct Kimi single-axis judging over saved completions."""

import json
from pathlib import Path

from scripts.eval.run_direct_kimi_singleaxis import (
    JudgeConfig,
    build_judge_user,
    coerce_scores,
    completion_relative_dir,
    load_error_attempts,
    output_dir_for,
    parse_json_object,
    run_one_responses_file,
    summarize_scores,
    write_deduped_scores,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_judge_user_splits_context_and_uses_response():
    user = build_judge_user({
        "prompt": "Context:\nFact one.\n\nQuestion: What is fact one?",
        "reference_completion": "Fact one is true.",
        "response": "Fact one is true.",
    })

    assert "Question:\nWhat is fact one?" in user
    assert "Source context:\nContext:\nFact one." in user
    assert "Reference answer:\nFact one is true." in user
    assert "Model answer:\nFact one is true." in user


def test_parse_json_object_strips_think_tags_and_fences():
    parsed = parse_json_object(
        '<think>private</think>\n```json\n{"accuracy": 5, "faithfulness": 4}\n```'
    )

    assert parsed["accuracy"] == 5
    assert parsed["faithfulness"] == 4


def test_coerce_scores_clamps_to_rubric_bounds():
    scores = coerce_scores({
        "accuracy": 7,
        "completeness": 0,
        "faithfulness": 4,
        "clarity": "3",
        "reason": "ok",
    })

    assert scores == {
        "accuracy": 5,
        "completeness": 1,
        "faithfulness": 4,
        "clarity": 3,
        "reason": "ok",
    }


def test_output_dir_preserves_completion_layout(tmp_path):
    completions = tmp_path / "completions"
    responses = (
        completions
        / "nim_curated"
        / "llama-3.2-3b"
        / "lora-nim"
        / "r16"
        / "20260529t142723z"
        / "responses.jsonl"
    )

    assert completion_relative_dir(responses, completions) == Path(
        "nim_curated/llama-3.2-3b/lora-nim/r16/20260529t142723z"
    )
    assert output_dir_for(
        responses,
        completions_root=completions,
        output_root=tmp_path / "singleaxis-kimi",
        eval_run_id="evalrun",
    ) == (
        tmp_path
        / "singleaxis-kimi"
        / "nim_curated"
        / "llama-3.2-3b"
        / "lora-nim"
        / "r16"
        / "20260529t142723z"
        / "evalrun"
    )


def test_summarize_scores_includes_score_means_and_token_totals(tmp_path):
    scores = tmp_path / "scores.jsonl"
    errors = tmp_path / "errors.jsonl"
    _write_jsonl(scores, [
        {
            "source_row_index": 1,
            "scores": {
                "accuracy": 5,
                "completeness": 4,
                "faithfulness": 5,
                "clarity": 4,
            },
            "target_token_counts": {
                "prompt_tokens": 10,
                "completion_tokens_raw": 5,
                "completion_tokens_cleaned_est": 5,
                "total_tokens_raw": 15,
            },
            "judge": {
                "token_counts": {
                    "prompt_tokens": 20,
                    "completion_tokens_raw": 6,
                    "total_tokens_raw": 26,
                }
            },
        },
        {
            "source_row_index": 2,
            "scores": {
                "accuracy": 3,
                "completeness": 2,
                "faithfulness": 1,
                "clarity": 5,
            },
            "target_token_counts": {
                "prompt_tokens": 10,
                "completion_tokens_raw": 10,
                "completion_tokens_cleaned_est": 10,
                "total_tokens_raw": 20,
            },
            "judge": {
                "token_counts": {
                    "prompt_tokens": 20,
                    "completion_tokens_raw": 4,
                    "total_tokens_raw": 24,
                }
            },
        },
    ])
    _write_jsonl(errors, [{"source_row_index": 3, "error": "bad"}])

    summary = summarize_scores(scores, errors)

    assert summary["rows_scored"] == 2
    assert summary["rows_failed"] == 1
    assert summary["score_means"]["mean_accuracy"] == 4.0
    assert summary["score_means"]["mean_faithfulness"] == 3.0
    assert summary["target_generation"]["total_tokens_raw"] == 35
    assert summary["judge_scoring"]["total_tokens_raw"] == 50
    assert summary["combined_total_tokens_raw"] == 85


def test_summarize_scores_dedupes_by_source_row_index(tmp_path):
    scores = tmp_path / "scores.jsonl"
    errors = tmp_path / "errors.jsonl"
    _write_jsonl(scores, [
        {
            "source_row_index": 1,
            "scores": {"accuracy": 1, "completeness": 1, "faithfulness": 1, "clarity": 1},
            "target_token_counts": {"total_tokens_raw": 10},
            "judge": {"token_counts": {"total_tokens_raw": 20}},
        },
        {
            "source_row_index": 1,
            "scores": {"accuracy": 5, "completeness": 5, "faithfulness": 5, "clarity": 5},
            "target_token_counts": {"total_tokens_raw": 11},
            "judge": {"token_counts": {"total_tokens_raw": 21}},
        },
        {
            "source_row_index": 2,
            "scores": {"accuracy": 3, "completeness": 3, "faithfulness": 3, "clarity": 3},
            "target_token_counts": {"total_tokens_raw": 12},
            "judge": {"token_counts": {"total_tokens_raw": 22}},
        },
    ])
    _write_jsonl(errors, [
        {"source_row_index": 1, "error": "recovered"},
        {"source_row_index": 4, "error": "still missing"},
    ])

    summary = summarize_scores(scores, errors)

    assert summary["rows_scored_raw"] == 3
    assert summary["rows_scored"] == 2
    assert summary["score_duplicate_rows"] == 1
    assert summary["rows_failed"] == 1
    assert summary["unresolved_source_row_indices"] == [4]
    assert summary["score_means"]["mean_accuracy"] == 4.0
    assert summary["combined_total_tokens_raw"] == 66


def test_write_deduped_scores_keeps_latest_score(tmp_path):
    raw = tmp_path / "scores.jsonl"
    deduped = tmp_path / "scores_deduped.jsonl"
    _write_jsonl(raw, [
        {"source_row_index": 2, "scores": {"accuracy": 2}},
        {"source_row_index": 1, "scores": {"accuracy": 1}},
        {"source_row_index": 2, "scores": {"accuracy": 5}},
    ])

    assert write_deduped_scores(raw, deduped) == 2
    rows = [json.loads(line) for line in deduped.read_text(encoding="utf-8").splitlines()]

    assert [row["source_row_index"] for row in rows] == [1, 2]
    assert rows[1]["scores"]["accuracy"] == 5


def test_resume_writes_missing_and_retry_exhausted_rows_without_judging(tmp_path):
    completions = tmp_path / "completions"
    responses = completions / "dataset" / "model" / "target" / "rank" / "run" / "responses.jsonl"
    _write_jsonl(responses, [
        {"source_row_index": 1, "prompt": "p1", "response": "r1"},
        {"source_row_index": 2, "prompt": "p2", "response": "r2"},
        {"source_row_index": 3, "prompt": "p3", "response": "r3"},
    ])
    out_dir = tmp_path / "singleaxis" / "dataset" / "model" / "target" / "rank" / "run" / "eval"
    _write_jsonl(out_dir / "scores.jsonl", [
        {"source_row_index": 1, "scores": {"accuracy": 5, "completeness": 5, "faithfulness": 5, "clarity": 5}},
    ])
    _write_jsonl(out_dir / "errors.jsonl", [
        {"source_row_index": 2, "max_attempts": 5, "error": "exhausted"},
        {"source_row_index": 3, "max_attempts": 5, "error": "exhausted"},
    ])
    config = JudgeConfig(
        judge_api_url="http://judge.invalid/v1",
        judge_model="judge",
        api_key="key",
        max_tokens=10,
        temperature=0.0,
        timeout_s=1.0,
        max_attempts=5,
        retry_backoff_s=0.0,
        retry_backoff_max_s=0.0,
        retry_jitter_s=0.0,
    )

    run_one_responses_file(
        responses,
        completions_root=completions,
        output_root=tmp_path / "singleaxis",
        eval_run_id="eval",
        judge_config=config,
        limit=None,
        resume=True,
        concurrency=1,
        log_every=0,
        max_total_attempts_per_row=5,
    )

    missing = [json.loads(line) for line in (out_dir / "missing_rows.jsonl").read_text().splitlines()]
    exhausted = [json.loads(line) for line in (out_dir / "retry_exhausted_rows.jsonl").read_text().splitlines()]
    rows_to_judge = (out_dir / "rows_to_judge.jsonl").read_text(encoding="utf-8")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    assert [row["source_row_index"] for row in missing] == [2, 3]
    assert [row["source_row_index"] for row in exhausted] == [2, 3]
    assert rows_to_judge == ""
    assert load_error_attempts(out_dir / "errors.jsonl") == {2: 5, 3: 5}
    assert manifest["resume"]["existing_scores"] == 1
    assert manifest["resume"]["missing_rows"] == 2
    assert manifest["resume"]["retry_exhausted_rows"] == 2

