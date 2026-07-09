"""Tests for durable completion collection helpers."""

import json

from pathlib import Path

from scripts.eval.collect_completions import (
    build_run_dir,
    infer_dataset_slug,
    parse_model_descriptor,
    strip_think_tags,
    build_token_counts,
    parse_retry_status_codes,
    summarize_run_files,
)


def test_parse_dense_lora_descriptor_captures_rank_and_corpus():
    desc = parse_model_descriptor("lora-nemo-usvcs-llama-3.2-1b-r32")

    assert desc["target_type"] == "lora"
    assert desc["base_slug"] == "llama-3.2-1b"
    assert desc["target_slug"] == "lora-nemo-usvcs"
    assert desc["rank"] == 32
    assert desc["rank_slug"] == "r32"



def test_parse_retrained_nano_lora_descriptor_keeps_rank_and_variant():
    desc = parse_model_descriptor("lora-nemo-usvcs-nemotron-nano-30b-r16-retrain-20260530")

    assert desc["target_type"] == "lora"
    assert desc["base_slug"] == "nemotron-nano-30b"
    assert desc["target_slug"] == "lora-nemo-usvcs"
    assert desc["rank"] == 16
    assert desc["rank_slug"] == "r16"
    assert desc["adapter_variant"] == "retrain-20260530"

def test_parse_base_descriptor_uses_base_rank_slug():
    desc = parse_model_descriptor("llama-3.2-1b-instruct")

    assert desc["target_type"] == "base"
    assert desc["base_slug"] == "llama-3.2-1b"
    assert desc["target_slug"] == "base"
    assert desc["rank"] is None
    assert desc["rank_slug"] == "base"


def test_run_dir_layout_includes_rank_component():
    desc = parse_model_descriptor("lora-nim-llama-3.2-1b-r16")

    run_dir = build_run_dir(
        Path("/outputs"),
        dataset_slug="nim_curated",
        descriptor=desc,
        run_id="20260529T140000Z",
    )

    assert run_dir == Path(
        "/outputs/nim_curated/llama-3.2-1b/lora-nim/r16/20260529t140000z"
    )


def test_dataset_slug_prefers_parent_for_context_baked_testset():
    assert infer_dataset_slug(Path("/mnt/nvme2/peft/datasets/v2/nim_curated/test_set_with_context.jsonl")) == "nim_curated"


def test_strip_think_tags_matches_reasoning_filter():
    assert strip_think_tags("<think>hidden</think>\nFinal") == "Final"


def test_token_counts_do_not_infer_think_overhead_without_stripping():
    counts = build_token_counts(
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "Final answer only",
        "Final answer only",
    )

    assert counts["completion_tokens_raw"] == 20
    assert counts["completion_tokens_cleaned_est"] == 20
    assert counts["think_tokens_est"] == 0
    assert counts["reasoning_overhead_ratio_est"] == 0


def test_parse_retry_status_codes_accepts_comma_list():
    assert parse_retry_status_codes("502, 503,504") == {502, 503, 504}


def test_summarize_run_files_treats_success_as_resolving_prior_error(tmp_path):
    responses = tmp_path / "responses.jsonl"
    errors = tmp_path / "errors.jsonl"
    errors.write_text(
        json.dumps({"source_row_index": 7, "error_type": "HTTPStatusError"}) + "\n",
        encoding="utf-8",
    )
    responses.write_text(
        json.dumps(
            {
                "source_row_index": 7,
                "token_counts": {
                    "prompt_tokens": 3,
                    "completion_tokens_raw": 5,
                    "completion_tokens_cleaned_est": 5,
                    "think_tokens_est": 0,
                    "total_tokens_raw": 8,
                    "raw_response_chars": 20,
                    "cleaned_response_chars": 20,
                    "think_chars_stripped": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run_files(responses, errors)

    assert summary["rows_completed"] == 1
    assert summary["rows_failed"] == 0
    assert summary["total_tokens_raw"] == 8

