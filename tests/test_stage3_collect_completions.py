"""Tests for durable completion collection helpers."""

import json

from pathlib import Path

from scripts.eval.collect_completions import (
    _parse_chat_completion_response,
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


def test_parse_hosted_namespaced_base_descriptor_uses_last_path_component():
    desc = parse_model_descriptor("nvidia/meta/llama-3.3-70b-instruct")

    assert desc["target_type"] == "base"
    assert desc["base_slug"] == "llama-3.3-70b"
    assert desc["target_slug"] == "base"
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
    assert infer_dataset_slug(Path("<DATASET_ROOT>/nim_curated/test_set_with_context.jsonl")) == "nim_curated"


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

class _FakeResponse:
    def __init__(self, content_type: str, text: str = "", payload=None):
        self.headers = {"content-type": content_type}
        self.text = text
        self._payload = payload

    def json(self):
        return self._payload


def test_parse_chat_completion_response_keeps_standard_json_body():
    payload = {
        "choices": [{"message": {"content": "standard answer"}}],
        "usage": {"total_tokens": 12},
    }

    parsed = _parse_chat_completion_response(
        _FakeResponse("application/json", payload=payload)
    )

    assert parsed is payload


def test_parse_chat_completion_response_merges_sse_chunks_and_metadata():
    response = _FakeResponse(
        "text/event-stream",
        "\n".join(
            [
                'data: {"id":"chat-1","model":"rag-model","created":123,"choices":[{"delta":{"content":"Hello "}}]}',
                'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}],"citations":{"results":[{"id":"doc-1"}]},"metrics":{"retrieval_ms":42},"usage":{"completion_tokens":2,"total_tokens":10}}',
                "data: [DONE]",
            ]
        ),
    )

    parsed = _parse_chat_completion_response(response)

    assert parsed["choices"][0]["message"]["content"] == "Hello world"
    assert parsed["choices"][0]["finish_reason"] == "stop"
    assert parsed["usage"]["total_tokens"] == 10
    assert parsed["citations"]["results"][0]["id"] == "doc-1"
    assert parsed["metrics"]["retrieval_ms"] == 42
    assert parsed["stream_chunks"] == 2

