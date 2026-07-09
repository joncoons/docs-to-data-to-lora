"""Tests for the live RAGAS smoke helper."""

from scripts.eval.run_live_ragas_smoke import (
    REASONING_GENERATION_BUDGET,
    build_judge_model,
    build_live_payload,
    build_token_counts,
    redact_secrets,
    split_context_baked_prompt,
    strip_think_tags,
    summarize_token_roi,
)


def test_strip_think_tags_balanced_block():
    raw = "<think>private reasoning</think>\n\nFinal answer."
    assert strip_think_tags(raw) == "Final answer."


def test_strip_think_tags_open_tail():
    assert strip_think_tags("<think>private reasoning only") == ""


def test_strip_think_tags_prelude_without_opening_tag():
    raw = "private reasoning accidentally leaked</think>\n\nFinal answer."
    assert strip_think_tags(raw) == "Final answer."


def test_split_context_baked_prompt_extracts_question():
    prompt = "Context:\nFact one.\n\nQuestion: What is fact one?"
    parts = split_context_baked_prompt(prompt)

    assert parts["context"] == "Context:\nFact one."
    assert parts["question"] == "What is fact one?"


def test_build_token_counts_keeps_raw_and_cleaned_roi_fields():
    counts = build_token_counts(
        {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110},
        raw_text="<think>long private reasoning</think>\n\nFinal answer.",
        cleaned_text="Final answer.",
    )

    assert counts["prompt_tokens"] == 10
    assert counts["completion_tokens_raw"] == 100
    assert counts["total_tokens_raw"] == 110
    assert counts["completion_tokens_cleaned_est"] > 0
    assert counts["think_tokens_est"] < 100


def test_build_judge_model_uses_8192_generation_budget_and_kimi_key():
    judge = build_judge_model(
        judge_api_url="https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/kimi-k2-6/v1",
        model_id="kimi-k2-6",
        api_key="secret-key",
    )

    assert judge["api_endpoint"]["url"] == (
        "https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/kimi-k2-6/v1/chat/completions"
    )
    assert judge["api_endpoint"]["model_id"] == "kimi-k2-6"
    assert judge["api_endpoint"]["api_key"] == "secret-key"
    assert judge["api_endpoint"]["format"] == "openai"
    assert judge["prompt"]["inference_params"]["max_tokens"] == REASONING_GENERATION_BUDGET
    assert judge["prompt"]["inference_params"]["max_retries"] == 3


def test_redact_secrets_masks_judge_api_key():
    payload = {"config": {"judge": {"api_endpoint": {"api_key": "secret-key"}}}}
    assert redact_secrets(payload)["config"]["judge"]["api_endpoint"]["api_key"] == "<redacted>"


def test_build_live_payload_uses_rows_target_and_data_task():
    rows = [{"prompt": "Context\nQuestion?", "completion": "Answer", "response": "Answer"}]
    judge = build_judge_model(
        judge_api_url="https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/kimi-k2-6/v1",
        model_id="kimi-k2-6",
        api_key="secret-key",
    )

    payload = build_live_payload(
        rows=rows,
        judge_model=judge,
        metric_types=["faithfulness", "answer_accuracy"],
    )

    assert payload["target"]["type"] == "rows"
    assert payload["target"]["rows"] == rows
    task = payload["config"]["tasks"]["ragas_data_rubric"]
    assert task["type"] == "data"
    assert set(task["metrics"]) == {"faithfulness", "answer_accuracy"}
    template = task["metrics"]["faithfulness"]["params"]["input_template"]
    assert "item.response" in template
    assert "item.question" in template
    assert "item.context" in template


def test_summarize_token_roi_includes_target_and_judge_counts():
    target_rows = [
        {
            "target_token_counts": {
                "prompt_tokens": 10,
                "completion_tokens_raw": 100,
                "completion_tokens_cleaned_est": 25,
                "think_tokens_est": 75,
                "total_tokens_raw": 110,
            }
        }
    ]
    live_result = {
        "logs": {
            "ragas_data_rubric": [
                {
                    "requests": [
                        {
                            "response": {
                                "usage_metadata": {
                                    "input_tokens": 30,
                                    "output_tokens": 40,
                                    "total_tokens": 70,
                                }
                            }
                        }
                    ]
                }
            ]
        }
    }

    summary = summarize_token_roi(target_rows, live_result)

    assert summary["target_generation"]["think_tokens_est"] == 75
    assert summary["judge_scoring"]["total_tokens_raw"] == 70
    assert summary["combined_total_tokens_raw"] == 180
    assert summary["target_reasoning_overhead_ratio_est"] == 0.75
