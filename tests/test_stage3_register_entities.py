"""Tests for Evaluator target/config payload builders.

These tests intentionally cover the native NIM Proxy model-target path. The
legacy rag-oai-proxy/RAG target path should not be the default showcase path.
"""
import sys

import pytest

import scripts.eval.register_evaluator_entities as ree
from scripts.eval.register_evaluator_entities import (
    AdapterRow,
    build_49b_target,
    build_adapter_target,
    build_base_target,
    build_dataset_payload,
    build_pairwise_config,
    build_singleaxis_config,
)


# --- model targets -------------------------------------------------------

def test_build_adapter_target_points_at_nim_proxy():
    row = AdapterRow(
        name="lora-nim-llama-3.2-3b-r16",
        base_model="meta/llama-3.2-3b-instruct",
        job_id="cust-abc",
        collection="nim_curated",
    )
    target = build_adapter_target(row, proxy_url="http://nemo-nim-proxy:8000")

    endpoint = target["model"]["api_endpoint"]
    assert target["name"] == "lora-nim-llama-3.2-3b-r16"
    assert target["namespace"] == "default"
    assert target["type"] == "model"
    assert endpoint["url"] == "http://nemo-nim-proxy:8000/v1/chat/completions"
    assert endpoint["model_id"] == "lora-nim-llama-3.2-3b-r16"
    assert endpoint["format"] == "nim"


def test_build_base_target_uses_base_slug_as_target_name():
    target = build_base_target(
        base_model="meta/llama-3.2-3b-instruct",
        proxy_url="http://nemo-nim-proxy:8000",
    )

    endpoint = target["model"]["api_endpoint"]
    assert target["name"] == "llama-3.2-3b-instruct"
    assert target["type"] == "model"
    assert endpoint["url"] == "http://nemo-nim-proxy:8000/v1/chat/completions"
    assert endpoint["model_id"] == "llama-3.2-3b-instruct"
    assert endpoint["format"] == "nim"


def test_build_49b_target_is_model_target_not_rag_target():
    target = build_49b_target(proxy_url="http://nemo-nim-proxy:8000")

    endpoint = target["model"]["api_endpoint"]
    assert target["type"] == "model"
    assert target["name"] == "llama-3.3-nemotron-super-49b-v1.5"
    assert "rag" not in target
    assert endpoint["url"] == "http://nemo-nim-proxy:8000/v1/chat/completions"
    assert endpoint["model_id"] == "llama-3.3-nemotron-super-49b-v1.5"
    assert endpoint["format"] == "nim"


# --- dataset -------------------------------------------------------------

def test_build_dataset_payload_for_nim():
    payload = build_dataset_payload(
        "nim_curated",
        files_url="hf://datasets/default/stage3-nim-curated-test",
    )
    assert payload["name"] == "stage3-nim-curated-test"
    assert payload["namespace"] == "default"
    assert payload["format"] == "hf"
    assert payload["files_url"] == "hf://datasets/default/stage3-nim-curated-test"
    assert payload["hf_endpoint"] == "http://nemo-data-store:3000/v1/hf"


# --- configs -------------------------------------------------------------

def test_singleaxis_config_uses_ragas_metrics_and_external_judge_ref():
    cfg = build_singleaxis_config()
    assert cfg["name"] == "stage3-singleaxis-rubric"
    assert cfg["type"] == "custom"
    assert cfg["params"]["temperature"] == 0.0001
    assert cfg["params"]["max_tokens"] == 8192

    task = cfg["tasks"]["ragas_rubric"]
    assert task["type"] == "chat-completion"
    metrics = task["metrics"]
    assert set(metrics) == {"faithfulness", "response_relevancy", "answer_accuracy"}
    for metric in metrics.values():
        assert metric["params"]["judge"]["model"] == "default/llama-3.3-nemotron-super-49b-v1.5"
        assert "retrieved_contexts" in metric["params"]["input_template"]


def test_pairwise_config_has_position_swap_and_judge():
    cfg = build_pairwise_config()
    assert cfg["name"] == "stage3-pairwise-tournament"
    assert cfg["type"] == "custom"
    extra = cfg["params"]["extra"]
    assert extra["position_swap"] is True
    assert "A" in extra["pairwise_prompt"] and "B" in extra["pairwise_prompt"]
    assert extra["judge_model"] == "nvidia/llama-3.3-nemotron-super-49b-v1.5"


# --- AdapterRow round-trip ----------------------------------------------

def test_adapter_row_from_log_line():
    line = ("| lora-nim-llama-3.2-3b-r16     | cust-96dMd4ziS1inhYUG7Q8nBM     "
            "|     1.204  |   1.506  | ~12 min    |")
    row = AdapterRow.from_log_line(line)
    assert row.name == "lora-nim-llama-3.2-3b-r16"
    assert row.job_id == "cust-96dMd4ziS1inhYUG7Q8nBM"
    assert row.base_model == "meta/llama-3.2-3b-instruct"
    assert row.collection == "nim_curated"


def test_adapter_row_from_log_line_moe_nano():
    line = ("| lora-nemo-usvcs-nemotron-nano-30b-r16 | "
            "`<ARTIFACT_ROOT>/checkpoints/lora/lora-nemo-usvcs-nemotron-nano-30b-r16/` "
            "| 886 MB | This run (2026-05-27) |")
    row = AdapterRow.from_log_line(line)
    assert row.name == "lora-nemo-usvcs-nemotron-nano-30b-r16"
    assert row.job_id == "ties-merged"
    assert row.base_model == "nvidia/nemotron-3-nano-30b-a3b"
    assert row.collection == "nemo_usvcs_curated"


def test_adapter_row_from_log_line_skips_shard_rows():
    line = ("| lora-nemo-usvcs-nemotron-nano-30b-r16-shard-a  | "
            "cust-4dLpWrjfy2GUn14StTnnVY     |    0.488   |   1.042  |  ~87 min   |")
    with pytest.raises(ValueError, match="shard row not registrable"):
        AdapterRow.from_log_line(line)


def test_adapter_row_from_log_line_raises_on_malformed():
    with pytest.raises(ValueError, match="Cannot parse row"):
        AdapterRow.from_log_line("not a pipe-delimited row at all")


def test_adapter_row_from_log_line_raises_on_unknown_size():
    bad = "| lora-nim-llama-9.9-99b-r16 | cust-xyz | 1.0 | 1.0 | ~5 min |"
    with pytest.raises(ValueError, match="Unknown base size in name"):
        AdapterRow.from_log_line(bad)


def test_main_all_registers_targets_and_configs_via_nim_proxy(tmp_path, monkeypatch):
    log_path = tmp_path / "training_session.log"
    log_path.write_text(
        "| lora-nim-llama-3.2-3b-r16 | cust-abc | 1.0 | 1.0 | ~5 min |\n"
    )
    created_targets = []
    created_configs = []

    class DummyClient:
        def __init__(self, base_url, api_key=None):
            assert base_url == "http://evaluator.test"
            assert api_key == "secret-token"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def create_target(self, payload):
            created_targets.append(payload)
            return f"target-{len(created_targets)}"

        def create_config(self, payload):
            created_configs.append(payload)
            return f"config-{len(created_configs)}"

    monkeypatch.setattr(ree, "EvaluatorClient", DummyClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "register_evaluator_entities.py",
            "--all",
            "--evaluator-url",
            "http://evaluator.test",
            "--evaluator-api-key",
            "secret-token",
            "--proxy-url",
            "http://nim-proxy.test",
            "--log-path",
            str(log_path),
        ],
    )

    assert ree.main() == 0

    # One adapter target from the fixture log, four base targets,
    # and the 49B comparator.
    assert len(created_targets) == 6
    assert len(created_configs) == 2
    assert {cfg["name"] for cfg in created_configs} == {
        "stage3-singleaxis-rubric",
        "stage3-pairwise-tournament",
    }
    for target in created_targets:
        endpoint = target["model"]["api_endpoint"]
        assert endpoint["url"] == "http://nim-proxy.test/v1/chat/completions"
        assert endpoint["format"] == "nim"
