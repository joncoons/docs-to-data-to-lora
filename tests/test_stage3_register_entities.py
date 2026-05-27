"""Tests for entity payload builders (single-axis config, pairwise config,
target shapes, dataset shapes). Doesn't make real network calls."""
import pytest

from scripts.eval.register_evaluator_entities import (
    AdapterRow,
    build_adapter_target,
    build_base_target,
    build_dataset_payload,
    build_pairwise_config,
    build_rag_target,
    build_singleaxis_config,
)


# --- adapter target -------------------------------------------------------

def test_build_adapter_target_for_3b():
    row = AdapterRow(
        name="lora-nim-llama-3.2-3b-r16",
        base_model="meta/llama-3.2-3b-instruct",
        job_id="cust-abc",
        collection="nim_curated",
    )
    t = build_adapter_target(row, nim_url="http://nim-llama-3.2-3b:8000")
    assert t["name"] == "lora-nim-llama-3.2-3b-r16"
    assert t["namespace"] == "default"
    assert t["type"] == "model"
    assert t["model"]["api_endpoint"]["url"] == "http://nim-llama-3.2-3b:8000/v1/chat/completions"
    assert t["model"]["api_endpoint"]["model_id"] == "lora-nim-llama-3.2-3b-r16"


# --- base target ----------------------------------------------------------

def test_build_base_target_omits_adapter_model_id():
    t = build_base_target(base_model="meta/llama-3.2-3b-instruct",
                          nim_url="http://nim-llama-3.2-3b-base:8000")
    assert t["name"] == "base-llama-3.2-3b-instruct"
    assert t["type"] == "model"
    # base model identifier is the base slug, not an adapter dir name
    assert t["model"]["api_endpoint"]["model_id"] == "meta/llama-3.2-3b-instruct"


# --- RAG target -----------------------------------------------------------

def test_build_rag_target_scopes_to_collection():
    t = build_rag_target(
        collection="nim_curated",
        rag_url="http://rag-agent-toolkit:8000",
    )
    assert t["type"] == "rag"
    assert t["name"] == "rag-49b-nim-curated"
    # the spec's collection-scope expectation:
    assert t["rag"]["collection_name"] == "nim_curated"


# --- dataset --------------------------------------------------------------

def test_build_dataset_payload_for_nim():
    d = build_dataset_payload("nim_curated",
                              files_url="hf://datasets/default/stage3-nim-curated-test")
    assert d["name"] == "stage3-nim-curated-test"
    assert d["namespace"] == "default"
    assert d["format"] == "hf"
    assert d["files_url"] == "hf://datasets/default/stage3-nim-curated-test"


# --- configs --------------------------------------------------------------

def test_singleaxis_config_has_4_axis_rubric_and_judge():
    cfg = build_singleaxis_config()
    assert cfg["name"] == "stage3-singleaxis-rubric"
    assert cfg["type"] == "custom"
    rubric_text = cfg["params"]["extra"]["rubric_prompt"]
    for axis in ("Accuracy", "Completeness", "Faithfulness", "Clarity"):
        assert axis in rubric_text
    assert cfg["params"]["extra"]["judge_model"] == \
        "aws/anthropic/bedrock-claude-sonnet-4-6"
    # adapter inference temperature must be 0.0 — deterministic
    assert cfg["params"]["temperature"] == 0.0
    assert cfg["params"]["max_tokens"] == 600


def test_singleaxis_judge_prompt_includes_context_but_inference_does_not():
    cfg = build_singleaxis_config()
    extra = cfg["params"]["extra"]
    # judge prompt sees the source context (for grounding/faithfulness)
    assert "{context}" in extra["rubric_prompt"]
    # inference prompt receives only the question (no context leakage)
    assert "{context}" not in extra["inference_prompt"]
    assert "{question}" in extra["inference_prompt"]


def test_pairwise_config_has_position_swap_and_judge():
    cfg = build_pairwise_config()
    assert cfg["name"] == "stage3-pairwise-tournament"
    assert cfg["type"] == "custom"
    extra = cfg["params"]["extra"]
    assert extra["position_swap"] is True
    assert "A" in extra["pairwise_prompt"] and "B" in extra["pairwise_prompt"]
    assert extra["judge_model"] == "aws/anthropic/bedrock-claude-sonnet-4-6"


# --- AdapterRow round-trip ------------------------------------------------

def test_adapter_row_from_log_line():
    """Parse a dense-Llama row of evals/training_session.log into AdapterRow."""
    line = ("| lora-nim-llama-3.2-3b-r16     | cust-96dMd4ziS1inhYUG7Q8nBM     "
            "|     1.204  |   1.506  | ~12 min    |")
    row = AdapterRow.from_log_line(line)
    assert row.name == "lora-nim-llama-3.2-3b-r16"
    assert row.job_id == "cust-96dMd4ziS1inhYUG7Q8nBM"
    assert row.base_model == "meta/llama-3.2-3b-instruct"
    assert row.collection == "nim_curated"


def test_adapter_row_from_log_line_moe_nano():
    """Parse a MoE merged-adapter row (Final r=16 MoE adapter inventory)."""
    line = ("| lora-nemo-usvcs-nemotron-nano-30b-r16 | "
            "`/mnt/nvme2/peft/checkpoints/lora/lora-nemo-usvcs-nemotron-nano-30b-r16/` "
            "| 886 MB | This run (2026-05-27) |")
    row = AdapterRow.from_log_line(line)
    assert row.name == "lora-nemo-usvcs-nemotron-nano-30b-r16"
    assert row.job_id == "ties-merged"
    assert row.base_model == "nvidia/nemotron-3-nano-30b-a3b"
    assert row.collection == "nemo_usvcs_curated"


def test_adapter_row_from_log_line_skips_shard_rows():
    """Shard rows (-shard-a / -shard-b) are not registrable eval targets."""
    line = ("| lora-nemo-usvcs-nemotron-nano-30b-r16-shard-a  | "
            "cust-4dLpWrjfy2GUn14StTnnVY     |    0.488   |   1.042  |  ~87 min   |")
    with pytest.raises(ValueError, match="shard row not registrable"):
        AdapterRow.from_log_line(line)


def test_adapter_row_from_log_line_raises_on_malformed():
    """Lines without pipe delimiters or missing cells must raise ValueError."""
    with pytest.raises(ValueError, match="Cannot parse row"):
        AdapterRow.from_log_line("not a pipe-delimited row at all")


def test_adapter_row_from_log_line_raises_on_unknown_size():
    """An adapter name whose size suffix isn't 1B/3B/8B/Nano must raise ValueError."""
    bad = "| lora-nim-llama-9.9-99b-r16 | cust-xyz | 1.0 | 1.0 | ~5 min |"
    with pytest.raises(ValueError, match="Unknown base size in name"):
        AdapterRow.from_log_line(bad)
