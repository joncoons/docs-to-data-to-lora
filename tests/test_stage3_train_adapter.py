"""Tests for train_adapter CLI submission logic."""
from unittest.mock import MagicMock

from scripts.stage3.models import AdapterSpec
from scripts.stage3.train_adapter import (
    build_customizer_config, submit_adapter_job,
)


def test_build_customizer_config_includes_all_required_fields():
    spec = AdapterSpec(
        adapter_name="lora-nim-llama3.2-3b-r16",
        collection="nim_curated",
        base_model="meta/llama-3.2-3b-instruct",
        rank=16,
        alpha=32,
    )
    cfg = build_customizer_config(
        spec, base_template="meta/llama-3.2-3b-instruct@v1.0.0+80GB",
        adapter_train_path="/data/adapter_train.jsonl",
        adapter_val_path="/data/adapter_val.jsonl",
        output_path="/checkpoints/lora-llama-3.2-3b/lora-nim-llama3.2-3b-r16",
    )
    assert cfg["config"] == "meta/llama-3.2-3b-instruct@v1.0.0+80GB"
    assert cfg["hyperparameters"]["lora"]["adapter_dim"] == 16
    assert cfg["hyperparameters"]["lora"]["alpha"] == 32
    assert cfg["hyperparameters"]["epochs"] == 2
    assert cfg["hyperparameters"]["sequence_packing_enabled"] is False
    assert cfg["dataset"]["train_file"] == "/data/adapter_train.jsonl"
    assert cfg["output_model_path"].endswith("lora-nim-llama3.2-3b-r16")


def test_build_customizer_config_attention_only_target_modules():
    spec = AdapterSpec(
        adapter_name="x", collection="nim_curated",
        base_model="meta/llama-3.2-3b-instruct", rank=32, alpha=64,
    )
    cfg = build_customizer_config(spec, "meta/llama-3.2-3b-instruct@v1.0.0+80GB",
                                   "/a", "/b", "/c")
    tm = cfg["hyperparameters"]["lora"]["target_modules"]
    assert set(tm) == {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_submit_adapter_job_calls_client():
    spec = AdapterSpec(
        adapter_name="lora-x", collection="nim_curated",
        base_model="meta/llama-3.2-3b-instruct", rank=16, alpha=32,
    )
    fake_client = MagicMock()
    fake_client.submit_job.return_value = "cust-abc"
    job_id = submit_adapter_job(
        spec, "meta/llama-3.2-3b-instruct@v1.0.0+80GB",
        adapter_train_path="/a", adapter_val_path="/b", output_path="/c",
        client=fake_client,
    )
    assert job_id == "cust-abc"
    fake_client.submit_job.assert_called_once()
