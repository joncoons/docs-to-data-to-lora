"""Tests for train_adapter CLI submission logic (Customizer 25.12 wire schema)."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

from scripts.stage3.models import AdapterSpec
from scripts.stage3.train_adapter import (
    _DATASET_FOR_COLLECTION,
    _TEMPLATE_FOR_BASE,
    build_customizer_config,
    submit_adapter_job,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nim_curated_spec(rank: int = 16) -> AdapterSpec:
    return AdapterSpec(
        adapter_name=f"lora-nim-llama-3.2-3b-r{rank}",
        collection="nim_curated",
        base_model="meta/llama-3.2-3b-instruct",
        rank=rank,
        alpha=2 * rank,
    )


def _nim_curated_config(rank: int = 16) -> dict:
    spec = _nim_curated_spec(rank)
    return build_customizer_config(
        spec,
        base_template=_TEMPLATE_FOR_BASE["meta/llama-3.2-3b-instruct"],
        dataset_entity=_DATASET_FOR_COLLECTION["nim_curated"],
        output_model_entity=f"default/lora-nim-llama-3.2-3b-r{rank}",
        description=f"Stage 3 — nim × llama-3.2-3b LoRA r{rank}",
    )


# ---------------------------------------------------------------------------
# Test 1: build_customizer_config produces the correct wire shape
# ---------------------------------------------------------------------------

def test_build_config_top_level_shape():
    """Top-level keys match Customizer 25.12 schema; legacy fields absent."""
    cfg = _nim_curated_config()

    # Required top-level fields
    assert cfg["config"] == "meta/llama-3.2-3b-instruct@v1.0.0+80GB"
    assert cfg["dataset"] == "default/stage3-nim-curated"
    assert cfg["output_model"] == "default/lora-nim-llama-3.2-3b-r16"
    assert isinstance(cfg["description"], str) and cfg["description"]

    # Fields that must NOT exist
    for forbidden in ("name", "output_model_path", "output_format"):
        assert forbidden not in cfg, f"Forbidden top-level key present: {forbidden!r}"


def test_build_config_hyperparameters_shape():
    """Hyperparameter block matches canonical schema; legacy fields absent."""
    hp = _nim_curated_config()["hyperparameters"]

    # Required fields with exact types / values
    assert hp["finetuning_type"] == "lora"
    assert hp["training_type"] == "sft"
    assert hp["epochs"] == 2
    assert hp["learning_rate"] == 1.0e-4
    assert hp["batch_size"] == 16
    assert hp["warmup_steps"] == 30
    assert hp["seed"] == 42
    assert hp["max_steps"] == -1
    assert hp["optimizer"] == "adamw_with_cosine_annealing"
    assert hp["adam_beta1"] == 0.9
    assert hp["adam_beta2"] == 0.99
    assert hp["log_every_n_steps"] == 10
    assert hp["sequence_packing_enabled"] is False

    # LoRA sub-block
    lora = hp["lora"]
    assert lora["adapter_dim"] == 16
    assert lora["alpha"] == 32
    assert lora["adapter_dropout"] is None
    assert lora["target_modules"] is None

    # Fields that must NOT exist
    for forbidden in (
        "precision",
        "warmup_ratio",
        "micro_batch_size",
        "global_batch_size",
        "val_check_interval",
        "save_top_k",
        "checkpoint_metric",
        "checkpoint_mode",
        "early_stopping",
    ):
        assert forbidden not in hp, f"Forbidden hyperparameter key present: {forbidden!r}"

    lora_forbidden = ("dropout",)
    for forbidden in lora_forbidden:
        assert forbidden not in lora, f"Forbidden lora key present: {forbidden!r}"


def test_build_config_dataset_and_output_model_are_strings():
    """dataset and output_model must be entity-ref strings, not dicts or paths."""
    cfg = _nim_curated_config()
    assert isinstance(cfg["dataset"], str)
    assert isinstance(cfg["output_model"], str)
    # No path separators (should be entity refs like "default/...")
    assert cfg["dataset"].startswith("default/")
    assert cfg["output_model"].startswith("default/")


def test_build_config_rank32_doubles_alpha_and_adapter_dim():
    """rank=32 correctly sets adapter_dim=32, alpha=64."""
    cfg = _nim_curated_config(rank=32)
    lora = cfg["hyperparameters"]["lora"]
    assert lora["adapter_dim"] == 32
    assert lora["alpha"] == 64


def test_template_map_includes_all_three_bases():
    """_TEMPLATE_FOR_BASE covers 1B (low-param baseline), 3B, and 8B."""
    assert _TEMPLATE_FOR_BASE["meta/llama-3.2-1b-instruct"] == \
        "meta/llama-3.2-1b-instruct@v1.0.0+80GB"
    assert _TEMPLATE_FOR_BASE["meta/llama-3.2-3b-instruct"] == \
        "meta/llama-3.2-3b-instruct@v1.0.0+80GB"
    assert _TEMPLATE_FOR_BASE["meta/llama-3.1-8b-instruct"] == \
        "meta/llama-3.1-8b-instruct@v1.0.0+80GB"


def test_build_config_for_1b_base():
    """1B base resolves to its template and the wire schema stays consistent."""
    spec = AdapterSpec(
        adapter_name="lora-nim-llama-3.2-1b-r16",
        collection="nim_curated",
        base_model="meta/llama-3.2-1b-instruct",
        rank=16,
        alpha=32,
    )
    cfg = build_customizer_config(
        spec,
        base_template=_TEMPLATE_FOR_BASE["meta/llama-3.2-1b-instruct"],
        dataset_entity=_DATASET_FOR_COLLECTION["nim_curated"],
        output_model_entity="default/lora-nim-llama-3.2-1b-r16",
        description="test",
    )
    assert cfg["config"] == "meta/llama-3.2-1b-instruct@v1.0.0+80GB"
    assert cfg["hyperparameters"]["lora"]["adapter_dim"] == 16
    assert cfg["hyperparameters"]["lora"]["alpha"] == 32


def test_dataset_entity_for_nemo_usvcs():
    """nemo_usvcs_curated collection maps to its own dataset entity."""
    spec = AdapterSpec(
        adapter_name="lora-nemo-usvcs-llama-3.2-3b-r16",
        collection="nemo_usvcs_curated",
        base_model="meta/llama-3.2-3b-instruct",
        rank=16,
        alpha=32,
    )
    cfg = build_customizer_config(
        spec,
        base_template=_TEMPLATE_FOR_BASE["meta/llama-3.2-3b-instruct"],
        dataset_entity=_DATASET_FOR_COLLECTION["nemo_usvcs_curated"],
        output_model_entity="default/lora-nemo-usvcs-llama-3.2-3b-r16",
        description="test",
    )
    assert cfg["dataset"] == "default/stage3-nemo-usvcs-curated"


# ---------------------------------------------------------------------------
# Test 2: submit_adapter_job calls client and returns job_id
# ---------------------------------------------------------------------------

def test_submit_adapter_job_calls_client_and_returns_job_id():
    """submit_adapter_job passes the built config to client.submit_job and forwards job_id."""
    spec = _nim_curated_spec()
    fake_client = MagicMock()
    fake_client.submit_job.return_value = "cust-abc123"

    job_id = submit_adapter_job(
        spec,
        base_template=_TEMPLATE_FOR_BASE["meta/llama-3.2-3b-instruct"],
        dataset_entity=_DATASET_FOR_COLLECTION["nim_curated"],
        output_model_entity="default/lora-nim-llama-3.2-3b-r16",
        description="unit test",
        client=fake_client,
    )

    assert job_id == "cust-abc123"
    fake_client.submit_job.assert_called_once()

    # The config dict passed to submit_job must have the correct top-level shape
    submitted_cfg = fake_client.submit_job.call_args[0][0]
    assert submitted_cfg["dataset"] == "default/stage3-nim-curated"
    assert submitted_cfg["output_model"] == "default/lora-nim-llama-3.2-3b-r16"
    assert "output_model_path" not in submitted_cfg


# ---------------------------------------------------------------------------
# Test 3: CLI --dry-run smoke test (subprocess)
# ---------------------------------------------------------------------------

def test_cli_dry_run_produces_valid_json_with_correct_shape():
    """CLI --dry-run prints valid JSON matching the Customizer 25.12 wire schema."""
    python = "<USER_HOME>/anaconda3/envs/nat/bin/python3"
    script = str(
        Path(__file__).resolve().parents[1]
        / "scripts" / "stage3" / "train_adapter.py"
    )
    result = subprocess.run(
        [
            python, script,
            "--collection", "nim_curated",
            "--base-model", "meta/llama-3.2-3b-instruct",
            "--rank", "16",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI exited non-zero:\n{result.stderr}"
    cfg = json.loads(result.stdout)

    assert cfg["config"] == "meta/llama-3.2-3b-instruct@v1.0.0+80GB"
    assert cfg["dataset"] == "default/stage3-nim-curated"
    assert cfg["output_model"] == "default/lora-nim-llama-3.2-3b-r16"

    hp = cfg["hyperparameters"]
    assert hp["batch_size"] == 16
    assert hp["warmup_steps"] == 30
    assert hp["optimizer"] == "adamw_with_cosine_annealing"
    assert hp["lora"]["adapter_dim"] == 16
    assert hp["lora"]["alpha"] == 32
    assert hp["lora"]["target_modules"] is None

    # No legacy fields
    assert "name" not in cfg
    assert "output_model_path" not in cfg
    assert "precision" not in hp



def test_cli_dry_run_accepts_augmented_dataset_overrides():
    """CLI can target an augmented dataset entity without changing baseline mappings."""
    python = "<USER_HOME>/anaconda3/envs/nat/bin/python3"
    script = str(
        Path(__file__).resolve().parents[1]
        / "scripts" / "stage3" / "train_adapter.py"
    )
    result = subprocess.run(
        [
            python, script,
            "--collection", "nim_curated",
            "--base-model", "meta/llama-3.2-1b-instruct",
            "--rank", "16",
            "--dataset-entity", "default/stage3-nim-curated-dd-kimi-v1",
            "--adapter-name", "lora-nim-dd-kimi-llama-3.2-1b-r16",
            "--description", "Stage 3 augmented NIM 1B r16 test",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI exited non-zero:\n{result.stderr}"
    cfg = json.loads(result.stdout)

    assert cfg["dataset"] == "default/stage3-nim-curated-dd-kimi-v1"
    assert cfg["output_model"] == "default/lora-nim-dd-kimi-llama-3.2-1b-r16"
    assert cfg["description"] == "Stage 3 augmented NIM 1B r16 test"
    assert cfg["hyperparameters"]["lora"]["adapter_dim"] == 16
