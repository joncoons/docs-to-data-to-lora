"""Tests for train_adapter_moe CLI: builder + dry-run smoke test."""
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from scripts.stage3.moe_models import MoEAdapterSpec
from scripts.stage3.train_adapter_moe import (
    _SHARD_DATASET_FOR,
    NANO_BASE_MODEL,
    NANO_CONFIG_TEMPLATE,
    build_customizer_config_moe,
    submit_adapter_job_moe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nano_spec(
    collection: str = "nim_curated",
    rank: int = 16,
    shard: str = "a",
) -> MoEAdapterSpec:
    short = "nim" if collection == "nim_curated" else "nemo-usvcs"
    return MoEAdapterSpec(
        adapter_name=f"lora-{short}-nemotron-nano-30b-r{rank}-shard-{shard}",
        collection=collection,
        base_model=NANO_BASE_MODEL,
        rank=rank,
        alpha=rank,   # α/r = 1.0
        shard=shard,
    )


def _nano_config(
    collection: str = "nim_curated",
    rank: int = 16,
    shard: str = "a",
) -> dict:
    spec = _nano_spec(collection, rank, shard)
    dataset_key = (collection, shard)
    return build_customizer_config_moe(
        spec=spec,
        base_template=NANO_CONFIG_TEMPLATE,
        dataset_entity=_SHARD_DATASET_FOR[dataset_key],
        output_model_entity=f"default/{spec.adapter_name}",
        description=f"Stage 3 MoE — {collection} shard-{shard} r{rank}",
    )


# ---------------------------------------------------------------------------
# Test 1: wire shape — top-level fields
# ---------------------------------------------------------------------------

def test_moe_config_top_level_shape():
    cfg = _nano_config()
    assert cfg["config"] == NANO_CONFIG_TEMPLATE
    assert cfg["dataset"] == "default/stage3-nim-curated-shard-a"
    assert cfg["output_model"] == "default/lora-nim-nemotron-nano-30b-r16-shard-a"
    assert isinstance(cfg["description"], str) and cfg["description"]

    for forbidden in ("name", "output_model_path", "output_format"):
        assert forbidden not in cfg


# ---------------------------------------------------------------------------
# Test 2: hyperparameters — MoE-specific values
# ---------------------------------------------------------------------------

def test_moe_config_hyperparameters():
    hp = _nano_config()["hyperparameters"]
    assert hp["finetuning_type"] == "lora"
    assert hp["training_type"] == "sft"
    assert hp["epochs"] == 2
    assert hp["learning_rate"] == 1.0e-4
    # MoE-specific: batch_size=8, warmup_steps=100 (different from dense defaults)
    assert hp["batch_size"] == 8
    assert hp["warmup_steps"] == 100
    assert hp["seed"] == 42
    assert hp["optimizer"] == "adamw_with_cosine_annealing"
    assert hp["sequence_packing_enabled"] is False

    lora = hp["lora"]
    assert lora["adapter_dim"] == 16
    # α must equal rank (α/r = 1.0) — MoE rule
    assert lora["alpha"] == 16
    assert lora["adapter_dim"] == lora["alpha"]


# ---------------------------------------------------------------------------
# Test 3: α = rank (not 2 × rank as in the dense path)
# ---------------------------------------------------------------------------

def test_moe_config_alpha_equals_rank_not_two_times():
    """α = rank for MoE — dense inadvertently uses α = 2 × rank."""
    for rank in (16, 32):
        cfg = _nano_config(rank=rank)
        lora = cfg["hyperparameters"]["lora"]
        assert lora["alpha"] == rank, f"Expected alpha={rank}, got {lora['alpha']}"
        assert lora["alpha"] != 2 * rank or rank == 0  # sanity: never the dense default


# ---------------------------------------------------------------------------
# Test 4: shard-specific dataset selection
# ---------------------------------------------------------------------------

def test_moe_config_shard_b_uses_shard_b_dataset():
    cfg = _nano_config(collection="nim_curated", shard="b")
    assert cfg["dataset"] == "default/stage3-nim-curated-shard-b"


def test_moe_config_nemo_usvcs_shard_a():
    cfg = _nano_config(collection="nemo_usvcs_curated", shard="a")
    assert cfg["dataset"] == "default/stage3-nemo-usvcs-curated-shard-a"


# ---------------------------------------------------------------------------
# Test 5: Nano template is used (not one of the dense Llama templates)
# ---------------------------------------------------------------------------

def test_moe_config_uses_nano_template():
    cfg = _nano_config()
    assert "nemotron" in cfg["config"].lower() or "nano" in cfg["config"].lower()
    assert "llama" not in cfg["config"].lower()


# ---------------------------------------------------------------------------
# Test 6: submit_adapter_job_moe passes config to client and returns job_id
# ---------------------------------------------------------------------------

def test_submit_adapter_job_moe_calls_client():
    spec = _nano_spec()
    fake_client = MagicMock()
    fake_client.submit_job.return_value = "cust-moe-abc123"

    job_id = submit_adapter_job_moe(
        spec=spec,
        base_template=NANO_CONFIG_TEMPLATE,
        dataset_entity=_SHARD_DATASET_FOR[("nim_curated", "a")],
        output_model_entity="default/lora-nim-nemotron-nano-30b-r16-shard-a",
        description="unit test",
        client=fake_client,
    )

    assert job_id == "cust-moe-abc123"
    fake_client.submit_job.assert_called_once()

    submitted_cfg = fake_client.submit_job.call_args[0][0]
    assert submitted_cfg["dataset"] == "default/stage3-nim-curated-shard-a"
    assert submitted_cfg["hyperparameters"]["lora"]["alpha"] == 16


# ---------------------------------------------------------------------------
# Test 7: CLI --dry-run smoke test (subprocess)
# ---------------------------------------------------------------------------

def test_cli_dry_run_produces_valid_json():
    python = "/home/joncoons/anaconda3/envs/nat/bin/python3"
    script = str(
        Path(__file__).resolve().parents[1]
        / "scripts" / "stage3" / "train_adapter_moe.py"
    )
    result = subprocess.run(
        [
            python, script,
            "--collection", "nim_curated",
            "--rank", "16",
            "--shard", "a",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI exited non-zero:\n{result.stderr}"
    cfg = json.loads(result.stdout)

    assert cfg["config"] == NANO_CONFIG_TEMPLATE
    assert cfg["dataset"] == "default/stage3-nim-curated-shard-a"

    hp = cfg["hyperparameters"]
    assert hp["batch_size"] == 8
    assert hp["warmup_steps"] == 100
    assert hp["lora"]["alpha"] == 16
    assert hp["lora"]["adapter_dim"] == 16
    assert hp["sequence_packing_enabled"] is False

    # No legacy fields
    assert "name" not in cfg
    assert "output_model_path" not in cfg
    assert "precision" not in hp
