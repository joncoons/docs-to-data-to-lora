"""Tests for MoEAdapterSpec — α=rank constraint and shard literal enforcement."""
import pytest
from pydantic import ValidationError

from scripts.stage3.moe_models import MoEAdapterSpec


def _spec(**overrides) -> MoEAdapterSpec:
    """Build a valid MoEAdapterSpec with sensible defaults."""
    defaults = dict(
        adapter_name="lora-nim-nemotron-nano-30b-r16-shard-a",
        collection="nim_curated",
        base_model="nvidia/nemotron-3-nano-30b-a3b",
        rank=16,
        alpha=16,
        shard="a",
    )
    defaults.update(overrides)
    return MoEAdapterSpec(**defaults)


# 1 — α=r=16 accepted
def test_alpha_equals_rank_16_accepted():
    spec = _spec(rank=16, alpha=16, shard="a")
    assert spec.alpha == spec.rank == 16


# 2 — α=r=32 accepted
def test_alpha_equals_rank_32_accepted():
    spec = _spec(rank=32, alpha=32, shard="b",
                 adapter_name="lora-nim-nemotron-nano-30b-r32-shard-b")
    assert spec.alpha == spec.rank == 32


# 3 — α=2r rejected (dense pattern must not bleed in)
def test_alpha_two_times_rank_rejected():
    with pytest.raises(ValidationError, match="MoE alpha"):
        _spec(rank=16, alpha=32)


# 4 — shard literal enforcement: only "a" or "b" accepted
def test_shard_literal_only_a_or_b():
    # valid shards
    _spec(shard="a")
    _spec(shard="b", adapter_name="lora-nim-nemotron-nano-30b-r16-shard-b")
    # invalid shard
    with pytest.raises(ValidationError):
        _spec(shard="c")


# 5 — default target_modules populated correctly
def test_default_target_modules():
    spec = _spec()
    assert spec.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
