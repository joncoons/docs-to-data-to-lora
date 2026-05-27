"""Pydantic models for MoE LoRA training (α = adapter_dim, ratio 1.0).

Per [[feedback_nemotron_nano_alpha_ratio]], Nemotron-3-Nano-30B-A3B LoRA SFT
requires α/r=1.0 — α/r=2.0 with lr=1e-4 diverges sharply at end-of-warmup.
Dense models (Llama) use α/r=2.0 and live in models.py — this module is the
parallel for MoE so the two specs cannot accidentally cross-contaminate.
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MoEAdapterSpec(BaseModel):
    """One MoE LoRA adapter (one shard) in the experiment matrix."""
    adapter_name: str           # e.g. "lora-nim-nemotron-nano-30b-r16-shard-a"
    collection: Literal["nim_curated", "nemo_usvcs_curated"]
    base_model: str             # canonically "nvidia/nemotron-3-nano-30b-a3b"
    rank: int = Field(ge=1, le=128)
    alpha: int = Field(ge=1)
    shard: Literal["a", "b"]
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    @model_validator(mode="after")
    def alpha_equals_rank(self):
        if self.alpha != self.rank:
            raise ValueError(
                f"MoE alpha ({self.alpha}) must equal rank ({self.rank}) — "
                f"see [[feedback_nemotron_nano_alpha_ratio]]"
            )
        return self
