"""Pydantic models for Stage 3 LoRA training + evaluation."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AdapterSpec(BaseModel):
    """One LoRA adapter in the experiment matrix."""
    adapter_name: str
    collection: Literal["nim_curated", "nemo_usvcs_curated"]
    base_model: str
    rank: int = Field(ge=1, le=128)
    alpha: int = Field(ge=1)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    @model_validator(mode="after")
    def alpha_is_two_times_rank(self):
        if self.alpha != 2 * self.rank:
            raise ValueError(f"alpha ({self.alpha}) must equal 2 × rank ({2 * self.rank})")
        return self


class JudgeScores(BaseModel):
    """Per-pair scores from the single-axis judge."""
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)


class ScoreCard(BaseModel):
    """Per-adapter aggregated single-axis scorecard."""
    adapter_name: str
    n_samples: int
    mean_accuracy: float
    mean_completeness: float
    mean_faithfulness: float
    mean_clarity: float


class PairwiseOutcome(BaseModel):
    """One adjudicated A-vs-B comparison (post position-swap)."""
    question_id: str
    a_name: str
    b_name: str
    winner: Literal["A", "B", "TIE"]
    reason: str = ""
