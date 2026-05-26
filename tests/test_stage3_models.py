"""Tests for Stage 3 Pydantic models."""
from scripts.stage3.models import (
    AdapterSpec, ScoreCard, PairwiseOutcome, JudgeScores,
)


def test_adapter_spec_basic():
    spec = AdapterSpec(
        adapter_name="lora-nim-llama3.2-3b-r16",
        collection="nim_curated",
        base_model="meta/llama-3.2-3b-instruct",
        rank=16,
        alpha=32,
    )
    assert spec.adapter_name.startswith("lora-")
    assert spec.alpha == 2 * spec.rank


def test_adapter_spec_rejects_bad_alpha_ratio():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AdapterSpec(
            adapter_name="bad",
            collection="nim_curated",
            base_model="meta/llama-3.2-3b-instruct",
            rank=16,
            alpha=99,  # not 2× rank → reject
        )


def test_judge_scores_clamped():
    s = JudgeScores(accuracy=4, completeness=5, faithfulness=3, clarity=4)
    assert all(1 <= v <= 5 for v in s.model_dump().values())


def test_judge_scores_rejects_out_of_range():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        JudgeScores(accuracy=6, completeness=5, faithfulness=3, clarity=4)


def test_scorecard_aggregates():
    sc = ScoreCard(
        adapter_name="lora-nim-llama3.2-3b-r16",
        n_samples=100,
        mean_accuracy=4.2, mean_completeness=4.0,
        mean_faithfulness=3.8, mean_clarity=4.5,
    )
    j = sc.model_dump_json()
    assert '"mean_accuracy":4.2' in j


def test_pairwise_outcome_three_values():
    from pydantic import ValidationError
    import pytest
    PairwiseOutcome(question_id="q1", a_name="A", b_name="B", winner="A", reason="r")
    PairwiseOutcome(question_id="q1", a_name="A", b_name="B", winner="B", reason="r")
    PairwiseOutcome(question_id="q1", a_name="A", b_name="B", winner="TIE", reason="r")
    with pytest.raises(ValidationError):
        PairwiseOutcome(question_id="q1", a_name="A", b_name="B", winner="C", reason="r")
