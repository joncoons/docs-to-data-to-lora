"""Tests for Stage 4 external-judge validation gate."""
import json
from unittest.mock import MagicMock

from scripts.pipeline.models import KVPRow
from scripts.pipeline.stage4_validation import (
    parse_judge_response, validate_pair, sample_for_validation,
)


def _row(stage="1a", q="Q?", a="A.", ctx="ctx"):
    return KVPRow(
        passage_id="p", source_url="u", product_family="nim",
        stage=stage, question=q, answer=a, context=ctx, refined=False,
    )


def test_parse_judge_response_all_true():
    raw = json.dumps({"grounded": True, "answer_fidelity": True,
                      "no_hallucination": True, "reason": "looks right"})
    g = parse_judge_response(raw)
    assert g["grounded"] is True
    assert g["all_three"] is True


def test_parse_judge_response_one_false():
    raw = json.dumps({"grounded": False, "answer_fidelity": True,
                      "no_hallucination": True, "reason": "missing source claim"})
    g = parse_judge_response(raw)
    assert g["all_three"] is False


def test_validate_pair_calls_judge():
    judge = MagicMock()
    judge.grade.return_value = json.dumps({
        "grounded": True, "answer_fidelity": True,
        "no_hallucination": True, "reason": "ok",
    })
    result = validate_pair(_row(), judge)
    assert result["all_three"] is True


def test_sample_for_validation_stratifies_by_stage():
    rows = [_row(stage="1a") for _ in range(50)] + [_row(stage="1b") for _ in range(50)]
    sample = sample_for_validation(rows, n=20, seed=42)
    stages = {r.stage for r in sample}
    assert stages == {"1a", "1b"}
    assert len(sample) == 20
