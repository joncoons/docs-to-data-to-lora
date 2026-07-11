"""Tests for Stage 4 external-judge validation gate."""
import json
from unittest.mock import MagicMock

from scripts.pipeline.models import KVPRow
from scripts.pipeline.stage4_validation import (
    join_customizer_rows_to_kvp, parse_judge_response, run_stage4,
    sample_for_validation, validate_pair,
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


def test_join_customizer_rows_restores_context():
    source = _row(q="What is NIM?", a="NIM is an NVIDIA inference service.", ctx="source text")
    joined = join_customizer_rows_to_kvp(
        [{"prompt": source.question, "completion": source.answer, "system": "s"}],
        [source],
    )
    assert joined == [source]


def test_join_customizer_rows_raises_on_missing_context():
    try:
        join_customizer_rows_to_kvp(
            [{"prompt": "missing", "completion": "missing", "system": "s"}],
            [_row()],
        )
    except ValueError as exc:
        assert "Could not restore context" in str(exc)
    else:
        raise AssertionError("expected missing finalized row to raise")


def test_run_stage4_writes_report_sample_and_judgments(tmp_path):
    judge = MagicMock()
    judge.grade.return_value = json.dumps({
        "grounded": True,
        "answer_fidelity": True,
        "no_hallucination": True,
        "reason": "ok",
    })
    rows = [_row(stage="1a", q=f"Question {i}?", a=f"Answer {i}.") for i in range(3)]

    report = run_stage4(
        rows,
        judge,
        tmp_path,
        "nim_curated",
        sample_size=2,
        threshold=0.9,
        seed=7,
        judge_metadata={"model": "judge-model"},
    )

    assert report["passed"] is True
    assert report["judge"]["model"] == "judge-model"
    assert (tmp_path / "validation_report.json").exists()
    assert (tmp_path / "validation_sample.jsonl").exists()
    assert (tmp_path / "validation_judgments.jsonl").exists()
    assert len((tmp_path / "validation_judgments.jsonl").read_text().splitlines()) == 2
