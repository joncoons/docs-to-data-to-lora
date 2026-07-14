"""Tests for Stage 2 QA admission/refinement."""
import json
from unittest.mock import MagicMock

from scripts.pipeline.models import KVPRow
from scripts.pipeline.stage2_qa_eval import (
    evaluate_row,
    parse_eval_response,
    refine_row,
    run_stage2,
)


def _row(q="Q?", a="A.", ctx="ctx"):
    return KVPRow(
        passage_id="p", source_url="u", product_family="nim",
        stage="1a", question=q, answer=a, context=ctx, refined=False,
    )


def _llm(response):
    llm = MagicMock()
    llm.call.return_value = response
    llm.model = "nvidia/nvidia/nemotron-3-ultra"
    llm.endpoints = ["http://llm-judge.default.svc.cluster.local:8000/v1"]
    return llm


def test_parse_eval_response_compact_payload_unchanged():
    raw = json.dumps({"prompt": "Q?", "completion": "A."})
    ev = parse_eval_response(raw)
    assert ev is not None
    assert ev.admit is True
    assert ev.prompt == "Q?"


def test_parse_eval_response_explicit_reject():
    raw = json.dumps({"admit": False, "reason": "not grounded"})
    ev = parse_eval_response(raw)
    assert ev is not None
    assert ev.admit is False
    assert ev.reason == "not grounded"


def test_refine_row_passthrough_when_match():
    out = refine_row(_row(), _llm(json.dumps({"prompt": "Q?", "completion": "A."})))
    assert out is not None
    assert out.refined is False  # text didn't change
    assert out.question == "Q?"
    assert out.qa_status == "accepted"
    assert out.qa_admitted is True
    assert out.qa_judge_model == "nvidia/nvidia/nemotron-3-ultra"


def test_refine_row_marks_refined_when_changed():
    llm = _llm(json.dumps({"prompt": "Better Q?", "completion": "Better A."}))
    out = refine_row(_row(), llm)
    assert out is not None
    assert out.refined is True
    assert out.qa_status == "refined"
    assert out.question == "Better Q?"


def test_refine_row_drops_on_bad_response():
    out = refine_row(_row(), _llm("not json"))
    assert out is None


def test_evaluate_row_drops_when_admit_false():
    response = json.dumps({
        "admit": False,
        "grounded": False,
        "answer_fidelity": False,
        "no_hallucination": False,
        "repairable": False,
        "reason": "answer cannot be derived from the source",
    })
    status, admitted, dropped, decision = evaluate_row(
        _row(),
        _llm(response),
        execution_surface="curator_llm_quality",
    )

    assert status == "dropped"
    assert admitted is None
    assert dropped is not None
    assert dropped.qa_status == "dropped"
    assert dropped.qa_admitted is False
    assert dropped.qa_execution_surface == "curator_llm_quality"
    assert decision["admitted"] is False
    assert decision["reason"] == "answer cannot be derived from the source"


def test_run_stage2_writes_quality_sidecar_and_resumes(tmp_path):
    llm = _llm(None)
    llm.call.side_effect = [
        json.dumps({
            "admit": True,
            "prompt": "Q?",
            "completion": "A.",
            "grounded": True,
            "answer_fidelity": True,
            "no_hallucination": True,
            "repairable": True,
            "reason": "grounded",
        }),
        json.dumps({
            "admit": False,
            "grounded": False,
            "answer_fidelity": False,
            "no_hallucination": False,
            "repairable": False,
            "reason": "unsupported",
        }),
    ]
    rows = [_row("Q?", "A."), _row("Bad?", "Unsupported.")]

    kept, dropped = run_stage2(
        rows,
        llm,
        tmp_path,
        max_workers=1,
        resume=False,
        execution_surface="curator_llm_quality",
    )

    assert len(kept) == 1
    assert len(dropped) == 1
    quality_lines = [
        json.loads(line)
        for line in (tmp_path / "provenance" / "stage2_quality.jsonl").read_text().splitlines()
        if line.strip()
    ]
    statuses = sorted(line["status"] for line in quality_lines)
    assert statuses == ["accepted", "dropped"]
    assert {line["judge_model"] for line in quality_lines} == {"nvidia/nvidia/nemotron-3-ultra"}
    assert (tmp_path / "stage2_eval.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "stage2_dropped.jsonl").read_text().count("\n") == 1

    llm.call.reset_mock()
    kept_again, dropped_again = run_stage2(
        rows,
        llm,
        tmp_path,
        max_workers=1,
        resume=True,
        execution_surface="curator_llm_quality",
    )
    llm.call.assert_not_called()
    assert len(kept_again) == 1
    assert len(dropped_again) == 1
