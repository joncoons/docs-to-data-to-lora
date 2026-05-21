"""Tests for Stage 2 QA Eval refinement."""
import json
from unittest.mock import MagicMock

from scripts.pipeline.models import KVPRow
from scripts.pipeline.stage2_qa_eval import refine_row, parse_eval_response


def _row(q="Q?", a="A.", ctx="ctx"):
    return KVPRow(
        passage_id="p", source_url="u", product_family="nim",
        stage="1a", question=q, answer=a, context=ctx, refined=False,
    )


def test_parse_eval_response_unchanged():
    raw = json.dumps({"prompt": "Q?", "completion": "A."})
    ev = parse_eval_response(raw)
    assert ev is not None
    assert ev.prompt == "Q?"


def test_refine_row_passthrough_when_match():
    llm = MagicMock()
    llm.call.return_value = json.dumps({"prompt": "Q?", "completion": "A."})
    out = refine_row(_row(), llm)
    assert out is not None
    assert out.refined is False  # text didn't change
    assert out.question == "Q?"


def test_refine_row_marks_refined_when_changed():
    llm = MagicMock()
    llm.call.return_value = json.dumps({"prompt": "Better Q?", "completion": "Better A."})
    out = refine_row(_row(), llm)
    assert out is not None
    assert out.refined is True
    assert out.question == "Better Q?"


def test_refine_row_drops_on_bad_response():
    llm = MagicMock()
    llm.call.return_value = "not json"
    out = refine_row(_row(), llm)
    assert out is None
