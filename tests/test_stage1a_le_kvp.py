"""Tests for Stage 1A: LE extraction → multi-premise KVP expansion."""
import json
from unittest.mock import MagicMock

import pytest

from scripts.pipeline.models import Passage
from scripts.pipeline.stage1a_le_kvp import (
    parse_le_response, parse_kvp_response, process_passage_1a,
)


def test_parse_le_response_valid_json():
    raw = json.dumps({
        "conclusion": "Widgets support FP8.",
        "premises": ["Premise A.", "Premise B."],
        "context": "ctx", "entities": "Widget, FP8",
    })
    le = parse_le_response(raw)
    assert le is not None
    assert le.conclusion == "Widgets support FP8."
    assert len(le.premises) == 2


def test_parse_le_response_with_fences():
    raw = "```json\n" + json.dumps({
        "conclusion": "C.",
        "premises": ["P."],
    }) + "\n```"
    le = parse_le_response(raw)
    assert le is not None


def test_parse_le_response_no_premises_returns_none():
    raw = json.dumps({"conclusion": "C.", "premises": []})
    assert parse_le_response(raw) is None


def test_parse_kvp_response_valid():
    raw = json.dumps({"question": "Q?", "answer": "A."})
    kvp = parse_kvp_response(raw)
    assert kvp.question == "Q?"


def test_process_passage_1a_emits_one_kvp_per_premise():
    passage = Passage(
        passage_id="p#0", url="https://x.com/p",
        text="text body " * 50, token_count=100,
        chunk_ids=["1"], product_family="nim", product_name="nim-llm",
        doc_kind="html",
    )
    llm = MagicMock()
    # First call: LE extraction returns 2 premises
    le_resp = json.dumps({
        "conclusion": "Concl.",
        "premises": ["P1.", "P2."],
        "context": "", "entities": "",
    })
    # Then 2 KVP calls
    kvp_resp_1 = json.dumps({"question": "Q1?", "answer": "A1."})
    kvp_resp_2 = json.dumps({"question": "Q2?", "answer": "A2."})
    llm.call.side_effect = [le_resp, kvp_resp_1, kvp_resp_2]

    rows = process_passage_1a(passage, llm)
    assert len(rows) == 2
    assert rows[0].question == "Q1?" and rows[0].premise_index == 0
    assert rows[1].question == "Q2?" and rows[1].premise_index == 1
    assert all(r.stage == "1a" for r in rows)
    assert all(r.product_family == "nim" for r in rows)
