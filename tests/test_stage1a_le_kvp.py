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
    # fallback wraps single-entailment as a list of one
    assert len(le.entailments) == 1
    assert le.entailments[0].conclusion == "Widgets support FP8."
    assert len(le.entailments[0].premises) == 2


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
    # Single-entailment shape — fallback wraps as LogEntailmentList of 1
    le_resp = json.dumps({
        "conclusion": "Concl.",
        "premises": ["P1.", "P2."],
        "context": "", "entities": "",
    })
    kvp_resp_1 = json.dumps({"question": "Q1?", "answer": "A1."})
    kvp_resp_2 = json.dumps({"question": "Q2?", "answer": "A2."})
    llm.call.side_effect = [le_resp, kvp_resp_1, kvp_resp_2]

    rows = process_passage_1a(passage, llm)
    assert len(rows) == 2
    assert rows[0].question == "Q1?" and rows[0].premise_index == 0
    assert rows[1].question == "Q2?" and rows[1].premise_index == 1
    assert all(r.entailment_index == 0 for r in rows)  # single entailment
    assert all(r.stage == "1a" for r in rows)
    assert all(r.product_family == "nim" for r in rows)


def test_parse_le_response_multi_entailment():
    raw = json.dumps({
        "entailments": [
            {"conclusion": "C1.", "premises": ["P1a.", "P1b."]},
            {"conclusion": "C2.", "premises": ["P2."]},
        ]
    })
    le = parse_le_response(raw)
    assert le is not None
    assert len(le.entailments) == 2


def test_process_passage_1a_iterates_entailments():
    """Multi-entailment passage emits KVPs across all entailments."""
    passage = Passage(
        passage_id="p#0", url="https://x.com/p",
        text="text body " * 50, token_count=100,
        chunk_ids=["1"], product_family="nim", product_name="nim-llm",
        doc_kind="html",
    )
    llm = MagicMock()
    # LE returns 2 entailments
    le_resp = json.dumps({"entailments": [
        {"conclusion": "C1.", "premises": ["P1a."]},
        {"conclusion": "C2.", "premises": ["P2a.", "P2b."]},
    ]})
    # 3 KVP calls total (1 + 2)
    kvp1 = json.dumps({"question": "Q1?", "answer": "A1."})
    kvp2 = json.dumps({"question": "Q2?", "answer": "A2."})
    kvp3 = json.dumps({"question": "Q3?", "answer": "A3."})
    llm.call.side_effect = [le_resp, kvp1, kvp2, kvp3]

    rows = process_passage_1a(passage, llm)
    assert len(rows) == 3
    assert {r.entailment_index for r in rows} == {0, 1}
    assert all(r.stage == "1a" for r in rows)
