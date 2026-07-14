"""Tests for Stage 1A: LE extraction → multi-premise KVP expansion."""
import json
from unittest.mock import MagicMock

import pytest

from scripts.pipeline.models import Passage
from scripts.pipeline.stage1a_le_kvp import (
    output_filenames_for_shard,
    parse_le_response,
    parse_kvp_response,
    process_passage_1a,
    select_shard_passages,
    shard_label,
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
    assert llm.call.call_args_list[0].kwargs["max_tokens"] == 16384
    assert rows[0].question == "Q1?" and rows[0].premise_index == 0
    assert rows[1].question == "Q2?" and rows[1].premise_index == 1
    assert all(r.entailment_index == 0 for r in rows)  # single entailment
    assert all(r.stage == "1a" for r in rows)
    assert all(r.product_family == "nim" for r in rows)


def test_process_passage_1a_emits_all_premises_without_cap():
    passage = Passage(
        passage_id="p#0", url="https://x.com/p",
        text="text body " * 50, token_count=100,
        chunk_ids=["1"], product_family="nim", product_name="nim-llm",
        doc_kind="html",
    )
    llm = MagicMock()
    premises = [f"P{i}." for i in range(4)]
    le_resp = json.dumps({
        "conclusion": "Concl.",
        "premises": premises,
        "context": "", "entities": "",
    })
    kvp_responses = [
        json.dumps({"question": f"Q{i}?", "answer": f"A{i}."})
        for i in range(4)
    ]
    llm.call.side_effect = [le_resp, *kvp_responses]

    rows = process_passage_1a(passage, llm)

    assert len(rows) == 4
    assert llm.call.call_count == 5
    assert [row.premise_index for row in rows] == [0, 1, 2, 3]
    assert [row.question for row in rows] == ["Q0?", "Q1?", "Q2?", "Q3?"]


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


def test_parse_le_response_accepts_unbounded_entailments_and_premises():
    raw = json.dumps({
        "entailments": [
            {
                "conclusion": f"C{ent_idx}.",
                "premises": [f"P{ent_idx}-{prem_idx}." for prem_idx in range(4)],
            }
            for ent_idx in range(12)
        ]
    })

    le = parse_le_response(raw)

    assert le is not None
    assert len(le.entailments) == 12
    assert all(len(ent.premises) == 4 for ent in le.entailments)


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



def _passage_for_shard_test(index: int) -> Passage:
    return Passage(
        passage_id=f"https://x.com/doc-{index}#p0",
        url=f"https://x.com/doc-{index}",
        text="text body " * 50,
        token_count=100,
        chunk_ids=[f"chunk-{index}"],
        product_family="nim",
        product_name="nim-llm",
        doc_kind="html",
    )


def test_select_shard_passages_assigns_each_passage_once():
    passages = [_passage_for_shard_test(i) for i in range(40)]
    selected_ids = []
    for shard_index in range(4):
        shard_passages = select_shard_passages(
            passages,
            shard_index=shard_index,
            shard_count=4,
        )
        selected_ids.extend(p.passage_id for p in shard_passages)

    assert sorted(selected_ids) == sorted(p.passage_id for p in passages)
    assert len(selected_ids) == len(set(selected_ids))


def test_output_filenames_for_shard():
    assert output_filenames_for_shard(0, 1) == ("stage1a_le.jsonl", "entailments.jsonl")
    assert shard_label(2, 8) == "shard-00002-of-00008"
    assert output_filenames_for_shard(2, 8) == (
        "stage1a_le.shard-00002-of-00008.jsonl",
        "entailments.shard-00002-of-00008.jsonl",
    )
