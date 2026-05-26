"""Tests for Stage 1C instruction diversity."""
import json
from unittest.mock import MagicMock

from scripts.pipeline.models import Passage
from scripts.pipeline.stage1c_instruction import (
    density_score, select_top_density_passages,
    parse_instruction_response, process_passage_1c,
)


def _make_passage(token_count, product_family="nim", chunk_ids=None, text=None):
    return Passage(
        passage_id=f"p_{token_count}",
        url=f"https://x.com/p_{token_count}",
        text=text or ("body " * (token_count // 2)),
        token_count=token_count,
        chunk_ids=chunk_ids or ["1"],
        product_family=product_family,
        product_name="prod",
        doc_kind="html",
    )


def test_density_score_proportional_to_chunk_span():
    p1 = _make_passage(token_count=200, chunk_ids=["a", "b", "c", "d"])  # 4 chunks
    p2 = _make_passage(token_count=200, chunk_ids=["a"])  # 1 chunk
    assert density_score(p1) > density_score(p2)


def test_select_top_25_pct_with_floor():
    passages = [_make_passage(token_count=100 + i) for i in range(200)]
    selected = select_top_density_passages(passages, top_percent=0.25, min_passages=100)
    assert len(selected) == 50  # 25% of 200 (above floor)

    # Small corpus — floor kicks in
    small = [_make_passage(token_count=100 + i) for i in range(50)]
    selected = select_top_density_passages(small, top_percent=0.25, min_passages=100)
    assert len(selected) == 50  # all of them


def test_parse_instruction_response_three_types():
    raw = json.dumps({"pairs": [
        {"type": "summary",    "question": "S?", "answer": "Sa."},
        {"type": "listicle",   "question": "L?", "answer": "La."},
        {"type": "procedural", "question": "P?", "answer": "Pa."},
    ]})
    pairs = parse_instruction_response(raw)
    assert len(pairs) == 3
    assert {p["type"] for p in pairs} == {"summary", "listicle", "procedural"}


def test_process_passage_1c_emits_rows():
    passage = _make_passage(token_count=500)
    llm = MagicMock()
    llm.call.return_value = json.dumps({"pairs": [
        {"type": "summary",  "question": "S?", "answer": "Sa."},
        {"type": "listicle", "question": "L?", "answer": "La."},
    ]})
    rows = process_passage_1c(passage, "NVIDIA NIM", llm)
    assert len(rows) == 2
    assert all(r.stage == "1c" for r in rows)
    assert {r.instr_type for r in rows} == {"summary", "listicle"}
