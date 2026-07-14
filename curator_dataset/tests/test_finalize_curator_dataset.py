from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from finalize_curator_dataset import (  # noqa: E402
    balance_by_document,
    dedupe_questions,
    split_documents,
)


def _row(sample: str, document: str, segment: int, question: str) -> dict:
    return {
        "sample_id": sample,
        "prompt": question,
        "completion": "Answer",
        "system": "System",
        "lineage": {"document_id": document, "segment_id": segment},
    }


def test_question_dedupe_is_case_and_whitespace_insensitive():
    rows = [_row("b", "d1", 0, "What  is NIM?"), _row("a", "d2", 0, "what is nim?")]
    deduped, dropped = dedupe_questions(rows)
    assert [row["sample_id"] for row in deduped] == ["a"]
    assert dropped == 1


def test_document_cap_round_robins_segments():
    rows = [
        _row("a1", "d1", 0, "Q1"),
        _row("a2", "d1", 0, "Q2"),
        _row("b1", "d1", 1, "Q3"),
        _row("b2", "d1", 1, "Q4"),
    ]
    balanced, dropped = balance_by_document(rows, 2)
    assert [row["sample_id"] for row in balanced] == ["a1", "b1"]
    assert dropped == 2


def test_split_has_no_document_leakage():
    rows = [
        _row(f"{document}-{index}", document, 0, f"Q {document} {index}")
        for document in ("d1", "d2", "d3", "d4")
        for index in range(2)
    ]
    train, validation, train_ids, validation_ids = split_documents(rows, 0.25, "seed")
    assert len(validation_ids) == 1
    assert not set(train_ids) & set(validation_ids)
    assert {row["lineage"]["document_id"] for row in train} == set(train_ids)
    assert {row["lineage"]["document_id"] for row in validation} == set(validation_ids)
