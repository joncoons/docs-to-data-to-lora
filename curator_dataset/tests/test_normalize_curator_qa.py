from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from normalize_curator_qa import normalize_rows, parse_pairs  # noqa: E402


def test_parse_pairs_accepts_curator_same_line_format():
    response = """Here are the questions and answers based on the provided text:
- Question: What is NIM? Answer: A service.
- Question: Is it supported? Answer: Yes.
"""
    assert parse_pairs(response) == [
        ("What is NIM?", "A service."),
        ("Is it supported?", "Yes."),
    ]


def test_normalize_rows_preserves_lineage_and_drops_exact_pairs():
    raw = {
        "id": "doc_1",
        "document_id": "doc_1",
        "segment_id": 2,
        "url": "https://example.test",
        "collection": "nim_curated",
        "text": "Context",
        "diverse_qa": "- Question: What? Answer: This.",
    }
    rows, metrics = normalize_rows([raw, raw])
    assert len(rows) == 1
    assert rows[0]["lineage"]["document_id"] == "doc_1"
    assert rows[0]["lineage"]["segment_id"] == 2
    assert metrics["exact_duplicate_pairs_dropped"] == 1
    assert metrics["parsed_segments"] == 2
