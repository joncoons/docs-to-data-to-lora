"""Tests for Stage 1A provenance sidecar emission."""

import json
from unittest.mock import MagicMock

from scripts.pipeline.models import Passage
from scripts.pipeline.stage1a_le_kvp import run_stage1a


def test_run_stage1a_writes_entailment_sidecar(tmp_output_dir):
    passage = Passage(
        passage_id="https://docs.example.com/p#p0",
        url="https://docs.example.com/p",
        text="NIM can serve models through an OpenAI-compatible endpoint. " * 20,
        token_count=120,
        chunk_ids=["es-1"],
        product_family="nim",
        product_name="nim-llm",
        doc_kind="html",
    )
    llm = MagicMock()
    llm.call.side_effect = [
        json.dumps({
            "conclusion": "NIM can serve models through an OpenAI-compatible endpoint.",
            "premises": ["NIM exposes compatible inference endpoints."],
        }),
        json.dumps({
            "question": "What endpoint style can NIM expose?",
            "answer": "NIM can expose an OpenAI-compatible endpoint.",
        }),
    ]

    rows = run_stage1a([passage], llm, tmp_output_dir, max_workers=1)

    sidecar = tmp_output_dir / "provenance" / "entailments.jsonl"
    assert len(rows) == 1
    assert sidecar.exists()
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["entailment_id"] == rows[0].entailment_id
    assert records[0]["evidence"][0]["chunk_id"].startswith("chunk_")
