"""Tests for the isolated Stage 1A batched-KVP prototype."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

from scripts.pipeline.models import Passage


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "refactors"
        / "stage2-efficiency"
        / "stage1a_batched_kvp.py"
    )
    spec = importlib.util.spec_from_file_location("stage1a_batched_kvp_refactor", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


batched = _load_module()


def _passage() -> Passage:
    return Passage(
        passage_id="https://x.com/doc#p0",
        url="https://x.com/doc",
        text="NIM_MODEL_PROFILE selects the model engine variant. "
        "Set NIM_PEFT_SOURCE to load a LoRA adapter. " * 30,
        token_count=240,
        chunk_ids=["chunk-1"],
        product_family="NIM",
        product_name="NIM LLM",
        doc_kind="html",
        source_revision_id="srcrev_test",
        source_chunk_ids=["chunk_test"],
        source_systems=["web_crawl"],
        source_kinds=["html"],
        modalities=["text"],
    )


def test_parse_batched_kvp_response_valid_json_with_fences():
    raw = "```json\n" + json.dumps({
        "pairs": [
            {
                "entailment_index": 0,
                "premise_index": 1,
                "question": "Which variable selects the profile?",
                "answer": "NIM_MODEL_PROFILE selects the model engine variant.",
            }
        ]
    }) + "\n```"

    parsed = batched.parse_batched_kvp_response(raw)

    assert parsed is not None
    assert len(parsed.pairs) == 1
    assert parsed.pairs[0].entailment_index == 0
    assert parsed.pairs[0].premise_index == 1


def test_process_passage_uses_one_batch_call_for_multiple_premises():
    llm = MagicMock()
    llm.model = "model"
    llm.temperature = 0.2
    le_resp = json.dumps({
        "entailments": [
            {
                "conclusion": "NIM runtime behavior can be configured.",
                "premises": [
                    "NIM_MODEL_PROFILE selects the model engine variant.",
                    "NIM_PEFT_SOURCE loads a LoRA adapter.",
                ],
            }
        ]
    })
    batch_resp = json.dumps({
        "pairs": [
            {
                "entailment_index": 0,
                "premise_index": 0,
                "question": "Which variable selects the NIM model engine variant?",
                "answer": "NIM_MODEL_PROFILE selects the model engine variant.",
            },
            {
                "entailment_index": 0,
                "premise_index": 1,
                "question": "Which variable points NIM at a LoRA adapter?",
                "answer": "NIM_PEFT_SOURCE is used to load a LoRA adapter.",
            },
        ]
    })
    llm.call.side_effect = [le_resp, batch_resp]

    rows = batched.process_passage_1a_batched(_passage(), llm)

    assert len(rows) == 2
    assert llm.call.call_count == 2
    assert [row.premise_index for row in rows] == [0, 1]
    assert all(row.entailment_index == 0 for row in rows)
    assert all(row.entailment_id for row in rows)
    assert all(row.source_revision_ids == ["srcrev_test"] for row in rows)
    assert all(row.source_chunk_ids == ["chunk_test"] for row in rows)


def test_process_passage_falls_back_for_missing_batched_pair():
    llm = MagicMock()
    llm.model = "model"
    llm.temperature = 0.2
    le_resp = json.dumps({
        "entailments": [
            {
                "conclusion": "NIM runtime behavior can be configured.",
                "premises": [
                    "NIM_MODEL_PROFILE selects the model engine variant.",
                    "NIM_PEFT_SOURCE loads a LoRA adapter.",
                ],
            }
        ]
    })
    partial_batch_resp = json.dumps({
        "pairs": [
            {
                "entailment_index": 0,
                "premise_index": 0,
                "question": "Which variable selects the NIM model engine variant?",
                "answer": "NIM_MODEL_PROFILE selects the model engine variant.",
            }
        ]
    })
    fallback_resp = json.dumps({
        "question": "Which variable points NIM at a LoRA adapter?",
        "answer": "NIM_PEFT_SOURCE is used to load a LoRA adapter.",
    })
    llm.call.side_effect = [le_resp, partial_batch_resp, fallback_resp]

    rows = batched.process_passage_1a_batched(_passage(), llm)

    assert len(rows) == 2
    assert llm.call.call_count == 3
    assert sorted(row.premise_index for row in rows) == [0, 1]
    fallback_rows = [
        row for row in rows
        if row.question == "Which variable points NIM at a LoRA adapter?"
    ]
    assert len(fallback_rows) == 1
    assert fallback_rows[0].extractor_prompt_hash == batched.FALLBACK_PROMPT_HASH


def test_process_passage_falls_back_when_batch_response_is_invalid():
    llm = MagicMock()
    llm.model = "model"
    llm.temperature = 0.2
    le_resp = json.dumps({
        "entailments": [
            {
                "conclusion": "NIM runtime behavior can be configured.",
                "premises": [
                    "NIM_MODEL_PROFILE selects the model engine variant.",
                    "NIM_PEFT_SOURCE loads a LoRA adapter.",
                ],
            }
        ]
    })
    fallback_resp_1 = json.dumps({
        "question": "Which variable selects the NIM model engine variant?",
        "answer": "NIM_MODEL_PROFILE selects the model engine variant.",
    })
    fallback_resp_2 = json.dumps({
        "question": "Which variable points NIM at a LoRA adapter?",
        "answer": "NIM_PEFT_SOURCE is used to load a LoRA adapter.",
    })
    llm.call.side_effect = [le_resp, "not json", fallback_resp_1, fallback_resp_2]

    rows = batched.process_passage_1a_batched(
        _passage(),
        llm,
        batch_parse_attempts=1,
    )

    assert len(rows) == 2
    assert llm.call.call_count == 4
    assert sorted(row.premise_index for row in rows) == [0, 1]
    assert all(row.extractor_prompt_hash == batched.FALLBACK_PROMPT_HASH for row in rows)
