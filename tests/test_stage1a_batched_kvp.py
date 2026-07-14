"""Tests for optional production Stage 1A batched-KVP mode."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from scripts.pipeline.models import KVPRow, Passage
from scripts.pipeline.stage1a_batched_kvp import (
    BATCHED_PROMPT_HASH,
    FALLBACK_PROMPT_HASH,
    parse_batched_kvp_response,
    process_passage_1a_batched,
    run_stage1a_batched,
)


def _passage() -> Passage:
    return Passage(
        passage_id="https://x.com/doc#p0",
        url="https://x.com/doc",
        text=(
            "NIM_MODEL_PROFILE selects the model engine variant. "
            "Set NIM_PEFT_SOURCE to load a LoRA adapter. "
        )
        * 30,
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

    parsed = parse_batched_kvp_response(raw)

    assert parsed is not None
    assert len(parsed.pairs) == 1
    assert parsed.pairs[0].entailment_index == 0
    assert parsed.pairs[0].premise_index == 1


def test_process_passage_uses_one_batch_call_for_multiple_premises():
    llm = MagicMock()
    llm.model = "model"
    llm.temperature = 0.95
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

    rows = process_passage_1a_batched(_passage(), llm)

    assert len(rows) == 2
    assert llm.call.call_args_list[0].kwargs["max_tokens"] == 16384
    assert llm.call.call_args_list[1].kwargs["max_tokens"] == 16384
    assert llm.call.call_count == 2
    assert [row.premise_index for row in rows] == [0, 1]
    assert all(row.extractor_prompt_hash == BATCHED_PROMPT_HASH for row in rows)
    assert all(row.entailment_id for row in rows)
    assert all(row.source_revision_ids == ["srcrev_test"] for row in rows)
    assert all(row.source_chunk_ids == ["chunk_test"] for row in rows)


def test_process_passage_batches_all_premises_without_cap():
    llm = MagicMock()
    llm.model = "model"
    llm.temperature = 0.95
    premises = [f"NIM premise {idx}." for idx in range(5)]
    le_resp = json.dumps({
        "entailments": [
            {
                "conclusion": "NIM runtime behavior can be configured.",
                "premises": premises,
            }
        ]
    })
    batch_responses = [
        json.dumps({
            "pairs": [
                {
                    "entailment_index": 0,
                    "premise_index": premise_index,
                    "question": f"Question {premise_index}?",
                    "answer": f"Answer {premise_index}.",
                }
                for premise_index in premise_indexes
            ]
        })
        for premise_indexes in ([0, 1], [2, 3], [4])
    ]
    llm.call.side_effect = [le_resp, *batch_responses]

    rows = process_passage_1a_batched(
        _passage(),
        llm,
        max_premises_per_batch=2,
    )

    assert len(rows) == 5
    assert llm.call.call_count == 4
    assert [row.premise_index for row in rows] == [0, 1, 2, 3, 4]
    assert all(row.extractor_prompt_hash == BATCHED_PROMPT_HASH for row in rows)


def test_process_passage_falls_back_for_missing_batched_pair():
    llm = MagicMock()
    llm.model = "model"
    llm.temperature = 0.95
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

    rows = process_passage_1a_batched(_passage(), llm)

    assert len(rows) == 2
    assert llm.call.call_count == 3
    assert llm.call.call_args_list[2].kwargs["max_tokens"] == 16384
    assert sorted(row.premise_index for row in rows) == [0, 1]
    fallback_rows = [
        row for row in rows
        if row.question == "Which variable points NIM at a LoRA adapter?"
    ]
    assert len(fallback_rows) == 1
    assert fallback_rows[0].extractor_prompt_hash == FALLBACK_PROMPT_HASH


def test_run_stage1a_batched_writes_canonical_output_names(tmp_output_dir):
    llm = MagicMock()
    llm.model = "model"
    llm.temperature = 0.95
    llm.call.side_effect = [
        json.dumps({
            "entailments": [
                {
                    "conclusion": "NIM runtime behavior can be configured.",
                    "premises": ["NIM_MODEL_PROFILE selects the model engine variant."],
                }
            ]
        }),
        json.dumps({
            "pairs": [
                {
                    "entailment_index": 0,
                    "premise_index": 0,
                    "question": "Which variable selects the NIM model engine variant?",
                    "answer": "NIM_MODEL_PROFILE selects the model engine variant.",
                }
            ]
        }),
    ]

    rows = run_stage1a_batched([_passage()], llm, tmp_output_dir, max_workers=1)

    out_file = tmp_output_dir / "stage1a_le.jsonl"
    entailments_file = tmp_output_dir / "provenance" / "entailments.jsonl"
    assert len(rows) == 1
    assert out_file.exists()
    assert entailments_file.exists()
    persisted = [
        KVPRow.model_validate_json(line)
        for line in out_file.read_text().splitlines()
        if line
    ]
    assert persisted[0].question == "Which variable selects the NIM model engine variant?"
