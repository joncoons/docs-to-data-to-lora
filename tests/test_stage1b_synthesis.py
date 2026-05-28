"""Tests for Stage 1B kNN neighborhood synthesis."""
import json
from unittest.mock import MagicMock

from scripts.pipeline.models import Passage
from scripts.pipeline.stage1b_synthesis import (
    build_neighborhood_context, parse_synthesis_response, process_passage_1b,
)


def test_build_neighborhood_context_caps_tokens():
    seed_text = "Seed passage. " * 50  # ~150 tokens
    neighbors = [
        {"text": "Neighbor 1. " * 50},
        {"text": "Neighbor 2. " * 50},
        {"text": "Neighbor 3. " * 50},
    ]
    ctx = build_neighborhood_context(seed_text, neighbors, max_tokens=300)
    # Should be capped; tiktoken count must be <= cap
    from scripts.pipeline.noise_filter import count_tokens
    assert count_tokens(ctx) <= 320  # small slack for separator chars


def test_parse_synthesis_response_two_pairs():
    raw = json.dumps({
        "pairs": [
            {"type": "bridging",    "question": "B?", "answer": "Ba."},
            {"type": "contrastive", "question": "C?", "answer": "Ca."},
        ]
    })
    pairs = parse_synthesis_response(raw)
    assert len(pairs) == 2
    assert pairs[0]["type"] == "bridging"


def test_process_passage_1b_returns_two_rows():
    passage = Passage(
        passage_id="p#0", url="https://x.com/p",
        text="seed text " * 30, token_count=60,
        chunk_ids=["1"], product_family="nim", product_name="nim-llm",
        doc_kind="html",
        source_systems=["web_crawl"],
        source_kinds=["web_page"],
        modalities=["text"],
    )
    seed_vec = [0.1, 0.2, 0.3]
    es = MagicMock()
    es.search.return_value = {"hits": {"hits": [
        {"_id": "n1", "_source": {"text": "neighbor 1 " * 30,
            "metadata": {"content_metadata": {"content_url": "https://x.com/n1"}}}},
        {"_id": "n2", "_source": {"text": "neighbor 2 " * 30,
            "metadata": {"content_metadata": {"content_url": "https://x.com/n2"}}}},
    ]}}
    llm = MagicMock()
    llm.call.return_value = json.dumps({"pairs": [
        {"type": "bridging",    "question": "B?", "answer": "Ba."},
        {"type": "contrastive", "question": "C?", "answer": "Ca."},
    ]})

    rows = process_passage_1b(
        passage, seed_vec, es,
        index="nim_curated", domain="NVIDIA NIM", llm=llm,
        k=6, num_candidates=50, top_neighbors=3, max_context_tokens=500,
    )

    assert len(rows) == 2
    assert rows[0].qa_type == "bridging"
    assert rows[1].qa_type == "contrastive"
    assert all(r.stage == "1b" for r in rows)
    assert all(r.source_systems == ["web_crawl"] for r in rows)
    assert all(r.source_kinds == ["web_page"] for r in rows)
    assert all(r.modalities == ["text"] for r in rows)
    assert all("https://x.com/n1" in (r.neighbor_urls or []) for r in rows)
