"""Tests for Stage 0 corpus preparation."""
import json
from pathlib import Path

import pytest

from scripts.pipeline.stage0_corpus_prep import (
    classify_doc_kind, group_html_chunks_by_url, build_passages,
)


def test_classify_doc_kind_pdf():
    assert classify_doc_kind("https://x.com/foo.pdf", "pdf") == "pdf"
    assert classify_doc_kind("https://x.com/foo.PDF", "md") == "pdf"  # URL wins on disagreement
    assert classify_doc_kind("https://x.com/foo.docx", "md") == "pdf"  # binary extension


def test_classify_doc_kind_html():
    assert classify_doc_kind("https://x.com/foo.html", "md") == "html"
    assert classify_doc_kind("https://x.com/foo/", "md") == "html"
    assert classify_doc_kind("https://x.com/foo", "md") == "html"


def test_group_html_chunks_concatenates_all():
    chunks = [
        {"_id": "c1", "url": "https://x.com/a", "chunk_index": 0,
         "text": "Para 1.", "vector": [0.1], "doc_type": "md"},
        {"_id": "c2", "url": "https://x.com/a", "chunk_index": 1,
         "text": "Para 2.", "vector": [0.2], "doc_type": "md"},
        {"_id": "c3", "url": "https://x.com/b", "chunk_index": 0,
         "text": "Other.", "vector": [0.3], "doc_type": "md"},
    ]
    grouped = group_html_chunks_by_url(chunks)
    assert set(grouped.keys()) == {"https://x.com/a", "https://x.com/b"}
    assert grouped["https://x.com/a"][0]["text"].startswith("Para 1.")
    assert "Para 2." in grouped["https://x.com/a"][0]["text"]


def test_build_passages_pdf_kept_per_chunk(sample_chunks):
    """PDF chunks should become individual passages."""
    passages = build_passages(sample_chunks, min_passage_tokens=10)
    pdf_passages = [p for p in passages if p.doc_kind == "pdf"]
    assert len(pdf_passages) >= 1
    # PDF passages should have single chunk_ids
    for p in pdf_passages:
        assert len(p.chunk_ids) == 1


def test_build_passages_html_grouped_by_url(sample_chunks):
    """The two HTML chunks for the same URL should merge into one passage."""
    passages = build_passages(sample_chunks, min_passage_tokens=10)
    html_passages = [p for p in passages if p.doc_kind == "html"]
    same_url = [p for p in html_passages if "nemotron-3-nano" in p.url]
    assert len(same_url) == 1
    assert len(same_url[0].chunk_ids) == 2


def test_build_passages_carries_product_family(sample_chunks):
    passages = build_passages(sample_chunks, min_passage_tokens=10)
    families = {p.product_family for p in passages}
    assert "nim-llm" in families
    assert "nim-medical" in families
