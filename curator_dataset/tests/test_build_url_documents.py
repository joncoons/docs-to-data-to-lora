from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.build_url_documents import (  # noqa: E402
    HtmlChunk,
    build_documents,
    parse_hit,
    require_experiment_path,
)


def _hit(
    es_id: str,
    url: str,
    text: str,
    chunk_index: int,
    *,
    modality: str = "text",
) -> dict:
    return {
        "_id": es_id,
        "_source": {
            "text": text,
            "metadata": {
                "product_family": "NIM",
                "product_name": "Example",
                "content_metadata": {
                    "content_url": url,
                    "chunk_index": chunk_index,
                    "modality": modality,
                    "source_chunk_id": f"chunk_{es_id}",
                },
            },
        },
    }


def _chunk(es_id: str, url: str, text: str, chunk_index: int) -> HtmlChunk:
    return HtmlChunk(
        es_id=es_id,
        url=url,
        text=text,
        chunk_index=chunk_index,
        product_family="NIM",
        product_name="Example",
        source_revision_id=None,
        source_chunk_id=None,
        source_system=None,
        source_kind=None,
        modality="text",
    )


def test_parse_hit_accepts_html_and_excludes_pdf_and_non_text_modalities():
    chunk, reason = parse_hit(_hit("a", "https://example.test/page", "HTML", 0))
    assert chunk is not None
    assert reason is None

    chunk, reason = parse_hit(_hit("b", "https://example.test/file.pdf", "PDF", 0))
    assert chunk is None
    assert reason == "non_html"

    chunk, reason = parse_hit(
        _hit("c", "https://example.test/image", "caption", 0, modality="image")
    )
    assert chunk is None
    assert reason == "non_html"


def test_build_documents_groups_by_url_and_orders_by_index_then_id():
    chunks = [
        _chunk("z", "https://example.test/a", "third", 2),
        _chunk("b", "https://example.test/a", "second-b", 1),
        _chunk("a", "https://example.test/a", "second-a", 1),
        _chunk("x", "https://example.test/b", "other", 0),
    ]

    documents = build_documents(chunks, "nim_curated")

    assert [document["url"] for document in documents] == [
        "https://example.test/a",
        "https://example.test/b",
    ]
    assert documents[0]["text"] == "second-a\n\nsecond-b\n\nthird"
    assert documents[0]["es_chunk_ids"] == ["a", "b", "z"]
    assert documents[0]["doc_kind"] == "html"


def test_build_documents_reports_but_retains_exact_duplicate_text():
    chunks = [
        _chunk("a", "https://example.test/a", "same", 0),
        _chunk("b", "https://example.test/a", "same", 1),
    ]

    document = build_documents(chunks, "nim_curated")[0]

    assert document["text"] == "same\n\nsame"
    assert document["exact_duplicate_chunk_text_count"] == 1


def test_output_path_must_remain_in_experiment_directory(tmp_path):
    allowed = EXPERIMENT_ROOT / "data" / "output.jsonl"
    assert require_experiment_path(allowed) == allowed.resolve()

    with pytest.raises(ValueError, match="output must be beneath"):
        require_experiment_path(tmp_path / "outside.jsonl")


def test_document_rows_are_json_serializable():
    chunk, _ = parse_hit(_hit("a", "https://example.test/page", "HTML text", 0))
    assert chunk is not None
    json.dumps(build_documents([chunk], "nim_curated")[0])
