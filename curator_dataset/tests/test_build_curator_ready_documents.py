from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from build_curator_ready_documents import (  # noqa: E402
    build_curator_ready_documents,
    is_acknowledgement_url,
)
from build_url_documents import HtmlChunk  # noqa: E402


def _chunk(es_id: str, url: str, text: str, chunk_index: int) -> HtmlChunk:
    return HtmlChunk(
        es_id=es_id,
        url=url,
        text=text,
        chunk_index=chunk_index,
        product_family="NIM",
        product_name="Example",
        source_revision_id=None,
        source_chunk_id=f"chunk_{es_id}",
        source_system="web_crawl",
        source_kind="web_page",
        modality="text",
    )


def test_acknowledgement_url_variants_are_excluded():
    assert is_acknowledgement_url("https://example.test/acknowledgements.html")
    assert is_acknowledgement_url("https://example.test/Acknowledgments/index.html?x=1")
    assert not is_acknowledgement_url("https://example.test/deployment.html")


def test_builder_excludes_acknowledgements_and_deduplicates_within_url():
    chunks = [
        _chunk("ack", "https://example.test/acknowledgements.html", "license", 0),
        _chunk("b", "https://example.test/page", "same", 1),
        _chunk("a", "https://example.test/page", "same", 0),
        _chunk("c", "https://example.test/page", "different", 2),
        _chunk("d", "https://example.test/other", "same", 0),
    ]

    documents = build_curator_ready_documents(chunks, "nim_curated")

    assert [document["url"] for document in documents] == [
        "https://example.test/other",
        "https://example.test/page",
    ]
    page = documents[1]
    assert page["text"] == "same\n\ndifferent"
    assert page["source_chunk_count"] == 3
    assert page["chunk_count"] == 2
    assert page["dropped_exact_duplicate_es_chunk_ids"] == ["b"]
    assert documents[0]["text"] == "same"
