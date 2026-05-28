"""Tests for source-agnostic provenance metadata stamped into ES chunks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler.es_client import _build_doc
from crawler.provenance import build_es_provenance, normalize_sha256_hash, sha256_text


def test_normalize_sha256_hash_accepts_legacy_bare_hex():
    digest = "A" * 64
    assert normalize_sha256_hash(digest) == "sha256:" + digest.lower()
    assert normalize_sha256_hash("sha256:" + digest) == "sha256:" + digest.lower()
    assert normalize_sha256_hash("not-a-hash") is None


def test_build_es_provenance_is_source_agnostic():
    raw_hash = "b" * 64
    metadata = {
        "source_system": "image_dense_caption",
        "source_kind": "dense_caption",
        "modality": "image",
        "ingestion_run_id": "ingest_test",
        "source_content_hash": raw_hash,
        "retrieved_at": "2026-05-28T12:00:00Z",
        "content_type": "image/png",
        "bbox": [0, 0, 640, 480],
        "parser_version": "dense-caption-v1",
    }

    bundle = build_es_provenance(
        text="a generated dense caption",
        source_uri="s3://capture/image-1.png",
        chunk_index=0,
        metadata=metadata,
        date_created="2026-05-28T12:01:00Z",
    )

    content_meta = bundle["content_metadata"]
    provenance = bundle["provenance"]
    assert content_meta["source_system"] == "image_dense_caption"
    assert content_meta["modality"] == "image"
    assert content_meta["source_content_hash"] == "sha256:" + raw_hash
    assert content_meta["text_sha256"] == sha256_text("a generated dense caption")
    assert provenance["anchors"]["bbox"] == [0, 0, 640, 480]
    assert provenance["hashes"]["raw_sha256"] == "sha256:" + raw_hash


def test_build_doc_stamps_common_provenance_into_es_metadata():
    raw_hash = "c" * 64
    doc = _build_doc(
        text="video summary chunk",
        vector=[0.1, 0.2, 0.3],
        source_uri="s3://capture/video-1.mp4",
        chunk_index=3,
        metadata={
            "source_system": "video_summary",
            "source_kind": "video_summary_text",
            "modality": "video",
            "ingestion_run_id": "ingest_video",
            "source_content_hash": raw_hash,
            "retrieved_at": "2026-05-28T13:00:00Z",
            "time_start_seconds": 30,
            "time_end_seconds": 60,
            "parser_version": "video-summary-v1",
            "chunker_version": "transcript-window-v1",
        },
    )

    metadata = doc["metadata"]
    content_meta = metadata["content_metadata"]
    assert content_meta["source_system"] == "video_summary"
    assert content_meta["modality"] == "video"
    assert content_meta["source_revision_id"].startswith("srcrev_")
    assert content_meta["source_chunk_id"].startswith("chunk_")
    assert content_meta["source_content_hash"] == "sha256:" + raw_hash
    assert metadata["source"]["ingestion_run_id"] == "ingest_video"
    assert metadata["provenance"]["anchors"]["time_start_seconds"] == 30
    assert metadata["provenance"]["anchors"]["time_end_seconds"] == 60
