"""Tests for Stage 0 corpus preparation."""
import json
from pathlib import Path

import pytest

from scripts.pipeline import stage0_corpus_prep as stage0
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


def test_classify_doc_kind_source_agnostic_modalities_kept_per_chunk():
    assert (
        classify_doc_kind(
            "s3://capture/images/device-panel.png",
            "caption",
            source_system="image_dense_caption",
            source_kind="dense_caption",
            modality="image",
        )
        == "pdf"
    )
    assert (
        classify_doc_kind(
            "s3://capture/video/demo.mp4",
            "summary",
            source_system="video_summary",
            source_kind="video_summary_text",
            modality="video",
        )
        == "pdf"
    )


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


def test_run_stage0_writes_provenance_and_observability(tmp_path, sample_chunks, monkeypatch):
    monkeypatch.setattr(stage0, "scroll_all_chunks", lambda es, index: iter(sample_chunks))

    output_dir = tmp_path / "dataset"
    observability_dir = tmp_path / "observability"
    passages, seed_vectors = stage0.run_stage0(
        object(),
        "nim_curated",
        output_dir,
        min_passage_tokens=10,
        es_host="http://elasticsearch:9200",
        observability_dir=observability_dir,
        pipeline_run_id="pipeline-run-1",
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="docs-to-data-to-lora",
        mlflow_parent_run_id="parent-run-1",
    )

    assert (output_dir / "passages.jsonl").exists()
    assert (output_dir / "manifests" / "crawl_run.json").exists()
    assert (output_dir / "provenance" / "source_revisions.jsonl").exists()
    assert (output_dir / "provenance" / "source_chunks.jsonl").exists()
    assert len(passages) == 2
    assert set(seed_vectors) == {p.passage_id for p in passages}

    run_context = json.loads((observability_dir / "run_context.json").read_text())
    metrics = json.loads((observability_dir / "metrics.json").read_text())
    artifacts = json.loads((observability_dir / "artifacts_manifest.json").read_text())
    service_refs = json.loads((observability_dir / "service_refs.json").read_text())

    assert run_context["pipeline_stage"] == "stage0-corpus-prep"
    assert run_context["pipeline_run_id"] == "pipeline-run-1"
    assert run_context["mlflow"]["tracking_uri"] == "http://mlflow:5000"
    assert metrics["stage0.raw_hits.count"] == 3
    assert metrics["stage0.chunks.extracted"] == 3
    assert metrics["stage0.passages.count"] == 2
    assert metrics["stage0.doc_kind.html"] == 1
    assert metrics["stage0.doc_kind.pdf"] == 1
    assert service_refs["services"]["elasticsearch"]["index"] == "nim_curated"

    artifact_paths = {item["artifact_path"] for item in artifacts["artifacts"]}
    assert "passages.jsonl" in artifact_paths
    assert "manifests/crawl_run.json" in artifact_paths
    assert "provenance/source_revisions.jsonl" in artifact_paths
    assert "provenance/source_chunks.jsonl" in artifact_paths

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _es_provenance_hit(
    es_id: str,
    uri: str,
    text: str,
    *,
    chunk_index: int = 0,
    source_revision_id: str,
    source_chunk_id: str,
    source_system: str,
    source_kind: str,
    modality: str,
    document_type: str = "text",
    raw_sha256: str = "sha256:" + "c" * 64,
    anchors: dict | None = None,
) -> dict:
    anchors = anchors or {}
    provenance = {
        "schema_version": "unstructured-source-provenance.v1",
        "ingestion_run_id": "ingest_test",
        "source_revision_id": source_revision_id,
        "source_chunk_id": source_chunk_id,
        "source_system": source_system,
        "source_kind": source_kind,
        "modality": modality,
        "canonical_uri": uri,
        "final_uri": uri,
        "observed_at": "2026-05-28T12:00:00Z",
        "hashes": {
            "raw_sha256": raw_sha256,
            "text_sha256": "sha256:" + "d" * 64,
        },
        "http": {
            "content_type": "text/plain",
        },
        "anchors": anchors,
    }
    content_metadata = {
        "content_url": uri,
        "canonical_uri": uri,
        "final_uri": uri,
        "chunk_index": chunk_index,
        "document_type": document_type,
        "provenance_schema_version": "unstructured-source-provenance.v1",
        "ingestion_run_id": "ingest_test",
        "source_revision_id": source_revision_id,
        "source_chunk_id": source_chunk_id,
        "source_system": source_system,
        "source_kind": source_kind,
        "modality": modality,
        "raw_sha256": raw_sha256,
        "source_content_hash": raw_sha256,
        "text_sha256": "sha256:" + "d" * 64,
        "retrieved_at": "2026-05-28T12:00:00Z",
        "content_type": "text/plain",
    }
    source = {
        "source_id": f"source_{es_id}",
        "source_name": uri,
        "source_type": source_system,
        "source_system": source_system,
        "source_kind": source_kind,
        "modality": modality,
        "source_revision_id": source_revision_id,
        "source_chunk_id": source_chunk_id,
        "ingestion_run_id": "ingest_test",
        "date_created": "2026-05-28T12:00:00Z",
    }
    return {
        "_id": es_id,
        "_source": {
            "text": text,
            "vector": [0.1, 0.2],
            "metadata": {
                "content_metadata": content_metadata,
                "source": source,
                "provenance": provenance,
                "product_family": "nemo-microservices",
                "product_name": "source-provenance",
            },
        },
    }


def test_run_stage0_prefers_es_source_agnostic_provenance(tmp_path, monkeypatch):
    hits = [
        _es_provenance_hit(
            "image_caption_1",
            "s3://captures/images/panel.png",
            "The dense caption states that the panel shows a NeMo service status screen.",
            source_revision_id="srcrev_image_1",
            source_chunk_id="chunk_image_1",
            source_system="image_dense_caption",
            source_kind="dense_caption",
            modality="image",
            document_type="caption",
            anchors={"bbox": [0, 0, 128, 128]},
        ),
        _es_provenance_hit(
            "video_summary_1",
            "s3://captures/video/customizer-demo.mp4",
            "The video summary states that Customizer training produces adapter artifacts.",
            source_revision_id="srcrev_video_1",
            source_chunk_id="chunk_video_1",
            source_system="video_summary",
            source_kind="video_summary_text",
            modality="video",
            document_type="summary",
            anchors={"timestamp_range_ms": [1000, 8000]},
        ),
    ]
    monkeypatch.setattr(stage0, "scroll_all_chunks", lambda es, index: iter(hits))

    output_dir = tmp_path / "dataset"
    observability_dir = tmp_path / "observability"
    passages, _ = stage0.run_stage0(
        object(),
        "multimodal_curated",
        output_dir,
        min_passage_tokens=1,
        es_host="http://elasticsearch:9200",
        observability_dir=observability_dir,
    )

    source_revisions = _read_jsonl(output_dir / "provenance" / "source_revisions.jsonl")
    source_chunks = _read_jsonl(output_dir / "provenance" / "source_chunks.jsonl")
    metrics = json.loads((observability_dir / "metrics.json").read_text())

    assert {p.source_revision_id for p in passages} == {"srcrev_image_1", "srcrev_video_1"}
    assert {p.source_chunk_ids[0] for p in passages} == {"chunk_image_1", "chunk_video_1"}
    assert {p.doc_kind for p in passages} == {"pdf"}
    assert {tuple(p.source_systems or []) for p in passages} == {
        ("image_dense_caption",),
        ("video_summary",),
    }
    assert {tuple(p.modalities or []) for p in passages} == {("image",), ("video",)}

    image_revision = next(
        item for item in source_revisions if item["source_revision_id"] == "srcrev_image_1"
    )
    assert image_revision["canonical_url"] == "s3://captures/images/panel.png"
    assert image_revision["hashes"]["raw_sha256"] == "sha256:" + "c" * 64
    assert image_revision["classification"]["source_system"] == "image_dense_caption"
    assert image_revision["classification"]["modality"] == "image"
    assert image_revision["metadata"]["upstream"]["source_chunk_ids"] == ["chunk_image_1"]

    image_chunk = next(item for item in source_chunks if item["chunk_id"] == "chunk_image_1")
    assert image_chunk["source_revision_id"] == "srcrev_image_1"
    assert image_chunk["metadata"]["source_system"] == "image_dense_caption"
    assert image_chunk["metadata"]["upstream_source_revision_ids"] == ["srcrev_image_1"]
    assert image_chunk["metadata"]["upstream_source_chunk_ids"] == ["chunk_image_1"]
    assert image_chunk["metadata"]["upstream_provenance"][0]["anchors"]["bbox"] == [
        0,
        0,
        128,
        128,
    ]

    assert metrics["stage0.es_provenance.chunks.count"] == 2
    assert metrics["stage0.es_provenance.source_revisions.count"] == 2
    assert metrics["stage0.es_provenance.source_chunks.count"] == 2
    assert metrics["stage0.url_registry.records.count"] == 0


def test_run_stage0_groups_web_chunks_but_preserves_upstream_ids(tmp_path, monkeypatch):
    url = "https://docs.nvidia.com/nemo/microservices/latest/customizer.html"
    hits = [
        _es_provenance_hit(
            "web_chunk_0",
            url,
            "The Customizer service accepts a dataset reference and base model.",
            chunk_index=0,
            source_revision_id="srcrev_web_1",
            source_chunk_id="chunk_web_0",
            source_system="web_crawl",
            source_kind="web_page",
            modality="text",
            document_type="md",
            anchors={"heading_path": ["Customizer"]},
        ),
        _es_provenance_hit(
            "web_chunk_1",
            url,
            "The Customizer service returns job state and adapter output metadata.",
            chunk_index=1,
            source_revision_id="srcrev_web_1",
            source_chunk_id="chunk_web_1",
            source_system="web_crawl",
            source_kind="web_page",
            modality="text",
            document_type="md",
            anchors={"heading_path": ["Customizer", "Jobs"]},
        ),
    ]
    monkeypatch.setattr(stage0, "scroll_all_chunks", lambda es, index: iter(hits))

    output_dir = tmp_path / "dataset"
    observability_dir = tmp_path / "observability"
    passages, _ = stage0.run_stage0(
        object(),
        "nemo_usvcs_curated",
        output_dir,
        min_passage_tokens=1,
        es_host="http://elasticsearch:9200",
        observability_dir=observability_dir,
    )

    source_revisions = _read_jsonl(output_dir / "provenance" / "source_revisions.jsonl")
    source_chunks = _read_jsonl(output_dir / "provenance" / "source_chunks.jsonl")
    metrics = json.loads((observability_dir / "metrics.json").read_text())

    assert len(passages) == 1
    assert passages[0].doc_kind == "html"
    assert passages[0].source_revision_id == "srcrev_web_1"
    assert passages[0].source_chunk_ids[0] not in {"chunk_web_0", "chunk_web_1"}
    assert passages[0].source_systems == ["web_crawl"]
    assert passages[0].source_kinds == ["web_page"]
    assert passages[0].modalities == ["text"]

    assert source_revisions[0]["source_revision_id"] == "srcrev_web_1"
    assert source_revisions[0]["metadata"]["upstream"]["source_chunk_ids"] == [
        "chunk_web_0",
        "chunk_web_1",
    ]
    assert source_chunks[0]["metadata"]["upstream_source_revision_ids"] == ["srcrev_web_1"]
    assert source_chunks[0]["metadata"]["upstream_source_chunk_ids"] == [
        "chunk_web_0",
        "chunk_web_1",
    ]
    assert source_chunks[0]["metadata"]["external_chunk_ids"] == [
        "web_chunk_0",
        "web_chunk_1",
    ]

    assert metrics["stage0.doc_kind.html"] == 1
    assert metrics["stage0.doc_kind.pdf"] == 0
    assert metrics["stage0.es_provenance.chunks.count"] == 2
    assert metrics["stage0.es_provenance.source_revisions.count"] == 1
    assert metrics["stage0.es_provenance.source_chunks.count"] == 2


def test_url_registry_hash_normalization_and_matching(tmp_path):
    registry_path = tmp_path / "url_registry.json"
    bare_hash = "A" * 64
    registry_path.write_text(json.dumps({
        "https://docs.example.com/page/": {
            "collection": "nim_curated",
            "content_hash": bare_hash,
            "last_seen": "2026-05-19T17:50:09+00:00",
            "status_code": 200,
        }
    }))

    lookup, records = stage0.load_url_registry(registry_path)
    match = stage0.lookup_url_registry(lookup, "https://docs.example.com/page/#install")

    assert len(records) == 1
    assert match is not None
    assert stage0.normalize_sha256_hash(match["content_hash"]) == "sha256:" + bare_hash.lower()


def test_run_stage0_enriches_provenance_from_url_registry(tmp_path, sample_chunks, monkeypatch):
    monkeypatch.setattr(stage0, "scroll_all_chunks", lambda es, index: iter(sample_chunks))

    html_url = "https://docs.nvidia.com/nim/nemotron-3-nano/latest/profiles.html"
    bare_hash = "b" * 64
    registry_path = tmp_path / "url_registry.json"
    registry_path.write_text(json.dumps({
        html_url: {
            "collection": "nim_curated",
            "content_hash": bare_hash,
            "etag": "\"abc\"",
            "last_modified": "Mon, 23 Mar 2026 19:24:22 GMT",
            "last_seen": "2026-05-19T17:50:09+00:00",
            "last_ingested": "2026-05-19T17:45:15+00:00",
            "status_code": 200,
            "linked_hrefs": ["https://docs.nvidia.com/nim/"],
        }
    }))

    output_dir = tmp_path / "dataset"
    observability_dir = tmp_path / "observability"
    stage0.run_stage0(
        object(),
        "nim_curated",
        output_dir,
        min_passage_tokens=10,
        es_host="http://elasticsearch:9200",
        observability_dir=observability_dir,
        url_registry_path=registry_path,
        pipeline_run_id="pipeline-run-1",
    )

    crawl_run = json.loads((output_dir / "manifests" / "crawl_run.json").read_text())
    source_revisions = _read_jsonl(output_dir / "provenance" / "source_revisions.jsonl")
    source_chunks = _read_jsonl(output_dir / "provenance" / "source_chunks.jsonl")
    metrics = json.loads((observability_dir / "metrics.json").read_text())
    run_context = json.loads((observability_dir / "run_context.json").read_text())
    artifacts = json.loads((observability_dir / "artifacts_manifest.json").read_text())

    html_revision = next(item for item in source_revisions if item["canonical_url"] == html_url)
    assert html_revision["retrieved_at"] == "2026-05-19T17:45:15+00:00"
    assert html_revision["http"]["status_code"] == 200
    assert html_revision["http"]["etag"] == "\"abc\""
    assert html_revision["hashes"]["raw_sha256"] == "sha256:" + bare_hash
    assert html_revision["hashes"]["normalized_sha256"].startswith("sha256:")
    assert html_revision["metadata"]["url_registry"]["linked_href_count"] == 1
    assert (
        html_revision["metadata"]["url_registry"]["content_hash_kind"]
        == "crawler_source_content_hash"
    )
    assert html_revision["metadata"]["elasticsearch"]["content_metadata"]["chunk_index"] == 0

    html_chunk = next(item for item in source_chunks if item["metadata"]["source_url"] == html_url)
    assert html_chunk["metadata"]["url_registry"]["registry_url"] == html_url
    assert html_chunk["metadata"]["content_metadata"]["document_type"] == "md"

    assert crawl_run["scope"]["url_registry_uri"] == str(registry_path)
    assert crawl_run["metrics"]["url_registry_record_count"] == 1
    assert crawl_run["metrics"]["url_registry_matched_url_count"] == 1
    assert metrics["stage0.url_registry.records.count"] == 1
    assert metrics["stage0.url_registry.matched_urls.count"] == 1
    assert metrics["stage0.url_registry.unmatched_urls.count"] == 1
    assert metrics["stage0.url_registry.content_hash.count"] == 1
    assert metrics["stage0.url_registry.etag.count"] == 1
    assert metrics["stage0.url_registry.last_modified.count"] == 1
    assert run_context["inputs"]["url_registry_path"] == str(registry_path)
    assert "url_registry.json" in {item["artifact_path"] for item in artifacts["artifacts"]}
