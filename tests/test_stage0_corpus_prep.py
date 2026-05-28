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
