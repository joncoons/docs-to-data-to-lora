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
