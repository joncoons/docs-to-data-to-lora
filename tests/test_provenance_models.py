"""Tests for provenance model construction helpers."""

from scripts.pipeline.models import KVPRow, Passage
from scripts.pipeline.provenance import (
    attach_source_provenance,
    dataset_sample_from_kvp_row,
    entailments_from_kvp_rows,
    entailment_id_for_passage,
    sha256_text,
)


def _passage() -> Passage:
    return Passage(
        passage_id="https://docs.example.com/x#p0",
        url="https://docs.example.com/x",
        text="NIM supports model deployment. " * 20,
        token_count=100,
        chunk_ids=["es-1", "es-2"],
        product_family="nim",
        product_name="nim-llm",
        doc_kind="html",
    )


def test_attach_source_provenance_adds_revision_and_chunk_ids():
    updated, revisions, chunks = attach_source_provenance(
        [_passage()],
        crawl_run_id="crawlrun_test",
        retrieved_at="2026-05-28T00:00:00Z",
    )

    assert len(updated) == 1
    assert len(revisions) == 1
    assert len(chunks) == 1
    assert updated[0].source_revision_id == revisions[0].source_revision_id
    assert updated[0].source_chunk_ids == [chunks[0].chunk_id]
    assert chunks[0].metadata["external_chunk_ids"] == ["es-1", "es-2"]


def test_entailments_from_kvp_rows_deduplicates_by_entailment_id():
    passage = _passage()
    updated, _, _ = attach_source_provenance(
        [passage],
        crawl_run_id="crawlrun_test",
        retrieved_at="2026-05-28T00:00:00Z",
    )
    passage = updated[0]
    ent_id = entailment_id_for_passage(
        passage,
        0,
        "NIM supports model deployment.",
        ["NIM docs describe model deployment."],
    )
    row = KVPRow(
        passage_id=passage.passage_id,
        source_url=passage.url,
        product_family=passage.product_family,
        stage="1a",
        question="What does NIM support?",
        answer="NIM supports model deployment.",
        context=passage.text,
        entailment_id=ent_id,
        entailment_claim="NIM supports model deployment.",
        entailment_premises=["NIM docs describe model deployment."],
        source_revision_ids=[passage.source_revision_id],
        source_chunk_ids=passage.source_chunk_ids,
        extractor_model="test-model",
        extractor_prompt_hash=sha256_text("prompt"),
    )

    entailments = entailments_from_kvp_rows([row, row])
    assert len(entailments) == 1
    assert entailments[0].entailment_id == ent_id
    assert entailments[0].evidence[0]["chunk_id"] == passage.source_chunk_ids[0]


def test_dataset_sample_source_and_synthetic_lineage():
    source_row = KVPRow(
        passage_id="p0",
        source_url="https://docs.example.com/x",
        product_family="nim",
        stage="1b",
        qa_type="bridging",
        question="How do A and B connect?",
        answer="A depends on B.",
        context="ctx",
        source_revision_ids=["srcrev_abc"],
        source_chunk_ids=["chunk_abc"],
        source_systems=["web_crawl"],
        source_kinds=["web_page"],
        modalities=["text"],
    )
    source_sample = dataset_sample_from_kvp_row(source_row, system_prompt="system")
    assert source_sample.origin == "source_entailed"
    assert source_sample.task_type == "bridging"
    assert source_sample.lineage["source_chunk_ids"] == ["chunk_abc"]
    assert source_sample.lineage["source_systems"] == ["web_crawl"]
    assert source_sample.metadata["modalities"] == ["text"]
    assert source_sample.system == "system"

    synthetic_row = KVPRow(
        passage_id="gapfill#nim",
        source_url="<multi-chunk retrieval>",
        product_family="nim",
        stage="1.5",
        target_product_family="nim",
        question="Gap question?",
        answer="Gap answer.",
        context="ctx",
    )
    synthetic_sample = dataset_sample_from_kvp_row(synthetic_row)
    assert synthetic_sample.origin == "synthetic_gapfill"
    assert synthetic_sample.lineage["gap_id"].startswith("gap_")
    assert synthetic_sample.lineage["data_designer_job_id"] == "legacy_direct_llm_gapfill"
