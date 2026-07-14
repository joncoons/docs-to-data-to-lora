"""Tests for final dataset sample admission and service lineage overlays."""

import json

from scripts.pipeline.dataset_admission import admitted_dataset_samples_from_kvp_rows
from scripts.pipeline.models import KVPRow


def _synthetic_row(sample_id="sample_dd"):
    return KVPRow(
        passage_id="gapfill#gap_nim",
        source_url="<data-designer-synthetic>",
        product_family="nim",
        stage="1.5",
        target_product_family="nim",
        sample_id=sample_id,
        question="Refined question?",
        answer="Refined answer.",
        context="NIM documentation context.",
        refined=True,
        source_chunk_ids=["chunk_from_row"],
        source_systems=["data_designer"],
        source_kinds=["synthetic_gapfill"],
        modalities=["text"],
        qa_work_id="stage2qa_123",
        qa_status="refined",
        qa_admitted=True,
        qa_judge_model="nvidia/nvidia/nemotron-3-ultra",
        qa_judge_endpoints=["https://inference-api.nvidia.com/v1"],
        qa_execution_surface="curator_llm_quality",
        qa_grounded=True,
        qa_answer_fidelity=True,
        qa_no_hallucination=True,
        qa_repairable=True,
        qa_reason="grounded after rewrite",
    )


def test_admitted_samples_preserve_data_designer_sidecar_lineage(tmp_path):
    dataset_dir = tmp_path / "nim_curated"
    provenance_dir = dataset_dir / "provenance"
    provenance_dir.mkdir(parents=True)
    sidecar = {
        "schema_version": "provenance.v1",
        "sample_id": "sample_dd",
        "origin": "synthetic_gapfill",
        "task_type": "qa",
        "prompt": "Original generated question?",
        "completion": "Original generated answer.",
        "system": None,
        "lineage": {
            "entailment_ids": ["ent_seed"],
            "source_revision_ids": [],
            "source_chunk_ids": ["chunk_seed"],
            "source_systems": ["data_designer"],
            "source_kinds": ["synthetic_gapfill"],
            "modalities": ["text"],
            "gap_id": "gap_real",
            "data_designer_job_id": "dd_job_123",
            "seed_sample_ids": ["sample_seed"],
        },
        "quality": {
            "curator_job_id": None,
            "judge_model": None,
            "grounding_score": None,
        },
        "metadata": {
            "collection": "nim_curated",
            "product_family": "nim",
            "source_url": "<data-designer-synthetic>",
            "gap_manifest_id": "gapmanifest_123",
            "recipe_name": "nim-gapfill",
            "record_index": 0,
            "pair_index": 0,
        },
    }
    (provenance_dir / "data_designer_samples.jsonl").write_text(json.dumps(sidecar) + "\n")

    samples = admitted_dataset_samples_from_kvp_rows(
        [_synthetic_row()],
        dataset_dir=dataset_dir,
        system_prompt="System prompt.",
    )
    sample = samples[0].model_dump(mode="json")

    assert sample["prompt"] == "Refined question?"
    assert sample["completion"] == "Refined answer."
    assert sample["system"] == "System prompt."
    assert sample["lineage"]["gap_id"] == "gap_real"
    assert sample["lineage"]["data_designer_job_id"] == "dd_job_123"
    assert sample["lineage"]["seed_sample_ids"] == ["sample_seed"]
    assert sample["metadata"]["refined"] is True
    assert sample["quality"]["judge_model"] == "nvidia/nvidia/nemotron-3-ultra"
    assert sample["quality"]["qa_status"] == "refined"
    assert sample["quality"]["qa_execution_surface"] == "curator_llm_quality"
    assert sample["quality"]["grounding_score"] == 1.0
    assert sample["quality"]["reason"] == "grounded after rewrite"
    assert sample["metadata"]["gap_manifest_id"] == "gapmanifest_123"
    assert sample["metadata"]["recipe_name"] == "nim-gapfill"


def test_admitted_samples_fall_back_to_row_lineage_without_sidecar(tmp_path):
    samples = admitted_dataset_samples_from_kvp_rows(
        [_synthetic_row("sample_without_sidecar")],
        dataset_dir=tmp_path / "nim_curated",
    )

    assert samples[0].origin == "synthetic_gapfill"
    assert samples[0].lineage["data_designer_job_id"] == "legacy_direct_llm_gapfill"
