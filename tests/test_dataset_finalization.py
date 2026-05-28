"""Tests for dataset finalization manifests."""

import json

from scripts.pipeline.finalize_dataset import (
    build_dataset_version_manifest,
    finalize_dataset,
)
from scripts.pipeline.models import KVPRow


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _source_row(question="What does NIM expose?"):
    return KVPRow(
        passage_id="p0",
        source_url="https://docs.example.com/nim",
        product_family="nim",
        stage="1a",
        question=question,
        answer="NIM exposes OpenAI-compatible endpoints.",
        context="NIM exposes OpenAI-compatible endpoints.",
        entailment_id="ent_web",
        source_revision_ids=["srcrev_web"],
        source_chunk_ids=["chunk_web"],
        source_systems=["web_crawl"],
        source_kinds=["web_page"],
        modalities=["text"],
    )


def _dataset_sample(sample_id, origin, systems, modalities):
    return {
        "schema_version": "provenance.v1",
        "sample_id": sample_id,
        "origin": origin,
        "task_type": "qa",
        "prompt": "Prompt?",
        "completion": "Completion.",
        "system": None,
        "lineage": {
            "entailment_ids": ["ent_web"] if origin == "source_entailed" else [],
            "source_revision_ids": ["srcrev_web"] if origin == "source_entailed" else [],
            "source_chunk_ids": ["chunk_web"] if origin == "source_entailed" else [],
            "source_systems": systems,
            "source_kinds": ["web_page"] if systems == ["web_crawl"] else ["synthetic_gapfill"],
            "modalities": modalities,
            "gap_id": None if origin == "source_entailed" else "gap_nim",
            "data_designer_job_id": None if origin == "source_entailed" else "dd_job_1",
            "seed_sample_ids": [],
        },
        "quality": {},
        "metadata": {
            "source_url": (
                "https://docs.example.com/nim"
                if origin == "source_entailed"
                else "<synthetic>"
            ),
        },
    }


def test_finalize_dataset_writes_manifest_and_observability(tmp_path):
    dataset_dir = tmp_path / "nim_curated"
    dataset_dir.mkdir()
    _write_jsonl(dataset_dir / "training.jsonl", [{"prompt": "t1", "completion": "a1"}])
    _write_jsonl(dataset_dir / "validation.jsonl", [{"prompt": "v1", "completion": "a1"}])
    _write_jsonl(dataset_dir / "test_set.jsonl", [{"prompt": "x1", "completion": "a1"}])
    _write_jsonl(
        dataset_dir / "provenance" / "dataset_samples.jsonl",
        [
            _dataset_sample("sample_web", "source_entailed", ["web_crawl"], ["text"]),
            _dataset_sample("sample_dd", "synthetic_gapfill", ["data_designer"], ["text"]),
        ],
    )
    (dataset_dir / "manifests").mkdir()
    (dataset_dir / "manifests" / "crawl_run.json").write_text(json.dumps({
        "crawl_run_id": "crawlrun_abc",
    }))
    (dataset_dir / "test_kvp_uids.json").write_text(json.dumps({"seed": 123}))

    observability_dir = tmp_path / "observability"
    manifest = finalize_dataset(
        dataset_dir,
        dataset_name="nim_curated",
        observability_dir=observability_dir,
        created_at="2026-05-28T12:00:00Z",
    )

    manifest_path = dataset_dir / "manifests" / "dataset_version_manifest.json"
    assert manifest_path.exists()
    assert manifest["dataset_version_id"].startswith("dsv_")
    assert manifest["dataset_role"] == "blended"
    assert manifest["inputs"]["crawl_run_ids"] == ["crawlrun_abc"]
    assert manifest["inputs"]["data_designer_job_ids"] == ["dd_job_1"]
    assert manifest["split"]["method"] == "stage3-holdout"
    assert manifest["split"]["seed"] == 123
    assert manifest["metrics"]["rows"]["training"] == 1
    assert manifest["metrics"]["samples"]["provenance"] == 2
    assert manifest["metrics"]["samples"]["synthetic"] == 1
    assert manifest["source_composition"]["source_systems"] == {
        "data_designer": 1,
        "web_crawl": 1,
    }
    assert "training.jsonl" in manifest["outputs"]["hashes"]

    metrics = json.loads((observability_dir / "metrics.json").read_text())
    service_refs = json.loads((observability_dir / "service_refs.json").read_text())
    assert metrics["dataset.rows.training"] == 1
    assert metrics["dataset.samples.synthetic.count"] == 1
    assert metrics["dataset.source_system.web_crawl.samples"] == 1
    assert service_refs["dataset_version"]["dataset_version_id"] == manifest["dataset_version_id"]


def test_dataset_version_id_is_independent_of_created_at(tmp_path):
    dataset_dir = tmp_path / "nim_curated"
    dataset_dir.mkdir()
    _write_jsonl(dataset_dir / "training.jsonl", [{"prompt": "t1", "completion": "a1"}])
    _write_jsonl(dataset_dir / "validation.jsonl", [])
    _write_jsonl(dataset_dir / "test_set.jsonl", [])
    _write_jsonl(
        dataset_dir / "provenance" / "dataset_samples.jsonl",
        [_dataset_sample("sample_web", "source_entailed", ["web_crawl"], ["text"])],
    )

    first, _ = build_dataset_version_manifest(
        dataset_dir,
        dataset_name="nim_curated",
        created_at="2026-05-28T12:00:00Z",
    )
    second, _ = build_dataset_version_manifest(
        dataset_dir,
        dataset_name="nim_curated",
        created_at="2026-05-29T12:00:00Z",
    )

    assert first["dataset_version_id"] == second["dataset_version_id"]
    assert first["created_at"] != second["created_at"]


def test_finalize_dataset_backfills_missing_dataset_samples(tmp_path):
    dataset_dir = tmp_path / "nim_curated"
    dataset_dir.mkdir()
    _write_jsonl(dataset_dir / "training.jsonl", [{"prompt": "t1", "completion": "a1"}])
    _write_jsonl(dataset_dir / "validation.jsonl", [])
    _write_jsonl(dataset_dir / "test_set.jsonl", [])
    stage2_row = _source_row()
    (dataset_dir / "stage2_eval.jsonl").write_text(stage2_row.model_dump_json() + "\n")

    manifest = finalize_dataset(
        dataset_dir,
        dataset_name="nim_curated",
        created_at="2026-05-28T12:00:00Z",
    )

    samples_path = dataset_dir / "provenance" / "dataset_samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    assert len(samples) == 1
    assert samples[0]["lineage"]["source_systems"] == ["web_crawl"]
    assert manifest["source_composition"]["source_systems"] == {"web_crawl": 1}
