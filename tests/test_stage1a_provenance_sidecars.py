"""Tests for Stage 1A provenance sidecar emission."""

import json
from unittest.mock import MagicMock

from scripts.pipeline.models import Passage
from scripts.pipeline.stage1a_le_kvp import (
    build_stage1a_observability_documents,
    run_stage1a,
    write_observability_documents,
)


def test_run_stage1a_writes_entailment_sidecar(tmp_output_dir):
    passage = Passage(
        passage_id="https://docs.example.com/p#p0",
        url="https://docs.example.com/p",
        text="NIM can serve models through an OpenAI-compatible endpoint. " * 20,
        token_count=120,
        chunk_ids=["es-1"],
        product_family="nim",
        product_name="nim-llm",
        doc_kind="html",
    )
    llm = MagicMock()
    llm.call.side_effect = [
        json.dumps({
            "conclusion": "NIM can serve models through an OpenAI-compatible endpoint.",
            "premises": ["NIM exposes compatible inference endpoints."],
        }),
        json.dumps({
            "question": "What endpoint style can NIM expose?",
            "answer": "NIM can expose an OpenAI-compatible endpoint.",
        }),
    ]

    rows = run_stage1a([passage], llm, tmp_output_dir, max_workers=1)

    sidecar = tmp_output_dir / "provenance" / "entailments.jsonl"
    assert len(rows) == 1
    assert sidecar.exists()
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["entailment_id"] == rows[0].entailment_id
    assert records[0]["evidence"][0]["chunk_id"].startswith("chunk_")



def test_run_stage1a_writes_sharded_sidecar_and_observability(tmp_output_dir):
    passage = Passage(
        passage_id="https://docs.example.com/p#p0",
        url="https://docs.example.com/p",
        text="NIM can serve models through an OpenAI-compatible endpoint. " * 20,
        token_count=120,
        chunk_ids=["es-1"],
        product_family="nim",
        product_name="nim-llm",
        doc_kind="html",
    )
    input_path = tmp_output_dir / "passages.jsonl"
    input_path.write_text(passage.model_dump_json() + "\n")

    llm = MagicMock()
    llm.model = "test-model"
    llm.temperature = 0.1
    llm.call.side_effect = [
        json.dumps({
            "conclusion": "NIM can serve models through an OpenAI-compatible endpoint.",
            "premises": ["NIM exposes compatible inference endpoints."],
        }),
        json.dumps({
            "question": "What endpoint style can NIM expose?",
            "answer": "NIM can expose an OpenAI-compatible endpoint.",
        }),
    ]

    output_name = "stage1a_le.shard-00000-of-00002.jsonl"
    entailments_name = "entailments.shard-00000-of-00002.jsonl"
    rows = run_stage1a(
        [passage],
        llm,
        tmp_output_dir,
        max_workers=1,
        output_filename=output_name,
        entailments_filename=entailments_name,
    )
    output_file = tmp_output_dir / output_name
    entailments_file = tmp_output_dir / "provenance" / entailments_name
    docs = build_stage1a_observability_documents(
        input_passages_path=input_path,
        output_dir=tmp_output_dir,
        output_file=output_file,
        entailments_file=entailments_file,
        input_passage_count=1,
        selected_passage_count=1,
        rows=rows,
        shard_index=0,
        shard_count=2,
        max_workers=1,
        llm_model="test-model",
        llm_endpoints=["http://nim:8000/v1"],
        pipeline_run_id="pipeline-run-1",
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="docs-to-data-to-lora",
        mlflow_parent_run_id="parent-run-1",
    )
    obs_dir = tmp_output_dir / "observability"
    write_observability_documents(obs_dir, docs)

    assert output_file.exists()
    assert entailments_file.exists()
    metrics = json.loads((obs_dir / "metrics.json").read_text())
    run_context = json.loads((obs_dir / "run_context.json").read_text())
    service_refs = json.loads((obs_dir / "service_refs.json").read_text())
    artifacts = json.loads((obs_dir / "artifacts_manifest.json").read_text())

    assert metrics["stage1a.passages.input.count"] == 1
    assert metrics["stage1a.passages.selected.count"] == 1
    assert metrics["stage1a.rows.count"] == 1
    assert metrics["stage1a.entailments.count"] == 1
    assert run_context["shard"]["label"] == "shard-00000-of-00002"
    assert service_refs["services"]["llm"]["model"] == "test-model"
    assert {item["artifact_kind"] for item in artifacts["artifacts"]} == {
        "input_passages",
        "stage1a_rows",
        "entailment_provenance",
    }
