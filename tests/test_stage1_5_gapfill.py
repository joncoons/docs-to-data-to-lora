"""Tests for Stage 1.5: bias analysis + Data Designer handoff."""
import json
import math
from unittest.mock import MagicMock

from scripts.pipeline.models import KVPRow, Passage
from scripts.pipeline.stage1_5_gapfill import (
    build_data_designer_requests,
    build_gap_manifest,
    build_seed_styles,
    compute_bias_report,
    render_recipe_prompt,
    run_stage1_5,
    run_stage1_5_analysis,
)


def _row(product_family, q="Q?", a="A.", entailment_id=None):
    return KVPRow(
        passage_id="x#0", source_url="https://x.com",
        product_family=product_family, stage="1a", premise_index=0,
        question=q, answer=a, context="ctx", refined=False,
        entailment_id=entailment_id,
    )


def _passage(product_family):
    return Passage(
        passage_id=f"p_{product_family}", url="https://x.com",
        text="text body " * 50, token_count=100,
        chunk_ids=["1"], product_family=product_family, product_name=product_family,
        doc_kind="html",
    )


def test_compute_bias_report_flags_underrepresented():
    # 100 chunks for product A → many KVPs
    # 100 chunks for product B → few KVPs (under-represented)
    passages = (
        [_passage("A") for _ in range(100)] +
        [_passage("B") for _ in range(100)]
    )
    kvps = (
        [_row("A") for _ in range(200)] +   # density 2.0
        [_row("B") for _ in range(20)]      # density 0.2
    )
    report = compute_bias_report(passages, kvps, threshold_factor=0.5)
    by_family = {p["product_family"]: p for p in report["products"]}
    assert by_family["A"]["underrepresented"] is False
    assert by_family["B"]["underrepresented"] is True
    assert report["median_density"] > 0


def test_build_seed_styles_picks_three():
    rows = [_row("A", q=f"Q{i}?", a=f"A{i}.") for i in range(10)]
    styles = build_seed_styles(rows, product_family="A", n=3)
    assert len(styles) == 3


def test_render_recipe_prompt_substitutes_jinja_vars():
    prompt = render_recipe_prompt(
        retrieved_chunks="CHUNK1\nCHUNK2",
        product_family="nim-deploy",
        seed_styles=["- Q1?", "- Q2?", "- Q3?"],
    )
    assert "CHUNK1" in prompt
    assert "nim-deploy" in prompt
    assert "Q1?" in prompt



def _biased_inputs():
    passages = [_passage("A") for _ in range(100)] + [_passage("B") for _ in range(100)]
    kvps = [_row("A") for _ in range(200)] + [
        _row("B", q=f"QB{i}?", a=f"AB{i}.", entailment_id=f"ent_b_{i}")
        for i in range(20)
    ]
    report = compute_bias_report(passages, kvps, threshold_factor=0.5)
    return passages, kvps, report


def _fake_es():
    es = MagicMock()
    es.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "text": "B product documentation chunk",
                        "metadata": {
                            "content_metadata": {
                                "content_url": "https://docs.example.com/b"
                            }
                        },
                    }
                }
            ]
        }
    }
    return es


def test_build_gap_manifest_records_underrepresented_product():
    passages, kvps, report = _biased_inputs()

    manifest = build_gap_manifest(
        passages,
        kvps,
        report,
        collection="nim_curated",
        dataset_version_id="dsv_test",
        target_factor=0.8,
    )

    assert manifest["schema_version"] == "provenance.v1"
    assert manifest["gap_manifest_id"].startswith("gapmanifest_")
    assert manifest["dataset_version_id"] == "dsv_test"
    assert len(manifest["gaps"]) == 1
    gap = manifest["gaps"][0]
    assert gap["gap_id"].startswith("gap_")
    assert gap["dimension"]["product"] == "B"
    assert gap["recommendation"] == "generate_synthetic"
    assert gap["observed_count"] == 20
    assert gap["target_count"] == math.ceil(report["median_density"] * 0.8 * 100)
    assert gap["seed_entailment_ids"][:2] == ["ent_b_0", "ent_b_1"]
    assert gap["seed_chunk_ids"] == ["1"]


def test_build_data_designer_requests_materializes_retrieved_seed_inputs():
    passages, kvps, report = _biased_inputs()
    manifest = build_gap_manifest(
        passages,
        kvps,
        report,
        collection="nim_curated",
        dataset_version_id="dsv_test",
        target_factor=0.8,
    )

    requests = build_data_designer_requests(
        manifest,
        kvps,
        _fake_es(),
        "nim_curated",
        pairs_per_call=5,
    )

    assert len(requests) == 1
    request = requests[0]
    assert request["recipe_name"] == "nim-gapfill"
    assert request["gap_manifest_id"] == manifest["gap_manifest_id"]
    assert request["input"]["product_family"] == "B"
    assert request["input"]["pairs_count"] == 5
    assert "B product documentation chunk" in request["input"]["retrieved_chunks"]
    assert request["retrieved_urls"] == ["https://docs.example.com/b"]
    assert request["num_records"] == math.ceil(request["pairs_needed"] / 5)


def test_run_stage1_5_analysis_writes_gap_manifest_and_data_designer_plan(tmp_path):
    passages, kvps, _ = _biased_inputs()

    result = run_stage1_5_analysis(
        passages,
        kvps,
        _fake_es(),
        "nim_curated",
        tmp_path,
        target_factor=0.8,
    )

    assert len(result["gap_manifest"]["gaps"]) == 1
    assert (tmp_path / "bias_report.json").exists()
    assert (tmp_path / "provenance" / "gap_manifest.json").exists()
    assert (tmp_path / "data_designer" / "gapfill_requests.jsonl").exists()
    request_manifest = json.loads((tmp_path / "data_designer" / "request_manifest.json").read_text())
    assert request_manifest["status"] == "planned"
    assert request_manifest["request_count"] == 1


def test_run_stage1_5_defaults_to_data_designer_handoff_without_llm_calls(tmp_path):
    passages, kvps, _ = _biased_inputs()
    rows = run_stage1_5(passages, kvps, _fake_es(), "nim_curated", tmp_path)

    assert rows == []
    assert (tmp_path / "stage1_5_gapfill.jsonl").read_text() == ""
    assert (tmp_path / "provenance" / "gap_manifest.json").exists()
