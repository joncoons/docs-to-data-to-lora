"""Tests for Curator Kubernetes execution templates and configs."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_curator_base_image_is_pinned_to_ngc_release():
    containerfile = REPO_ROOT / "deploy" / "curator" / "Containerfile"
    content = containerfile.read_text()

    assert "ARG NEMO_CURATOR_IMAGE=nvcr.io/nvidia/nemo-curator:26.04" in content


def test_native_filter_job_writes_collect_compatible_paths():
    job_path = REPO_ROOT / "deploy" / "curator" / "native-filter-job.yaml"
    job = list(yaml.safe_load_all(job_path.read_text()))[0]

    container = job["spec"]["template"]["spec"]["containers"][0]
    command_text = "\n".join(container["args"])

    assert container["image"] == "<registry>/docs-to-data-to-lora/curator:latest"
    assert "filter_documents" in command_text
    assert '--input-data-dir="${CURATOR_DIR}/input"' in command_text
    assert '--output-retained-document-dir="${CURATOR_DIR}/retained"' in command_text
    assert '--output-removed-document-dir="${CURATOR_DIR}/removed"' in command_text
    assert '--output-document-score-dir="${CURATOR_DIR}/scores"' in command_text


def test_filter_documents_config_uses_curator_cli_shape():
    config_path = REPO_ROOT / "configs" / "curator" / "sft-filter-documents.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["input_field"] == "text"
    filters = config["filters"]
    assert [item["name"] for item in filters] == ["ScoreFilter"] * 4
    assert [item["filter"]["name"] for item in filters] == [
        "WordCountFilter",
        "NonAlphaNumericFilter",
        "PunctuationFilter",
        "RepeatingTopNGramsFilter",
    ]
    assert filters[0]["text_field"] == "text"
    assert filters[0]["score_field"] == "word_count"


def test_handoff_plan_references_cli_filter_config():
    plan_path = REPO_ROOT / "configs" / "curator" / "sft-dedup-quality.yaml"
    plan = yaml.safe_load(plan_path.read_text())

    assert plan["quality_filter_config"] == (
        "/workspace/configs/curator/sft-filter-documents.yaml"
    )
    assert plan["deduplication_plan"]["exact"]["enabled"] is True
    assert plan["deduplication_plan"]["fuzzy"]["enabled"] is True
    assert plan["deduplication_plan"]["semantic"]["enabled"] is False
