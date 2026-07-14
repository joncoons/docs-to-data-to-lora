"""Tests for config module."""
from scripts.pipeline.config import Config


def test_config_defaults():
    cfg = Config()
    assert cfg.es_host.startswith("https://")
    assert cfg.nim_endpoints  # at least one
    assert cfg.external_judge_model == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    assert cfg.stage2_qa_endpoints == ["http://llm-judge.default.svc.cluster.local:8000/v1"]
    assert cfg.stage2_qa_model == "nvidia/nvidia/nemotron-3-super-v3"
    assert cfg.stage2_qa_temperature == 0.0
    assert cfg.stage2_execution_surface == "curator_llm_quality"
    assert cfg.super120b_model == "nvidia/nemotron-3-super-120b-a12b"


def test_config_output_dir_per_collection(tmp_path):
    cfg = Config(base_output_dir=tmp_path)
    assert cfg.output_dir_for("nim_curated") == tmp_path / "nim_curated"
    assert cfg.output_dir_for("nemo_usvcs_curated") == tmp_path / "nemo_usvcs_curated"


def test_config_thresholds():
    cfg = Config()
    assert cfg.min_passage_tokens == 60
    assert cfg.train_val_split == 0.90
    assert cfg.stage1c_selection_mode == "stratified"
    assert cfg.stage1c_top_percent == 0.25
    assert cfg.stage1c_min_passages == 100
    assert cfg.bias_threshold_factor == 0.5
    assert cfg.bias_gapfill_target_factor == 0.8
