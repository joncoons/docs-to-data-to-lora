"""Tests for NeMo Platform Model Entity readiness planning."""

import json

from scripts.stage3.platform_customizer import model_entity_for_base
from scripts.stage3.platform_model_entities import (
    DEFAULT_MODEL_SPECS,
    PlatformModelEntityConfig,
    build_model_entity_plan,
    run,
    selected_model_specs,
)


def test_default_specs_cover_stage3_training_bases():
    assert set(DEFAULT_MODEL_SPECS) == {
        "meta/llama-3.2-1b-instruct",
        "meta/llama-3.2-3b-instruct",
        "meta/llama-3.1-8b-instruct",
        "nvidia/nemotron-3-nano-30b-a3b",
    }


def test_model_entity_plan_matches_customizer_model_ref():
    spec = DEFAULT_MODEL_SPECS["meta/llama-3.2-3b-instruct"]
    plan = build_model_entity_plan(spec, workspace="default")

    assert plan["model_entity_ref"] == "default/llama-3.2-3b-instruct"
    assert plan["customizer_model_ref"] == model_entity_for_base(
        "meta/llama-3.2-3b-instruct",
        "default",
    )
    assert plan["fileset_uri"] == "fileset://default/llama-3.2-3b-instruct"
    assert plan["source"]["repo_id"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert plan["source"]["requires_token"] is True


def test_selected_specs_attach_hf_token_secret():
    specs = selected_model_specs(
        ["meta/llama-3.2-1b-instruct"],
        hf_token_secret="stage3-hf-token",
    )

    assert specs[0].token_secret == "stage3-hf-token"


def test_run_dry_run_writes_manifest(tmp_path):
    out = tmp_path / "platform_model_entities_manifest.json"
    manifest = run(
        PlatformModelEntityConfig(
            workspace="default",
            nmp_base_url="http://nemo-platform-api:8080",
            models=("nvidia/nemotron-3-nano-30b-a3b",),
            out=out,
            dry_run=True,
        )
    )

    assert manifest["dry_run"] is True
    assert manifest["schema_version"] == "platform_model_entities.v1"
    written = json.loads(out.read_text())
    model = written["models"][0]
    assert model["model_entity_ref"] == "default/nemotron-3-nano-30b-a3b"
    assert model["fileset_uri"] == "fileset://default/nemotron-3-nano-30b-a3b"
    assert model["source"]["repo_id"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
