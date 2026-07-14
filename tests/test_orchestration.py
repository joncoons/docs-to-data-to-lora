"""Smoke test the CLI parses args correctly."""
import argparse
import subprocess
import sys

from scripts.pipeline.config import Config
import scripts.build_v2_dataset as build_v2


def test_cli_dry_run(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/build_v2_dataset.py",
         "--collection", "nim_curated",
         "--output", str(tmp_path),
         "--stage", "0",
         "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "DRY RUN" in result.stderr or "DRY RUN" in result.stdout


def test_cli_dry_run_accepts_batched_stage1a_mode(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/build_v2_dataset.py",
         "--collection", "nim_curated",
         "--output", str(tmp_path),
         "--stage", "0",
         "--stage1a-mode", "batched",
         "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "DRY RUN" in result.stderr or "DRY RUN" in result.stdout


def test_stage1a_batched_llm_uses_super_defaults():
    cfg = Config()
    args = argparse.Namespace(
        stage1a_mode="batched",
        stage1a_nim_endpoints=None,
        stage1a_model=None,
        stage1a_api_key=None,
        stage1a_temperature=None,
    )

    llm = build_v2._build_stage1a_llm(args, cfg)

    assert llm.endpoints == cfg.nim_endpoints
    assert llm.model == cfg.super120b_model
    assert llm.temperature == 0.95


def test_stage1a_legacy_llm_uses_existing_config_defaults():
    cfg = Config()
    args = argparse.Namespace(
        stage1a_mode="legacy",
        stage1a_nim_endpoints=None,
        stage1a_model=None,
        stage1a_api_key=None,
        stage1a_temperature=None,
    )

    llm = build_v2._build_stage1a_llm(args, cfg)

    assert llm.endpoints == cfg.nim_endpoints
    assert llm.model == cfg.super120b_model
    assert llm.temperature == 0.2
