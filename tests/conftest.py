"""Shared pytest fixtures for the pipeline test suite."""
import json
from pathlib import Path
import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_chunks(fixtures_dir) -> list[dict]:
    with open(fixtures_dir / "sample_chunks.json") as f:
        return json.load(f)


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    out = tmp_path / "v2_output"
    out.mkdir()
    return out
