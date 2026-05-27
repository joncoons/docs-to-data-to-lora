"""Tests for adapter sync logic — data-store → NFS per-base dir."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.eval.adapter_sync import (
    AdapterMeta,
    per_base_dir,
    sync_adapter,
)


# --- per_base_dir mapping --------------------------------------------------

def test_per_base_dir_maps_1b(tmp_path):
    assert per_base_dir("meta/llama-3.2-1b-instruct", tmp_path) == \
        tmp_path / "lora-llama-3.2-1b"


def test_per_base_dir_maps_3b(tmp_path):
    assert per_base_dir("meta/llama-3.2-3b-instruct", tmp_path) == \
        tmp_path / "lora-llama-3.2-3b"


def test_per_base_dir_maps_8b(tmp_path):
    assert per_base_dir("meta/llama-3.1-8b-instruct", tmp_path) == \
        tmp_path / "lora-llama-3.1-8b"


def test_per_base_dir_raises_for_unknown_base(tmp_path):
    with pytest.raises(ValueError, match="Unknown base"):
        per_base_dir("openai/gpt-4", tmp_path)


# --- sync_adapter file ops -------------------------------------------------

def test_sync_adapter_writes_safetensors_and_config(tmp_path):
    """Mock httpx to return fake adapter bytes; verify files written."""
    meta = AdapterMeta(
        name="lora-nim-llama-3.2-3b-r16",
        base_model="meta/llama-3.2-3b-instruct",
        job_id="cust-test123",
    )
    fake_safetensors = b"\x00\x01\x02\x03" * 10
    fake_config = b'{"adapter_dim": 16}'

    def fake_get(url, timeout):
        resp = MagicMock()
        if url.endswith("adapter_model.safetensors"):
            resp.content = fake_safetensors
        elif url.endswith("adapter_config.json"):
            resp.content = fake_config
        else:
            raise AssertionError(f"unexpected url: {url}")
        resp.raise_for_status = MagicMock()
        return resp

    with patch("scripts.eval.adapter_sync.httpx.get", side_effect=fake_get):
        out = sync_adapter(meta, root=tmp_path,
                           datastore_url="http://test:3000")

    assert out == tmp_path / "lora-llama-3.2-3b" / "lora-nim-llama-3.2-3b-r16"
    assert (out / "adapter_model.safetensors").read_bytes() == fake_safetensors
    assert (out / "adapter_config.json").read_bytes() == fake_config


def test_sync_adapter_url_includes_job_id_revision(tmp_path):
    """The HF resolve URL must include the job_id as the revision."""
    meta = AdapterMeta(
        name="lora-nim-llama-3.2-3b-r16",
        base_model="meta/llama-3.2-3b-instruct",
        job_id="cust-test123",
    )
    captured = []

    def fake_get(url, timeout):
        captured.append(url)
        resp = MagicMock(content=b"x")
        resp.raise_for_status = MagicMock()
        return resp

    with patch("scripts.eval.adapter_sync.httpx.get", side_effect=fake_get):
        sync_adapter(meta, root=tmp_path, datastore_url="http://ds:3000")

    assert any("/resolve/cust-test123/" in u for u in captured), \
        f"job_id not in any URL: {captured}"
    assert any(u.endswith("adapter_model.safetensors") for u in captured)
    assert any(u.endswith("adapter_config.json") for u in captured)


def test_sync_adapter_propagates_http_error(tmp_path):
    """A non-2xx response must surface as httpx.HTTPStatusError."""
    import httpx as _httpx

    meta = AdapterMeta(
        name="lora-nim-llama-3.2-3b-r16",
        base_model="meta/llama-3.2-3b-instruct",
        job_id="cust-test123",
    )

    def fake_get(url, timeout):
        request = _httpx.Request("GET", url)
        resp = _httpx.Response(404, request=request)
        return resp

    with patch("scripts.eval.adapter_sync.httpx.get", side_effect=fake_get):
        with pytest.raises(_httpx.HTTPStatusError):
            sync_adapter(meta, root=tmp_path, datastore_url="http://ds:3000")
