"""Sidecar-sync LoRA adapters from NeMo Data Store to NFS per-base dirs.

NIM 2.0.x requires NIM_PEFT_SOURCE to be a filesystem path with per-base
sub-directories. This module pulls the two HF-PEFT files (safetensors +
config) from the data-store's HF-compatible API.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

_BASE_DIR_FOR_MODEL = {
    "meta/llama-3.2-1b-instruct": "lora-llama-3.2-1b",
    "meta/llama-3.2-3b-instruct": "lora-llama-3.2-3b",
    "meta/llama-3.1-8b-instruct": "lora-llama-3.1-8b",
}


@dataclass(frozen=True)
class AdapterMeta:
    name: str         # adapter directory name (= NIM model identifier)
    base_model: str   # canonical base slug, e.g. meta/llama-3.2-3b-instruct
    job_id: str       # Customizer job id; used as the HF revision tag


def per_base_dir(base_model: str, root: Path) -> Path:
    sub = _BASE_DIR_FOR_MODEL.get(base_model)
    if sub is None:
        raise ValueError(f"Unknown base: {base_model!r}")
    return root / sub


def sync_adapter(meta: AdapterMeta, root: Path,
                 datastore_url: str = "http://nemo-data-store:3000",
                 timeout: float = 60.0) -> Path:
    """Download adapter_model.safetensors + adapter_config.json into per-base dir.

    Returns the local directory holding the two files.
    """
    out_dir = per_base_dir(meta.base_model, root) / meta.name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = (
        f"{datastore_url.rstrip('/')}/v1/hf/default/{meta.name}/resolve/{meta.job_id}"
    )
    for fname in ("adapter_model.safetensors", "adapter_config.json"):
        resp = httpx.get(f"{base_url}/{fname}", timeout=timeout)
        resp.raise_for_status()
        (out_dir / fname).write_bytes(resp.content)
    return out_dir
