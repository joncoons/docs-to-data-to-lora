"""Dataset sample admission helpers.

These helpers turn accepted KVP rows into final dataset-sample provenance while
preserving richer sidecar lineage from native services such as NeMo Data
Designer. The KVP row remains the QA payload source of truth after Stage 2
refinement; sidecars supply service/job/gap lineage that does not fit cleanly in
the row schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from scripts.pipeline.models import KVPRow
from scripts.pipeline.provenance import (
    DatasetSample,
    dataset_sample_from_kvp_row,
)

DEFAULT_SAMPLE_SIDECARS = (
    "provenance/data_designer_samples.jsonl",
)


def _dump_sample(sample: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(sample, BaseModel):
        return sample.model_dump(mode="json", exclude_none=False)
    return dict(sample)


def read_sample_sidecar(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_sample_sidecars(
    dataset_dir: Path,
    *,
    sidecar_rel_paths: tuple[str, ...] = DEFAULT_SAMPLE_SIDECARS,
) -> dict[str, dict[str, Any]]:
    samples_by_id: dict[str, dict[str, Any]] = {}
    for rel_path in sidecar_rel_paths:
        for sample in read_sample_sidecar(dataset_dir / rel_path):
            sample_id = sample.get("sample_id")
            if sample_id:
                samples_by_id[str(sample_id)] = sample
    return samples_by_id


def merge_dataset_sample_sidecar(
    base_sample: BaseModel | dict[str, Any],
    sidecar_sample: dict[str, Any],
) -> DatasetSample:
    """Overlay sidecar service lineage onto a row-derived dataset sample."""
    base = _dump_sample(base_sample)
    sidecar = _dump_sample(sidecar_sample)
    lineage = {
        **(base.get("lineage") if isinstance(base.get("lineage"), dict) else {}),
        **(sidecar.get("lineage") if isinstance(sidecar.get("lineage"), dict) else {}),
    }
    quality = {
        **(sidecar.get("quality") if isinstance(sidecar.get("quality"), dict) else {}),
        **(base.get("quality") if isinstance(base.get("quality"), dict) else {}),
    }
    metadata = {
        **(sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}),
        **(base.get("metadata") if isinstance(base.get("metadata"), dict) else {}),
    }
    merged = {
        **sidecar,
        **base,
        "origin": sidecar.get("origin") or base.get("origin"),
        "task_type": base.get("task_type") or sidecar.get("task_type"),
        "lineage": lineage,
        "quality": quality,
        "metadata": metadata,
    }
    return DatasetSample.model_validate(merged)


def admitted_dataset_samples_from_kvp_rows(
    rows: list[KVPRow],
    *,
    dataset_dir: Path,
    system_prompt: str | None = None,
    sidecar_rel_paths: tuple[str, ...] = DEFAULT_SAMPLE_SIDECARS,
) -> list[DatasetSample]:
    sidecars = load_sample_sidecars(dataset_dir, sidecar_rel_paths=sidecar_rel_paths)
    samples: list[DatasetSample] = []
    for row in rows:
        base_sample = dataset_sample_from_kvp_row(row, system_prompt=system_prompt)
        sidecar_sample = sidecars.get(base_sample.sample_id)
        if sidecar_sample:
            samples.append(merge_dataset_sample_sidecar(base_sample, sidecar_sample))
        else:
            samples.append(base_sample)
    return samples
