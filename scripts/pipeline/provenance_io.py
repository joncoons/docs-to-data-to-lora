"""File IO helpers for provenance sidecar artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel


def _dump_record(record: BaseModel | dict) -> dict:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json", exclude_none=False)
    return record


def write_json(path: Path, record: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_dump_record(record), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: Iterable[BaseModel | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(_dump_record(record), sort_keys=True) + "\n")
