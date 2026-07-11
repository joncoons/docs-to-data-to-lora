#!/usr/bin/env python3
"""Run downstream LE dataset stages from existing Stage 1A artifacts.

This runner is scoped to the 2026-07-09 uncapped LE rerun. It preserves the
existing Stage 0/1A files, reconstructs Stage 1B seed vectors in memory, and
uses target-aware load balancing so local and NVIDIA-hosted Super aliases can
be used together.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.pipeline.config import Config, get_es_password, get_external_judge_api_key  # noqa: E402
from scripts.pipeline.dataset_admission import admitted_dataset_samples_from_kvp_rows  # noqa: E402
from scripts.pipeline.es_client import make_es_client, scroll_all_chunks  # noqa: E402
from scripts.pipeline.finalize_dataset import finalize_dataset  # noqa: E402
from scripts.pipeline.models import KVPRow, Passage  # noqa: E402
from scripts.pipeline.provenance_io import write_jsonl  # noqa: E402
from scripts.pipeline.stage1_5_gapfill import run_stage1_5  # noqa: E402
from scripts.pipeline.stage1b_synthesis import run_stage1b  # noqa: E402
from scripts.pipeline.stage1c_instruction import run_stage1c  # noqa: E402
from scripts.pipeline.stage2_qa_eval import run_stage2  # noqa: E402
from scripts.pipeline.stage3_curator import run_stage3  # noqa: E402


log = logging.getLogger("le_downstream")

STAGE_ORDER = ("1b", "1c", "1.5", "2", "3", "finalize")
# This runner starts after Stage 1A. Synthesis defaults remain on Ultra-class
# targets, while Stage 2 QA admission has a separate Super 120B-class default to
# keep the required admission pass cost-conscious.
DEFAULT_TARGETS = (
    "https://inference-api.nvidia.com/v1=nvidia/nvidia/nemotron-3-ultra",
)
DEFAULT_CANONICAL_MODEL = "nvidia/nvidia/nemotron-3-ultra"
DEFAULT_STAGE2_TARGETS = (
    "https://inference-api.nvidia.com/v1=nvidia/nvidia/nemotron-3-super-v3",
)
DEFAULT_STAGE2_CANONICAL_MODEL = "nvidia/nvidia/nemotron-3-super-v3"
DEFAULT_SOURCE_DOC_KIND = "all"
COLLECTION_DOMAIN = {
    "nim_curated": "NVIDIA NIM",
    "nemo_usvcs_curated": "NVIDIA NeMo Microservices",
}

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_PRELUDE_RE = re.compile(r"^.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class LLMTarget:
    endpoint: str
    model: str
    api_key: str
    max_model_len: int | None = None


@dataclass(frozen=True)
class TargetCall:
    index: int
    client: OpenAI
    target: LLMTarget


def strip_think_blocks(content: str) -> str:
    if not content:
        return content
    content = _THINK_BLOCK_RE.sub("", content)
    content = _THINK_PRELUDE_RE.sub("", content)
    return content.strip()


class TargetAwareLLMClient:
    """Duck-compatible replacement for LLMClient with per-target model IDs."""

    def __init__(
        self,
        *,
        targets: list[LLMTarget],
        canonical_model: str,
        max_workers: int,
        min_interval_s: float,
        retry_attempts: int,
        retry_base_delay_s: float,
        no_think: bool,
        temperature: float,
        request_timeout_s: float,
    ) -> None:
        if not targets:
            raise ValueError("at least one LLM target is required")
        self.targets = targets
        self.endpoints = [target.endpoint for target in targets]
        self.model = canonical_model
        self.no_think = no_think
        self.temperature = temperature
        self.retry_attempts = retry_attempts
        self.retry_base_delay_s = retry_base_delay_s
        self._semaphore = threading.Semaphore(max_workers)
        self._last_call = [0.0]
        self._last_call_lock = threading.Lock()
        self._min_interval = min_interval_s
        self._counter = itertools.count()
        self._clients = [
            OpenAI(base_url=target.endpoint, api_key=target.api_key, timeout=request_timeout_s)
            for target in targets
        ]

    @staticmethod
    def _estimated_prompt_tokens(system: str, user: str) -> int:
        text = system + "\n" + user
        return max(int(len(text.split()) * 1.4), len(text) // 3)

    @staticmethod
    def _target_fits(target: LLMTarget, prompt_tokens: int, max_tokens: int) -> bool:
        if target.max_model_len is None:
            return True
        return prompt_tokens + max_tokens <= target.max_model_len - 512

    @staticmethod
    def _no_think_extra_body(endpoint: str) -> dict[str, Any]:
        if "inference-api.nvidia.com" in endpoint or "integrate.api.nvidia.com" in endpoint:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {"reasoning_effort": "none"}

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "maximum context length" in text or "input_tokens" in text

    def _next_client(
        self,
        system: str,
        user: str,
        max_tokens: int,
        *,
        avoid_indices: set[int] | None = None,
    ) -> TargetCall:
        avoid_indices = avoid_indices or set()
        prompt_tokens = self._estimated_prompt_tokens(system, user)
        start = next(self._counter)
        fallback_index = start % len(self._clients)
        for offset in range(len(self._clients)):
            index = (start + offset) % len(self._clients)
            if index in avoid_indices:
                continue
            target = self.targets[index]
            if self._target_fits(target, prompt_tokens, max_tokens):
                return TargetCall(index, self._clients[index], target)
        for offset in range(len(self._clients)):
            index = (start + offset) % len(self._clients)
            if index in avoid_indices:
                continue
            target = self.targets[index]
            if target.max_model_len is None:
                return TargetCall(index, self._clients[index], target)
        return TargetCall(fallback_index, self._clients[fallback_index], self.targets[fallback_index])

    def call(self, system: str, user: str, max_tokens: int = 1024) -> Optional[str]:
        with self._semaphore:
            avoid_indices: set[int] = set()
            for attempt in range(self.retry_attempts):
                with self._last_call_lock:
                    elapsed = time.time() - self._last_call[0]
                    if elapsed < self._min_interval:
                        time.sleep(self._min_interval - elapsed)
                    target_call = self._next_client(
                        system,
                        user,
                        max_tokens,
                        avoid_indices=avoid_indices,
                    )
                    self._last_call[0] = time.time()
                target = target_call.target
                try:
                    kwargs: dict[str, Any] = {
                        "model": target.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": max_tokens,
                    }
                    if self.no_think:
                        kwargs["extra_body"] = self._no_think_extra_body(target.endpoint)
                    response = target_call.client.chat.completions.create(**kwargs)
                    return strip_think_blocks(response.choices[0].message.content or "")
                except Exception as exc:  # noqa: BLE001 - per-call retries are recorded in logs.
                    if target.max_model_len is not None and self._is_context_length_error(exc):
                        avoid_indices.add(target_call.index)
                        log.warning(
                            "Context limit on %s; retrying on another target if available: %s",
                            target.endpoint,
                            exc,
                        )
                        continue
                    log.warning("LLM call attempt %d/%d failed: %s", attempt + 1, self.retry_attempts, exc)
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_base_delay_s * (attempt + 1))
            return None


def endpoint_api_key(endpoint: str, explicit_api_key: str | None = None) -> str:
    if explicit_api_key is not None:
        return explicit_api_key
    if "inference-api.nvidia.com" in endpoint or "integrate.api.nvidia.com" in endpoint:
        return get_external_judge_api_key()
    return "local"


def parse_targets(raw_targets: list[str], explicit_api_key: str | None = None) -> list[LLMTarget]:
    targets: list[LLMTarget] = []
    for raw in raw_targets:
        if "=" not in raw:
            raise ValueError("--target must be ENDPOINT=MODEL or ENDPOINT=MODEL@MAX_CONTEXT_TOKENS")
        endpoint, model = raw.split("=", 1)
        endpoint = endpoint.strip()
        model = model.strip()
        max_model_len: int | None = None
        if "@" in model:
            model, raw_max_model_len = model.rsplit("@", 1)
            max_model_len = int(raw_max_model_len)
        elif "inference-api.nvidia.com" not in endpoint and "integrate.api.nvidia.com" not in endpoint:
            max_model_len = 32768
        if not endpoint or not model:
            raise ValueError("--target must include non-empty endpoint and model")
        targets.append(
            LLMTarget(
                endpoint=endpoint,
                model=model,
                api_key=endpoint_api_key(endpoint, explicit_api_key),
                max_model_len=max_model_len,
            )
        )
    return targets


def read_passages(path: Path) -> list[Passage]:
    return [Passage.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def read_rows(path: Path) -> list[KVPRow]:
    if not path.exists():
        return []
    return [KVPRow.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def select_passages(passages: list[Passage], source_doc_kind: str) -> list[Passage]:
    if source_doc_kind == "all":
        return passages
    return [passage for passage in passages if passage.doc_kind == source_doc_kind]


def filter_rows_by_passage(rows: list[KVPRow], passage_ids: set[str]) -> list[KVPRow]:
    return [row for row in rows if row.passage_id in passage_ids]


def _list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _doc_kind_counts(passages: list[Passage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for passage in passages:
        counts[passage.doc_kind] = counts.get(passage.doc_kind, 0) + 1
    return counts


def _jsonl_projection(
    input_path: Path,
    output_path: Path,
    predicate: Any,
) -> dict[str, Any]:
    input_rows = 0
    selected_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        if input_path.exists():
            for line in input_path.read_text().splitlines():
                if not line.strip():
                    continue
                input_rows += 1
                row = json.loads(line)
                if predicate(row):
                    selected_rows += 1
                    out.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "path": str(output_path.relative_to(output_path.parents[2])),
        "input": input_rows,
        "selected": selected_rows,
        "excluded": input_rows - selected_rows,
    }


def write_selected_lineage_artifacts(
    output_dir: Path,
    *,
    source_doc_kind: str,
    selected_passages: list[Passage],
    selected_stage1a_rows: list[KVPRow],
) -> dict[str, Any]:
    selected_passage_ids = {passage.passage_id for passage in selected_passages}
    selected_source_urls = {passage.url for passage in selected_passages}
    selected_chunk_ids: set[str] = set()
    selected_source_revision_ids: set[str] = set()
    selected_entailment_ids: set[str] = set()

    for passage in selected_passages:
        selected_chunk_ids.update(_list_values(passage.chunk_ids))
        selected_chunk_ids.update(_list_values(passage.source_chunk_ids))
        selected_source_revision_ids.update(_list_values(passage.source_revision_id))

    for row in selected_stage1a_rows:
        selected_chunk_ids.update(_list_values(row.source_chunk_ids))
        selected_source_revision_ids.update(_list_values(row.source_revision_ids))
        selected_entailment_ids.update(_list_values(row.entailment_id))

    lineage_dir_name = "html_only" if source_doc_kind == "html" else "selected"
    lineage_dir = output_dir / "provenance" / lineage_dir_name

    def matching_doc_kind(row_doc_kind: Any) -> bool:
        return source_doc_kind == "all" or row_doc_kind == source_doc_kind

    source_chunks = _jsonl_projection(
        output_dir / "provenance" / "source_chunks.jsonl",
        lineage_dir / "source_chunks.jsonl",
        lambda row: matching_doc_kind((row.get("metadata") or {}).get("doc_kind"))
        and (
            row.get("chunk_id") in selected_chunk_ids
            or (row.get("metadata") or {}).get("passage_id") in selected_passage_ids
        ),
    )
    source_revisions = _jsonl_projection(
        output_dir / "provenance" / "source_revisions.jsonl",
        lineage_dir / "source_revisions.jsonl",
        lambda row: matching_doc_kind((row.get("classification") or {}).get("doc_kind"))
        and (
            row.get("source_revision_id") in selected_source_revision_ids
            or row.get("canonical_url") in selected_source_urls
            or row.get("final_url") in selected_source_urls
        ),
    )
    entailments = _jsonl_projection(
        output_dir / "provenance" / "entailments.jsonl",
        lineage_dir / "entailments.jsonl",
        lambda row: (row.get("metadata") or {}).get("passage_id") in selected_passage_ids
        and (
            not selected_entailment_ids
            or row.get("entailment_id") in selected_entailment_ids
        ),
    )

    manifest = {
        "schema_version": "le_downstream.selected_lineage.v1",
        "source_doc_kind_filter": source_doc_kind,
        "selected_passages": len(selected_passage_ids),
        "selected_stage1a_rows": len(selected_stage1a_rows),
        "selected_source_revision_ids": len(selected_source_revision_ids),
        "selected_source_chunk_ids": len(selected_chunk_ids),
        "selected_entailment_ids": len(selected_entailment_ids),
        "lineage_dir": str(lineage_dir.relative_to(output_dir)),
        "artifacts": {
            "source_revisions": source_revisions,
            "source_chunks": source_chunks,
            "entailments": entailments,
        },
    }
    (lineage_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_source_filter_artifacts(
    output_dir: Path,
    *,
    source_doc_kind: str,
    all_passages: list[Passage],
    selected_passages: list[Passage],
    raw_stage1a_rows: list[KVPRow],
    selected_stage1a_rows: list[KVPRow],
    excluded_passage_reasons: dict[str, str] | None = None,
) -> dict[str, Any]:
    selected_ids = {passage.passage_id for passage in selected_passages}
    excluded_passage_reasons = excluded_passage_reasons or {}
    excluded_passages = [
        passage for passage in all_passages if passage.passage_id not in selected_ids
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir = output_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    selected_stage1a_name = (
        "stage1a_le_html.jsonl" if source_doc_kind == "html" else "stage1a_le_selected.jsonl"
    )
    selected_stage1a_path = output_dir / selected_stage1a_name
    with selected_stage1a_path.open("w", encoding="utf-8") as stream:
        for row in selected_stage1a_rows:
            stream.write(row.model_dump_json() + "\n")

    selected_lineage = write_selected_lineage_artifacts(
        output_dir,
        source_doc_kind=source_doc_kind,
        selected_passages=selected_passages,
        selected_stage1a_rows=selected_stage1a_rows,
    )

    excluded_path = provenance_dir / "source_filter_excluded_passages.jsonl"
    with excluded_path.open("w", encoding="utf-8") as stream:
        for passage in excluded_passages:
            stream.write(
                json.dumps(
                    {
                        "passage_id": passage.passage_id,
                        "url": passage.url,
                        "doc_kind": passage.doc_kind,
                        "chunk_ids": passage.chunk_ids,
                        "reason": excluded_passage_reasons.get(
                            passage.passage_id,
                            "doc_kind_filter" if source_doc_kind != "all" else "not_selected",
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    payload = {
        "schema_version": "le_downstream.source_filter.v1",
        "source_doc_kind_filter": source_doc_kind,
        "passage_counts": {
            "input": len(all_passages),
            "selected": len(selected_passages),
            "excluded": len(excluded_passages),
            "by_doc_kind": _doc_kind_counts(all_passages),
        },
        "stage1a_row_counts": {
            "input": len(raw_stage1a_rows),
            "selected": len(selected_stage1a_rows),
            "excluded": len(raw_stage1a_rows) - len(selected_stage1a_rows),
        },
        "outputs": {
            "selected_stage1a_rows": selected_stage1a_name,
            "selected_lineage_dir": selected_lineage["lineage_dir"],
            "excluded_passages": "provenance/source_filter_excluded_passages.jsonl",
        },
        "selected_lineage": selected_lineage,
    }
    (provenance_dir / "source_filter.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as stream:
        return sum(1 for line in stream if line.strip())


def load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text())
    return set(payload.get("completed_stages", []))


def write_progress(path: Path, completed: set[str]) -> None:
    path.write_text(json.dumps({"completed_stages": sorted(completed)}, indent=2) + "\n")


def stage_selected(requested: str, stage: str) -> bool:
    return requested == "all" or requested == stage


def latest_stage1a_records(
    output_dir: Path,
    *,
    passage_ids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    path = output_dir / "stage1a_passage_results.jsonl"
    if not path.exists():
        return {}, set(passage_ids or [])
    latest: dict[str, dict[str, Any]] = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            latest[row["passage_id"]] = row
    if passage_ids is None:
        return latest, set()
    missing = passage_ids - set(latest)
    latest = {pid: row for pid, row in latest.items() if pid in passage_ids}
    return latest, missing


def latest_stage1a_status(
    output_dir: Path,
    *,
    passage_ids: set[str] | None = None,
) -> tuple[dict[str, int], int]:
    latest, missing_ids = latest_stage1a_records(output_dir, passage_ids=passage_ids)
    counts: dict[str, int] = {}
    for row in latest.values():
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts, len(missing_ids)


def complete_stage1a_passage_filter(
    output_dir: Path,
    passage_ids: set[str],
) -> tuple[set[str], dict[str, str]]:
    latest, missing_ids = latest_stage1a_records(output_dir, passage_ids=passage_ids)
    complete_ids: set[str] = set()
    exclusions: dict[str, str] = {pid: "missing" for pid in missing_ids}
    for pid, row in latest.items():
        status = str(row.get("status") or "unknown")
        if status in {"complete", "no_entailments"}:
            complete_ids.add(pid)
        else:
            exclusions[pid] = status
    return complete_ids, exclusions


def validate_stage1a_complete(
    output_dir: Path,
    *,
    allow_incomplete: bool,
    passage_ids: set[str] | None = None,
) -> None:
    counts, missing = latest_stage1a_status(output_dir, passage_ids=passage_ids)
    if not counts and not missing:
        log.warning("No stage1a_passage_results.jsonl found; Stage 1A completion cannot be verified")
        return
    incomplete = missing + sum(
        count
        for status, count in counts.items()
        if status not in {"complete", "no_entailments"}
    )
    log.info("Stage 1A latest status for selected passages: %s missing=%d", counts, missing)
    if incomplete and not allow_incomplete:
        raise SystemExit(
            f"Selected Stage 1A passages are not complete ({incomplete} retryable/missing passages remain); "
            "rerun recovery or pass --allow-incomplete-stage1a explicitly."
        )


def load_seed_vectors(es: Any, index: str, passages: list[Passage]) -> dict[str, list[float]]:
    wanted_chunk_ids = {chunk_id for passage in passages for chunk_id in passage.chunk_ids}
    vectors_by_chunk_id: dict[str, list[float]] = {}
    for hit in scroll_all_chunks(es, index):
        hit_id = hit.get("_id")
        if hit_id not in wanted_chunk_ids:
            continue
        vector = hit.get("_source", {}).get("vector")
        if vector:
            vectors_by_chunk_id[hit_id] = vector
            if len(vectors_by_chunk_id) == len(wanted_chunk_ids):
                break

    seed_vectors: dict[str, list[float]] = {}
    for passage in passages:
        for chunk_id in passage.chunk_ids:
            vector = vectors_by_chunk_id.get(chunk_id)
            if vector:
                seed_vectors[passage.passage_id] = vector
                break
    log.info(
        "Stage 1B seed vectors: %d/%d passages, %d/%d chunk IDs",
        len(seed_vectors),
        len(passages),
        len(vectors_by_chunk_id),
        len(wanted_chunk_ids),
    )
    return seed_vectors


def write_summary(
    output_dir: Path,
    *,
    collection: str,
    targets: list[LLMTarget],
    canonical_model: str,
    stage2_targets: list[LLMTarget],
    stage2_canonical_model: str,
    temperature: float,
    stage2_execution_surface: str,
    stage2_max_tokens: int,
    source_filter: dict[str, Any],
    completed: set[str],
    manifest: dict[str, Any] | None,
) -> None:
    counts = {
        "passages": count_jsonl(output_dir / "passages.jsonl"),
        "stage1a_rows": count_jsonl(output_dir / "stage1a_le.jsonl"),
        "stage1b_rows": count_jsonl(output_dir / "stage1b_synthesis.jsonl"),
        "stage1c_rows": count_jsonl(output_dir / "stage1c_instruction.jsonl"),
        "stage1_5_rows": count_jsonl(output_dir / "stage1_5_gapfill.jsonl"),
        "stage2_rows": count_jsonl(output_dir / "stage2_eval.jsonl"),
        "stage2_dropped_rows": count_jsonl(output_dir / "stage2_dropped.jsonl"),
        "training_rows": count_jsonl(output_dir / "training.jsonl"),
        "validation_rows": count_jsonl(output_dir / "validation.jsonl"),
    }
    payload = {
        "collection": collection,
        "output_dir": str(output_dir),
        "completed_stages": sorted(completed),
        "targets": [
            {k: v for k, v in asdict(target).items() if k != "api_key"}
            for target in targets
        ],
        "canonical_model": canonical_model,
        "stage2_targets": [
            {k: v for k, v in asdict(target).items() if k != "api_key"}
            for target in stage2_targets
        ],
        "stage2_canonical_model": stage2_canonical_model,
        "temperature": temperature,
        "stage2_execution_surface": stage2_execution_surface,
        "stage2_max_tokens": stage2_max_tokens,
        "source_filter": source_filter,
        "counts": counts,
        "dataset_version_id": manifest.get("dataset_version_id") if manifest else None,
    }
    (output_dir / "le_downstream_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# LE Downstream Dataset Summary",
        "",
        f"- Collection: `{collection}`",
        f"- Output dir: `{output_dir}`",
        f"- Completed stages: `{', '.join(sorted(completed))}`",
        f"- Dataset version: `{payload['dataset_version_id'] or 'not finalized'}`",
        f"- LLM temperature: `{temperature}`",
        f"- Stage 2 execution surface: `{stage2_execution_surface}`",
        f"- Stage 2 max tokens: `{stage2_max_tokens}`",
        f"- Source filter: `{source_filter.get('source_doc_kind_filter')}`",
        "",
        "## Synthesis Targets",
    ]
    for target in payload["targets"]:
        max_len = target["max_model_len"] or "uncapped"
        lines.append(f"- `{target['endpoint']}` -> `{target['model']}` (`{max_len}`)")
    lines.extend(["", "## Stage 2 QA Targets"])
    for target in payload["stage2_targets"]:
        max_len = target["max_model_len"] or "uncapped"
        lines.append(f"- `{target['endpoint']}` -> `{target['model']}` (`{max_len}`)")
    lines.extend(["", "## Source Filter"])
    passage_counts = source_filter.get("passage_counts", {})
    stage1a_counts = source_filter.get("stage1a_row_counts", {})
    lines.append(f"- Passages selected: {passage_counts.get('selected', 0)} / {passage_counts.get('input', 0)}")
    lines.append(f"- Stage 1A rows selected: {stage1a_counts.get('selected', 0)} / {stage1a_counts.get('input', 0)}")
    lines.extend(["", "## Counts"])
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    (output_dir / "le_downstream_summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", required=True, choices=sorted(COLLECTION_DOMAIN))
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--stage", default="all", choices=(*STAGE_ORDER, "all"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-incomplete-stage1a", action="store_true")
    ap.add_argument(
        "--source-doc-kind",
        choices=["html", "all"],
        default=DEFAULT_SOURCE_DOC_KIND,
        help="Filter source passages and raw KVPs by Stage 0 doc_kind. Default keeps all source kinds.",
    )
    ap.add_argument("--es-host", default=None)
    ap.add_argument("--target", action="append", default=None)
    ap.add_argument("--canonical-model", default=None)
    ap.add_argument(
        "--stage2-target",
        action="append",
        default=None,
        help=(
            "Stage 2 QA target override as ENDPOINT=MODEL[@MAX_CONTEXT_TOKENS]. "
            "Defaults to hosted Nemotron 3 Super; repeat for load balancing."
        ),
    )
    ap.add_argument("--stage2-canonical-model", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument(
        "--stage1c-selection-mode",
        choices=["stratified", "top_density", "all"],
        default=None,
        help="Stage 1C passage selection mode. Defaults to pipeline config.",
    )
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--min-request-interval-s", type=float, default=None)
    ap.add_argument("--retry-attempts", type=int, default=None)
    ap.add_argument("--retry-base-delay-s", type=float, default=None)
    ap.add_argument("--request-timeout-s", type=float, default=600.0)
    ap.add_argument("--stage2-qa-max-tokens", type=int, default=None)
    ap.add_argument(
        "--stage2-execution-surface",
        default=None,
        help="Audit label for Stage 2 QA execution surface. Defaults to config.",
    )
    ap.add_argument(
        "--stage3-tokenizer",
        default=None,
        help="Production tokenizer path/model for Stage 3 length filtering. Defaults to config.",
    )
    ap.add_argument("--stage3-min-question-tokens", type=int, default=None)
    ap.add_argument("--stage3-min-answer-tokens", type=int, default=None)
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    cfg = Config()
    if args.es_host:
        cfg.es_host = args.es_host
    if args.max_workers is not None:
        cfg.max_workers = args.max_workers
    if args.min_request_interval_s is not None:
        cfg.min_request_interval_s = args.min_request_interval_s
    if args.retry_attempts is not None:
        cfg.retry_attempts = args.retry_attempts
    if args.retry_base_delay_s is not None:
        cfg.retry_base_delay_s = args.retry_base_delay_s
    if args.stage2_qa_max_tokens is not None and args.stage2_qa_max_tokens < 1:
        raise SystemExit("--stage2-qa-max-tokens must be >= 1")
    if args.stage3_min_question_tokens is not None and args.stage3_min_question_tokens < 1:
        raise SystemExit("--stage3-min-question-tokens must be >= 1")
    if args.stage3_min_answer_tokens is not None and args.stage3_min_answer_tokens < 1:
        raise SystemExit("--stage3-min-answer-tokens must be >= 1")
    stage2_max_tokens = args.stage2_qa_max_tokens or cfg.stage2_qa_max_tokens
    stage2_execution_surface = args.stage2_execution_surface or cfg.stage2_execution_surface
    stage3_tokenizer = args.stage3_tokenizer or cfg.stage3_tokenizer_name_or_path
    stage3_min_question_tokens = args.stage3_min_question_tokens or cfg.min_question_tokens
    stage3_min_answer_tokens = args.stage3_min_answer_tokens or cfg.min_answer_tokens

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "le_downstream_progress.json"
    completed = load_progress(progress_path)

    all_passages = read_passages(output_dir / "passages.jsonl")
    if not all_passages:
        raise SystemExit(f"No passages found at {output_dir / 'passages.jsonl'}")
    passages = select_passages(all_passages, args.source_doc_kind)
    if not passages:
        raise SystemExit(f"No passages matched --source-doc-kind={args.source_doc_kind}")
    selected_passage_ids = {passage.passage_id for passage in passages}
    excluded_passage_reasons: dict[str, str] = {}
    if args.allow_incomplete_stage1a:
        complete_ids, incomplete_reasons = complete_stage1a_passage_filter(
            output_dir,
            selected_passage_ids,
        )
        if incomplete_reasons:
            excluded_passage_reasons.update(
                {pid: f"stage1a_{reason}" for pid, reason in incomplete_reasons.items()}
            )
            passages = [passage for passage in passages if passage.passage_id in complete_ids]
            selected_passage_ids = {passage.passage_id for passage in passages}
            log.warning(
                "Excluding %d selected passages with incomplete Stage 1A status: %s",
                len(incomplete_reasons),
                dict(sorted(incomplete_reasons.items())),
            )
    validate_stage1a_complete(
        output_dir,
        allow_incomplete=args.allow_incomplete_stage1a,
        passage_ids=selected_passage_ids,
    )
    stage1a_rows_raw = read_rows(output_dir / "stage1a_le.jsonl")
    stage1a_rows = filter_rows_by_passage(stage1a_rows_raw, selected_passage_ids)
    if not stage1a_rows_raw:
        raise SystemExit(f"No Stage 1A rows found at {output_dir / 'stage1a_le.jsonl'}")
    if not stage1a_rows:
        raise SystemExit(
            f"No Stage 1A rows matched --source-doc-kind={args.source_doc_kind}"
        )
    source_filter = write_source_filter_artifacts(
        output_dir,
        source_doc_kind=args.source_doc_kind,
        all_passages=all_passages,
        selected_passages=passages,
        raw_stage1a_rows=stage1a_rows_raw,
        selected_stage1a_rows=stage1a_rows,
        excluded_passage_reasons=excluded_passage_reasons,
    )
    log.info(
        "Source filter: doc_kind=%s passages=%d/%d stage1a_rows=%d/%d",
        args.source_doc_kind,
        len(passages),
        len(all_passages),
        len(stage1a_rows),
        len(stage1a_rows_raw),
    )

    domain = COLLECTION_DOMAIN[args.collection]
    system_prompt = (
        f"You are a precise {domain} technical assistant. "
        "Answer based on official documentation."
    )
    targets = parse_targets(args.target or list(DEFAULT_TARGETS), explicit_api_key=args.api_key)
    canonical_model = args.canonical_model or DEFAULT_CANONICAL_MODEL
    llm = TargetAwareLLMClient(
        targets=targets,
        canonical_model=canonical_model,
        max_workers=cfg.max_workers,
        min_interval_s=cfg.min_request_interval_s,
        retry_attempts=cfg.retry_attempts,
        retry_base_delay_s=cfg.retry_base_delay_s,
        no_think=True,
        temperature=args.temperature,
        request_timeout_s=args.request_timeout_s,
    )
    stage2_targets = parse_targets(
        args.stage2_target or list(DEFAULT_STAGE2_TARGETS),
        explicit_api_key=args.api_key,
    )
    stage2_canonical_model = args.stage2_canonical_model or DEFAULT_STAGE2_CANONICAL_MODEL
    stage2_llm = TargetAwareLLMClient(
        targets=stage2_targets,
        canonical_model=stage2_canonical_model,
        max_workers=cfg.max_workers,
        min_interval_s=cfg.min_request_interval_s,
        retry_attempts=cfg.retry_attempts,
        retry_base_delay_s=cfg.retry_base_delay_s,
        no_think=True,
        temperature=args.temperature,
        request_timeout_s=args.request_timeout_s,
    )
    es = make_es_client(cfg.es_host, get_es_password())

    log.info(
        "LE downstream start: collection=%s output=%s stage=%s resume=%s targets=%s stage2_targets=%s",
        args.collection,
        output_dir,
        args.stage,
        args.resume,
        [(target.endpoint, target.model, target.max_model_len) for target in targets],
        [(target.endpoint, target.model, target.max_model_len) for target in stage2_targets],
    )

    manifest: dict[str, Any] | None = None

    if stage_selected(args.stage, "1b"):
        if args.resume and "1b" in completed and (output_dir / "stage1b_synthesis.jsonl").exists():
            log.info("Stage 1B: skipping (already done)")
        else:
            seed_vectors = load_seed_vectors(es, args.collection, passages)
            run_stage1b(
                passages,
                seed_vectors,
                es,
                args.collection,
                domain,
                llm,
                output_dir,
                max_workers=cfg.max_workers,
                knn_k=cfg.knn_k,
                num_candidates=cfg.knn_num_candidates,
                top_neighbors=cfg.knn_top_neighbors,
                max_context_tokens=cfg.knn_max_context_tokens,
                resume=args.resume,
            )
            completed.add("1b")
            write_progress(progress_path, completed)
    stage1b_rows = filter_rows_by_passage(
        read_rows(output_dir / "stage1b_synthesis.jsonl"),
        selected_passage_ids,
    )

    if stage_selected(args.stage, "1c"):
        if args.resume and "1c" in completed and (output_dir / "stage1c_instruction.jsonl").exists():
            log.info("Stage 1C: skipping (already done)")
        else:
            run_stage1c(
                passages,
                domain,
                llm,
                output_dir,
                top_percent=cfg.stage1c_top_percent,
                min_passages=cfg.stage1c_min_passages,
                max_workers=cfg.max_workers,
                selection_mode=args.stage1c_selection_mode or cfg.stage1c_selection_mode,
                resume=args.resume,
            )
            completed.add("1c")
            write_progress(progress_path, completed)
    stage1c_rows = filter_rows_by_passage(
        read_rows(output_dir / "stage1c_instruction.jsonl"),
        selected_passage_ids,
    )

    if stage_selected(args.stage, "1.5"):
        if args.resume and "1.5" in completed and (output_dir / "stage1_5_gapfill.jsonl").exists():
            log.info("Stage 1.5: skipping (already done)")
        else:
            run_stage1_5(
                passages,
                stage1a_rows + stage1b_rows + stage1c_rows,
                es,
                args.collection,
                llm,
                output_dir,
                threshold_factor=cfg.bias_threshold_factor,
                target_factor=cfg.bias_gapfill_target_factor,
                top_n_chunks=cfg.gapfill_top_n_chunks,
                pairs_per_call=cfg.gapfill_pairs_per_call,
                max_attempt_factor=cfg.gapfill_max_attempt_factor,
                legacy_direct=False,
            )
            completed.add("1.5")
            write_progress(progress_path, completed)
    stage1_5_rows = filter_rows_by_passage(
        read_rows(output_dir / "stage1_5_gapfill.jsonl"),
        selected_passage_ids,
    )

    all_pre_eval = stage1a_rows + stage1b_rows + stage1c_rows + stage1_5_rows
    if stage_selected(args.stage, "2"):
        if args.resume and "2" in completed:
            log.info("Stage 2: progress is marked done; checking durable row-level resume")
        run_stage2(
            all_pre_eval,
            stage2_llm,
            output_dir,
            max_workers=cfg.max_workers,
            resume=args.resume,
            execution_surface=stage2_execution_surface,
            max_tokens=stage2_max_tokens,
        )
        completed.add("2")
        write_progress(progress_path, completed)
    stage2_rows = filter_rows_by_passage(
        read_rows(output_dir / "stage2_eval.jsonl"),
        selected_passage_ids,
    )

    if stage_selected(args.stage, "3"):
        if args.resume and "3" in completed and (output_dir / "training.jsonl").exists():
            log.info("Stage 3: skipping (already done)")
        else:
            if not stage2_rows:
                raise SystemExit("Stage 3 requires stage2_eval.jsonl rows")
            run_stage3(
                stage2_rows,
                output_dir,
                system_prompt,
                train_ratio=cfg.train_val_split,
                minhash_threshold=cfg.minhash_threshold,
                min_q_tokens=stage3_min_question_tokens,
                min_a_tokens=stage3_min_answer_tokens,
                tokenizer_name_or_path=stage3_tokenizer,
            )
            completed.add("3")
            write_progress(progress_path, completed)

    if stage_selected(args.stage, "finalize"):
        if (
            args.resume
            and "finalize" in completed
            and (output_dir / "manifests" / "dataset_version_manifest.json").exists()
        ):
            log.info("Finalize: skipping (already done)")
            manifest = json.loads((output_dir / "manifests" / "dataset_version_manifest.json").read_text())
        else:
            sample_rows = stage2_rows or all_pre_eval
            write_jsonl(
                output_dir / "provenance" / "dataset_samples.jsonl",
                admitted_dataset_samples_from_kvp_rows(
                    sample_rows,
                    dataset_dir=output_dir,
                    system_prompt=system_prompt,
                ),
            )
            manifest = finalize_dataset(
                output_dir,
                dataset_name=args.collection,
                system_prompt=system_prompt,
                observability_dir=None,
            )
            completed.add("finalize")
            write_progress(progress_path, completed)

    write_summary(
        output_dir,
        collection=args.collection,
        targets=targets,
        canonical_model=canonical_model,
        stage2_targets=stage2_targets,
        stage2_canonical_model=stage2_canonical_model,
        temperature=args.temperature,
        stage2_execution_surface=stage2_execution_surface,
        stage2_max_tokens=stage2_max_tokens,
        source_filter=source_filter,
        completed=completed,
        manifest=manifest,
    )
    log.info("LE downstream done: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
