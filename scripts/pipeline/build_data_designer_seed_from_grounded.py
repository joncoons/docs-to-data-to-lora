#!/usr/bin/env python3
"""Build a NeMo Data Designer seed set from grounded training rows.

This is the first step in the NIM 1B Data Designer augmentation experiment. It
selects provenance-rich rows from the grounded training split, writes a seed CSV
that NeMo Data Designer can consume, and records enough lineage to merge
accepted synthetic rows back into a new dataset entity later.

By default this script does not call an external model. Use ``--mode llm`` and
``--allow-external-llm`` to let a configured remote LLM create the coverage briefs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.provenance import (  # noqa: E402
    SCHEMA_VERSION as PROVENANCE_SCHEMA_VERSION,
    sha256_text,
    stable_id,
    utc_now,
)

DEFAULT_DATASET_DIR = Path(os.getenv("DATASET_DIR", "/datasets/nim_curated"))
DEFAULT_OUTPUT_DIR = Path(
    os.getenv(
        "DATA_DESIGNER_AUGMENTATION_DIR",
        "/datasets/experiments/nim_curated_dd_llm_1b",
    )
)
DEFAULT_LLM_URL = os.getenv(
    "LLM_API_URL",
    "https://llm.example.com/v1",
)
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "frontier-llm-model")
DEFAULT_LLM_API_KEY_ENV = os.getenv("LLM_API_KEY_ENV", "LLM_API_KEY")

SEED_SCHEMA_VERSION = "data-designer-grounded-seed.v1"

DATA_DESIGNER_PROMPT_TEMPLATE = """\
Seed ID: {{ seed_id }}
Coverage axis: {{ coverage_axis }}
Brief: {{ generation_brief }}

Generate {{ pairs_count }} question-answer pairs about the domain slice {{ domain_slice }}.

Use only the following grounded source text and original grounded QA as
evidence. Do not introduce claims that are absent from the source.

Original question:
{{ source_prompt }}

Original answer:
{{ source_completion }}

Source text:
{{ retrieved_chunks }}

Return JSON only in this shape:
{"pairs": [{"question": "...", "answer": "..."}]}
"""

LLM_SYSTEM_PROMPT = (
    "You create seed instructions for a synthetic data generation service. "
    "You do not create final training rows. You produce terse JSON only."
)


_THINK_BALANCED = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TAIL = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_PRELUDE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Strip reasoning-model think blocks before JSON parsing or persistence."""
    if not text:
        return text
    cleaned = _THINK_BALANCED.sub("", text)
    cleaned = _THINK_PRELUDE.sub("", cleaned)
    cleaned = _THINK_OPEN_TAIL.sub("", cleaned)
    return cleaned.strip()


@dataclass(frozen=True)
class SeedConfig:
    dataset_dir: Path
    output_dir: Path
    collection: str
    source_dataset_name: str
    mode: str
    seed_count: int
    pairs_per_seed: int
    random_seed: int
    judge_api_url: str
    judge_model: str
    judge_api_key_env: str
    allow_external_llm: bool
    temperature: float
    max_tokens: int
    timeout_s: float
    max_workers: int = 1
    max_retries: int = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_file_manifest(dataset_dir: Path) -> dict[str, Any]:
    files = [
        "training.jsonl",
        "adapter_train.jsonl",
        "validation.jsonl",
        "adapter_val.jsonl",
        "test_set.jsonl",
        "test_set_with_context.jsonl",
        "stage2_eval.jsonl",
        "validation_report.json",
        "bias_report.json",
    ]
    artifacts: list[dict[str, Any]] = []
    for rel in files:
        path = dataset_dir / rel
        if not path.exists():
            continue
        entry: dict[str, Any] = {
            "relative_path": rel,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".jsonl":
            entry["rows"] = count_jsonl_rows(path)
        artifacts.append(entry)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset_dir": str(dataset_dir),
        "artifacts": artifacts,
    }


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def pair_key(prompt: Any, completion: Any) -> tuple[str, str]:
    return (normalize_text(prompt), normalize_text(completion))


def row_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("question") or "")


def row_completion(row: dict[str, Any]) -> str:
    return str(row.get("completion") or row.get("answer") or "")


def load_excluded_pairs(dataset_dir: Path) -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    holdout_files = (
        "validation.jsonl",
        "adapter_val.jsonl",
        "test_set.jsonl",
        "test_set_with_context.jsonl",
    )
    for rel in holdout_files:
        for row in read_jsonl(dataset_dir / rel):
            excluded.add(pair_key(row_prompt(row), row_completion(row)))
    return excluded


def _source_sample_id(
    collection: str,
    source_index: int,
    prompt: str,
    completion: str,
    row: dict[str, Any],
) -> str:
    value = row.get("sample_id")
    if value:
        return str(value)
    return stable_id("srcsample", collection, source_index, prompt, completion)


def load_candidate_rows(config: SeedConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_rows = read_jsonl(config.dataset_dir / "training.jsonl")
    if not training_rows:
        raise FileNotFoundError(f"no training rows found at {config.dataset_dir / 'training.jsonl'}")

    excluded = load_excluded_pairs(config.dataset_dir)
    train_by_key: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for idx, row in enumerate(training_rows):
        key = pair_key(row_prompt(row), row_completion(row))
        if key in excluded or not all(key):
            continue
        train_by_key.setdefault(key, (idx, row))

    stage2_rows = read_jsonl(config.dataset_dir / "stage2_eval.jsonl")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in stage2_rows:
        key = pair_key(row.get("question"), row.get("answer"))
        if key in seen or key not in train_by_key:
            continue
        source_index, training_row = train_by_key[key]
        prompt = row_prompt(training_row)
        completion = row_completion(training_row)
        candidates.append({
            "source_split": "training",
            "source_row_index": source_index,
            "source_row": training_row,
            "prompt": prompt,
            "completion": completion,
            "system": training_row.get("system"),
            "source_url": row.get("source_url"),
            "domain_slice": row.get("domain_slice") or row.get("product_family") or config.collection,
            "context": row.get("context") or "",
            "stage": row.get("stage"),
            "qa_type": row.get("qa_type"),
            "instr_type": row.get("instr_type"),
            "passage_id": row.get("passage_id"),
            "retrieved_urls": row.get("retrieved_urls") or [],
            "neighbor_urls": row.get("neighbor_urls") or [],
            "source_sample_id": _source_sample_id(
                config.collection,
                source_index,
                prompt,
                completion,
                training_row,
            ),
            "candidate_source": "stage2_eval",
        })
        seen.add(key)

    for key, (source_index, training_row) in train_by_key.items():
        if key in seen:
            continue
        prompt = row_prompt(training_row)
        completion = row_completion(training_row)
        candidates.append({
            "source_split": "training",
            "source_row_index": source_index,
            "source_row": training_row,
            "prompt": prompt,
            "completion": completion,
            "system": training_row.get("system"),
            "source_url": None,
            "domain_slice": config.collection,
            "context": "",
            "stage": None,
            "qa_type": None,
            "instr_type": None,
            "passage_id": None,
            "retrieved_urls": [],
            "neighbor_urls": [],
            "source_sample_id": _source_sample_id(
                config.collection,
                source_index,
                prompt,
                completion,
                training_row,
            ),
            "candidate_source": "training",
        })

    metrics = {
        "training_rows": len(training_rows),
        "excluded_holdout_pairs": len(excluded),
        "eligible_training_pairs": len(train_by_key),
        "stage2_rows": len(stage2_rows),
        "candidate_rows": len(candidates),
        "candidate_source": candidates[0]["candidate_source"] if candidates else None,
        "candidate_source_counts": {
            "stage2_eval": sum(1 for candidate in candidates if candidate["candidate_source"] == "stage2_eval"),
            "training": sum(1 for candidate in candidates if candidate["candidate_source"] == "training"),
        },
    }
    return candidates, metrics


def length_bin(prompt: str, completion: str) -> str:
    chars = len(prompt) + len(completion)
    if chars < 500:
        return "short"
    if chars < 1200:
        return "medium"
    return "long"


def bucket_key(candidate: dict[str, Any]) -> tuple[str, str, str, str]:
    domain_slice = normalize_text(candidate.get("domain_slice") or candidate.get("product_family") or "unknown").lower()
    stage = normalize_text(candidate.get("stage") or "unknown").lower()
    style = normalize_text(candidate.get("qa_type") or candidate.get("instr_type") or "direct").lower()
    size = length_bin(candidate["prompt"], candidate["completion"])
    return (domain_slice, stage, style, size)


def select_candidates(candidates: list[dict[str, Any]], seed_count: int, random_seed: int) -> list[dict[str, Any]]:
    if seed_count <= 0:
        return candidates
    rng = random.Random(random_seed)
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        buckets.setdefault(bucket_key(candidate), []).append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (item["source_row_index"], item["prompt"]))
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while keys and len(selected) < seed_count:
        next_keys: list[tuple[str, str, str, str]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < seed_count:
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def _excerpt(value: str, limit: int = 800) -> str:
    text = normalize_text(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def infer_coverage_axis(candidate: dict[str, Any]) -> str:
    text = f"{candidate['prompt']} {candidate['completion']} {candidate.get('context') or ''}".lower()
    checks = [
        ("troubleshooting", r"\b(error|fail|fails|troubleshoot|issue|resolve|fix)\b"),
        ("deployment", r"\b(kubernetes|helm|deploy|deployment|pod|pvc|service|container)\b"),
        ("api", r"\b(endpoint|api|request|response|openai|grpc|http)\b"),
        ("runtime-profiles", r"\b(profile|gpu|vram|fp8|bf16|lora|tensor parallel|throughput)\b"),
        ("security", r"\b(secret|key|token|license|terms|authentication|authorization)\b"),
        ("observability", r"\b(metric|prometheus|grafana|status|logging|trace)\b"),
    ]
    for axis, pattern in checks:
        if re.search(pattern, text):
            return axis
    return "grounded-qa-variation"


def heuristic_seed_payload(candidate: dict[str, Any], pairs_per_seed: int) -> dict[str, Any]:
    axis = infer_coverage_axis(candidate)
    domain_slice = normalize_text(candidate.get("domain_slice") or candidate.get("product_family") or "domain slice")
    return {
        "coverage_axis": axis,
        "augmentation_intent": (
            f"Generate operationally useful {domain_slice} QA variants that preserve the "
            "same source-grounded facts while varying wording, prerequisites, and "
            "failure-mode framing."
        ),
        "generation_brief": (
            f"Create {pairs_per_seed} grounded {domain_slice} question-answer pairs for "
            f"the {axis} coverage axis. Keep answers specific and supported by the "
            "source text."
        ),
        "pairs_requested": pairs_per_seed,
        "constraints": [
            "answer must be supported by the provided source text",
            "do not introduce claims absent from the source",
            "do not use validation or test-set wording",
            "prefer practical operator-facing questions over generic summaries",
        ],
    }


def build_llm_prompt(candidate: dict[str, Any], pairs_per_seed: int) -> str:
    domain_slice = normalize_text(candidate.get("domain_slice") or candidate.get("product_family") or "domain slice")
    return f"""\
Create one seed instruction record for NeMo Data Designer.

Domain slice: {domain_slice}
Requested synthetic pairs from this seed: {pairs_per_seed}
Source URL: {candidate.get("source_url") or "unknown"}
Stage/style: {candidate.get("stage") or "unknown"} / {candidate.get("qa_type") or candidate.get("instr_type") or "direct"}

Grounded question:
{candidate["prompt"]}

Grounded answer:
{candidate["completion"]}

Grounded source text excerpt:
{_excerpt(candidate.get("context") or "", 3500)}

Return JSON only with these keys:
{{
  "coverage_axis": "deployment|profiles|api|security|operations|troubleshooting|observability|other",
  "augmentation_intent": "one sentence",
  "generation_brief": "one concise instruction for Data Designer",
  "pairs_requested": {pairs_per_seed},
  "constraints": ["short constraint", "short constraint"]
}}
"""


def chat_url(api_url: str) -> str:
    base = api_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_think_tags(text)
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.MULTILINE)
    stripped = re.sub(r"\s*```$", "", stripped, flags=re.MULTILINE)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response was not a JSON object")
    return parsed


def call_llm(config: SeedConfig, prompt: str, api_key: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    import httpx

    payload = {
        "model": config.judge_model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_exc: Exception | None = None
    for attempt in range(max(1, config.max_retries)):
        try:
            with httpx.Client(timeout=config.timeout_s) as client:
                resp = client.post(chat_url(config.judge_api_url), headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()
            choice = body["choices"][0]
            message = choice.get("message") or {}
            raw_response_text = message.get("content") or ""
            response_text = strip_think_tags(raw_response_text)
            reasoning_text = message.get("reasoning") or message.get("reasoning_content") or ""
            metadata = {
                "usage": body.get("usage") or {},
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
                "raw_content_chars": len(raw_response_text),
                "content_chars": len(response_text),
                "inline_think_chars_stripped": max(len(raw_response_text) - len(response_text), 0),
                "reasoning_chars": len(reasoning_text),
                "attempt": attempt + 1,
            }
            return parse_json_object(response_text), response_text, metadata
        except Exception as exc:  # retry covers transport errors and malformed JSON
            last_exc = exc
            if attempt < max(1, config.max_retries) - 1:
                time.sleep(2 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def coerce_seed_payload(payload: dict[str, Any], fallback: dict[str, Any], pairs_per_seed: int) -> dict[str, Any]:
    constraints = payload.get("constraints") or fallback["constraints"]
    if isinstance(constraints, str):
        constraints = [constraints]
    if not isinstance(constraints, list):
        constraints = fallback["constraints"]
    try:
        pairs_requested = int(payload.get("pairs_requested") or fallback["pairs_requested"])
    except (TypeError, ValueError):
        pairs_requested = pairs_per_seed
    return {
        "coverage_axis": normalize_text(payload.get("coverage_axis") or fallback["coverage_axis"]),
        "augmentation_intent": normalize_text(
            payload.get("augmentation_intent") or fallback["augmentation_intent"]
        ),
        "generation_brief": normalize_text(payload.get("generation_brief") or fallback["generation_brief"]),
        "pairs_requested": max(1, pairs_requested),
        "constraints": [normalize_text(item) for item in constraints if normalize_text(item)],
    }


def build_seed_record(
    config: SeedConfig,
    candidate: dict[str, Any],
    sequence_index: int,
    *,
    api_key: str | None,
) -> dict[str, Any]:
    fallback = heuristic_seed_payload(candidate, config.pairs_per_seed)
    llm_prompt = build_llm_prompt(candidate, config.pairs_per_seed)
    llm_response_text = ""
    llm_metadata: dict[str, Any] = {}
    seed_author = "heuristic"
    payload = fallback
    if config.mode == "llm":
        if not api_key:
            raise RuntimeError(f"{config.judge_api_key_env} is required for --mode llm")
        raw_payload, llm_response_text, llm_metadata = call_llm(config, llm_prompt, api_key)
        payload = coerce_seed_payload(raw_payload, fallback, config.pairs_per_seed)
        seed_author = "llm"
    else:
        payload = coerce_seed_payload(payload, fallback, config.pairs_per_seed)

    seed_id = stable_id(
        "ddseed",
        config.collection,
        candidate["source_row_index"],
        candidate["prompt"],
        candidate["completion"],
        sequence_index,
    )
    retrieved_urls = []
    if candidate.get("source_url"):
        retrieved_urls.append(candidate["source_url"])
    retrieved_urls.extend(candidate.get("retrieved_urls") or [])
    retrieved_urls.extend(candidate.get("neighbor_urls") or [])
    retrieved_urls = list(dict.fromkeys(str(url) for url in retrieved_urls if url))

    return {
        "schema_version": SEED_SCHEMA_VERSION,
        "seed_id": seed_id,
        "gap_id": seed_id,
        "collection": config.collection,
        "source_dataset": config.source_dataset_name,
        "source_split": candidate["source_split"],
        "source_row_indices": [candidate["source_row_index"]],
        "source_row_index": candidate["source_row_index"],
        "source_sample_ids": [candidate["source_sample_id"]],
        "source_sample_id": candidate["source_sample_id"],
        "source_candidate_source": candidate["candidate_source"],
        "source_url": candidate.get("source_url"),
        "passage_id": candidate.get("passage_id"),
        "domain_slice": candidate.get("domain_slice") or candidate.get("product_family") or config.collection,
        "source_stage": candidate.get("stage"),
        "qa_type": candidate.get("qa_type"),
        "instr_type": candidate.get("instr_type"),
        "pairs_count": payload["pairs_requested"],
        "pairs_requested": payload["pairs_requested"],
        "coverage_axis": payload["coverage_axis"],
        "augmentation_intent": payload["augmentation_intent"],
        "generation_brief": payload["generation_brief"],
        "constraints": payload["constraints"],
        "constraints_json": payload["constraints"],
        "source_prompt": candidate["prompt"],
        "source_completion": candidate["completion"],
        "source_prompt_excerpt": _excerpt(candidate["prompt"]),
        "source_completion_excerpt": _excerpt(candidate["completion"]),
        "source_context": candidate.get("context") or "",
        "retrieved_chunks": candidate.get("context") or "",
        "retrieved_urls": retrieved_urls,
        "seed_styles": f"Question: {candidate['prompt']}\nAnswer: {candidate['completion']}",
        "seed_author": seed_author,
        "llm": {
            "endpoint": config.judge_api_url if config.mode == "llm" else None,
            "model": config.judge_model if config.mode == "llm" else None,
            "prompt_sha256": sha256_text(llm_prompt),
            "response_sha256": sha256_text(llm_response_text) if llm_response_text else None,
            "max_tokens": config.max_tokens if config.mode == "llm" else None,
            "usage": llm_metadata.get("usage") if llm_metadata else {},
            "finish_reason": llm_metadata.get("finish_reason") if llm_metadata else None,
            "stop_reason": llm_metadata.get("stop_reason") if llm_metadata else None,
            "raw_content_chars": llm_metadata.get("raw_content_chars") if llm_metadata else None,
            "content_chars": llm_metadata.get("content_chars") if llm_metadata else None,
            "inline_think_chars_stripped": llm_metadata.get("inline_think_chars_stripped") if llm_metadata else None,
            "reasoning_chars": llm_metadata.get("reasoning_chars") if llm_metadata else None,
        },
    }


SEED_CSV_FIELDS = [
    "seed_id",
    "gap_id",
    "collection",
    "source_dataset",
    "source_split",
    "source_row_index",
    "source_sample_id",
    "source_url",
    "passage_id",
    "domain_slice",
    "source_stage",
    "qa_type",
    "instr_type",
    "pairs_count",
    "coverage_axis",
    "augmentation_intent",
    "generation_brief",
    "constraints_json",
    "source_prompt",
    "source_completion",
    "retrieved_chunks",
    "retrieved_urls",
    "seed_styles",
    "seed_author",
]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return value


def write_seed_csv(seeds: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SEED_CSV_FIELDS)
        writer.writeheader()
        for seed in seeds:
            writer.writerow({field: _csv_value(seed.get(field)) for field in SEED_CSV_FIELDS})


def artifact_manifest(path: Path, output_dir: Path, artifact_kind: str) -> dict[str, Any]:
    try:
        rel = str(path.relative_to(output_dir))
    except ValueError:
        rel = path.name
    entry: dict[str, Any] = {
        "path": str(path),
        "artifact_path": rel,
        "artifact_kind": artifact_kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        entry["rows"] = count_jsonl_rows(path)
    return entry


def build_submission_plan(
    config: SeedConfig,
    seed_csv: Path,
    seeds: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "collection": config.collection,
        "source_dataset": config.source_dataset_name,
        "status": "prepared",
        "seed_dataset": {
            "local_path": str(seed_csv),
            "repo_id": f"default/{config.collection}-dd-llm-seeds",
            "filename": seed_csv.name,
            "record_count": len(seeds),
        },
        "data_designer": {
            "model": os.getenv("DATA_DESIGNER_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
            "model_alias": os.getenv("DATA_DESIGNER_MODEL_ALIAS", "augmentation_model"),
            "model_provider": os.getenv("DATA_DESIGNER_MODEL_PROVIDER", "system/model-provider"),
            "temperature": float(os.getenv("DATA_DESIGNER_TEMPERATURE", "0.3")),
            "top_p": float(os.getenv("DATA_DESIGNER_TOP_P", "1.0")),
            "max_tokens": int(os.getenv("DATA_DESIGNER_MAX_TOKENS", "2048")),
            "sampling_strategy": os.getenv("DATA_DESIGNER_SAMPLING_STRATEGY", "ordered"),
            "num_records": len(seeds),
            "prompt_column": "qa_pairs_json",
            "prompt_template": DATA_DESIGNER_PROMPT_TEMPLATE,
        },
        "seed_builder": {
            "mode": config.mode,
            "seed_count_requested": config.seed_count,
            "seed_count_selected": len(seeds),
            "pairs_per_seed_requested": config.pairs_per_seed,
            "random_seed": config.random_seed,
            "llm": {
                "endpoint": config.judge_api_url if config.mode == "llm" else None,
                "model": config.judge_model if config.mode == "llm" else None,
                "api_key_env": config.judge_api_key_env if config.mode == "llm" else None,
            },
        },
        "metrics": {
            **metrics,
            "seed_records": len(seeds),
            "synthetic_pairs_requested": sum(int(seed.get("pairs_count") or 0) for seed in seeds),
        },
    }


def write_observability(
    output_dir: Path,
    config: SeedConfig,
    plan: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> None:
    observability_dir = output_dir / "observability" / "data-designer-grounded-seeds"
    write_json(observability_dir / "run_context.json", {
        "schema_version": "observability.v1",
        "pipeline_stage": "data-designer-grounded-seeds",
        "created_at": utc_now(),
        "collection": config.collection,
        "dataset_dir": str(config.dataset_dir),
        "output_dir": str(config.output_dir),
        "mode": config.mode,
        "mlflow": {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
            "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME"),
            "parent_run_id": os.getenv("MLFLOW_PARENT_RUN_ID"),
        },
    })
    write_json(observability_dir / "metrics.json", {
        "seed_records.count": plan["metrics"]["seed_records"],
        "seed_records.candidates": plan["metrics"]["candidate_rows"],
        "seed_records.eligible_training_pairs": plan["metrics"]["eligible_training_pairs"],
        "synthetic_pairs.requested": plan["metrics"]["synthetic_pairs_requested"],
        "holdout_pairs.excluded": plan["metrics"]["excluded_holdout_pairs"],
        "llm.prompt_tokens": plan["metrics"].get("llm.prompt_tokens", 0),
        "llm.completion_tokens": plan["metrics"].get("llm.completion_tokens", 0),
        "llm.total_tokens": plan["metrics"].get("llm.total_tokens", 0),
    })
    write_json(observability_dir / "artifacts_manifest.json", {
        "schema_version": "observability.v1",
        "artifacts": artifacts,
    })


def _build_seed_records(
    config: SeedConfig,
    selected: list[dict[str, Any]],
    api_key: str | None,
) -> list[dict[str, Any]]:
    if config.mode != "llm" or config.max_workers <= 1:
        return [
            build_seed_record(config, candidate, idx, api_key=api_key)
            for idx, candidate in enumerate(selected)
        ]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    seeds_by_index: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {
            pool.submit(build_seed_record, config, candidate, idx, api_key=api_key): idx
            for idx, candidate in enumerate(selected)
        }
        for future in as_completed(futures):
            idx = futures[future]
            seeds_by_index[idx] = future.result()
    return [seeds_by_index[idx] for idx in range(len(selected))]


def _token_usage_metrics(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    reasoning_chars = 0
    content_chars = 0
    llm_records = 0
    for seed in seeds:
        llm = seed.get("llm") if isinstance(seed.get("llm"), dict) else {}
        usage = llm.get("usage") if isinstance(llm.get("usage"), dict) else {}
        if usage:
            llm_records += 1
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        reasoning_chars += int(llm.get("reasoning_chars") or 0)
        content_chars += int(llm.get("content_chars") or 0)
    return {
        "llm.records": llm_records,
        "llm.prompt_tokens": prompt_tokens,
        "llm.completion_tokens": completion_tokens,
        "llm.total_tokens": total_tokens,
        "llm.reasoning_chars": reasoning_chars,
        "llm.content_chars": content_chars,
        "llm.reasoning_overhead_char_ratio": (
            reasoning_chars / content_chars if content_chars else None
        ),
    }


def run(config: SeedConfig) -> dict[str, Any]:
    if config.mode == "llm" and not config.allow_external_llm:
        raise RuntimeError("--mode llm requires --allow-external-llm")
    api_key = os.getenv(config.judge_api_key_env) if config.mode == "llm" else None

    candidates, metrics = load_candidate_rows(config)
    selected = select_candidates(candidates, config.seed_count, config.random_seed)
    seeds = _build_seed_records(config, selected, api_key)
    metrics.update(_token_usage_metrics(seeds))

    source_manifest_path = config.output_dir / "source_snapshot" / "manifest.json"
    write_json(source_manifest_path, source_file_manifest(config.dataset_dir))

    data_designer_dir = config.output_dir / "data_designer"
    seed_requests = data_designer_dir / "llm_seed_requests.jsonl"
    seed_csv = data_designer_dir / "seed_dataset.csv"
    submission_plan = data_designer_dir / "submission_plan.json"
    write_jsonl(seed_requests, seeds)
    write_seed_csv(seeds, seed_csv)
    plan = build_submission_plan(config, seed_csv, seeds, metrics)
    write_json(submission_plan, plan)

    artifacts = [
        artifact_manifest(source_manifest_path, config.output_dir, "source_snapshot_manifest"),
        artifact_manifest(seed_requests, config.output_dir, "data_designer_seed_requests"),
        artifact_manifest(seed_csv, config.output_dir, "data_designer_seed_csv"),
        artifact_manifest(submission_plan, config.output_dir, "data_designer_submission_plan"),
    ]
    write_observability(config.output_dir, config, plan, artifacts)

    return {
        "collection": config.collection,
        "mode": config.mode,
        "candidate_rows": metrics["candidate_rows"],
        "seed_records": len(seeds),
        "synthetic_pairs_requested": plan["metrics"]["synthetic_pairs_requested"],
        "output_dir": str(config.output_dir),
        "seed_requests": str(seed_requests),
        "seed_csv": str(seed_csv),
        "submission_plan": str(submission_plan),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--collection", default=os.getenv("COLLECTION", "nim_curated"))
    ap.add_argument("--source-dataset-name", default=None)
    ap.add_argument("--mode", choices=["prepare", "llm"], default="prepare")
    ap.add_argument("--seed-count", type=int, default=200)
    ap.add_argument("--pairs-per-seed", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--judge-api-url", default=DEFAULT_LLM_URL)
    ap.add_argument("--judge-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--judge-api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    ap.add_argument("--allow-external-llm", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--max-retries", type=int, default=3)
    return ap.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SeedConfig:
    return SeedConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        collection=args.collection,
        source_dataset_name=args.source_dataset_name or args.collection,
        mode=args.mode,
        seed_count=args.seed_count,
        pairs_per_seed=args.pairs_per_seed,
        random_seed=args.random_seed,
        judge_api_url=args.judge_api_url,
        judge_model=args.judge_model,
        judge_api_key_env=args.judge_api_key_env,
        allow_external_llm=args.allow_external_llm,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        max_workers=max(1, args.max_workers),
        max_retries=max(1, args.max_retries),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(config_from_args(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
