#!/usr/bin/env python3
"""Submit and collect NeMo Data Designer gap-fill work.

The Stage 1.5 analysis step writes `data_designer/gapfill_requests.jsonl` as a
seed-record handoff to NeMo Data Designer. This Job turns that handoff into a
Data Designer seed CSV/submission plan, can submit through the NVIDIA SDK when
it is available, and normalizes returned synthetic pairs back into Stage 1.5
KVP rows plus provenance samples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.models import KVPRow  # noqa: E402
from scripts.pipeline.provenance import SCHEMA_VERSION, stable_id, utc_now  # noqa: E402

DEFAULT_DATASET_DIR = Path(os.getenv("DATASET_DIR", "/datasets/nim_curated"))
DEFAULT_OBSERVABILITY_DIR = Path(
    os.getenv("OBSERVABILITY_DIR", "/observability/data-designer-gapfill/nim_curated")
)
DEFAULT_DATA_DESIGNER_URL = os.getenv("NEMO_MICROSERVICES_BASE_URL")
DEFAULT_DATASTORE_ENDPOINT = os.getenv("NEMO_MICROSERVICES_DATASTORE_ENDPOINT")
DEFAULT_MODEL = os.getenv("DATA_DESIGNER_MODEL", "nvidia/nemotron-3-super-120b-a12b")
DEFAULT_MODEL_ALIAS = os.getenv("DATA_DESIGNER_MODEL_ALIAS", "gapfill_model")
DEFAULT_PROVIDER = os.getenv("DATA_DESIGNER_MODEL_PROVIDER", "system/model-provider")
DEFAULT_TEMPERATURE = float(os.getenv("DATA_DESIGNER_TEMPERATURE", "0.3"))
DEFAULT_MAX_TOKENS = int(os.getenv("DATA_DESIGNER_MAX_TOKENS", "2048"))
DEFAULT_TOP_P = float(os.getenv("DATA_DESIGNER_TOP_P", "1.0"))
DEFAULT_SAMPLING_STRATEGY = os.getenv("DATA_DESIGNER_SAMPLING_STRATEGY", "ordered")

_THINK_BALANCED = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TAIL = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_PRELUDE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Strip reasoning-model think blocks before parsing or persisting synthetic rows."""
    if not text:
        return text
    cleaned = _THINK_BALANCED.sub("", text)
    cleaned = _THINK_PRELUDE.sub("", cleaned)
    cleaned = _THINK_OPEN_TAIL.sub("", cleaned)
    return cleaned.strip()


def _clean_generated_text(value: Any) -> str:
    return strip_think_tags(str(value or "")).strip()


PROMPT_TEMPLATE = """\
Gap ID: {{ gap_id }}
Brief: {{ generation_brief }}

Generate {{ pairs_count }} question-answer pairs about {{ product_family }} that are
answerable ONLY from the following retrieved documentation chunks. Vary the
question styles using these examples as reference:
{{ seed_styles }}

Documentation chunks:
{{ retrieved_chunks }}

Return JSON only in this shape:
{"pairs": [{"question": "...", "answer": "..."}]}
"""


@dataclass(frozen=True)
class DataDesignerConfig:
    dataset_dir: Path
    collection: str
    observability_dir: Path
    mode: str
    data_designer_url: str | None
    datastore_endpoint: str | None
    seed_repo_id: str
    seed_filename: str
    model: str
    model_alias: str
    model_provider: str
    temperature: float
    top_p: float
    max_tokens: int
    sampling_strategy: str
    wait: bool
    job_id: str | None = None
    results_jsonl: Path | None = None
    results_dir: Path | None = None


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def artifact_manifest(path: Path, dataset_dir: Path, artifact_kind: str) -> dict[str, Any]:
    try:
        artifact_path = str(path.relative_to(dataset_dir))
    except ValueError:
        artifact_path = path.name
    manifest: dict[str, Any] = {
        "path": str(path),
        "artifact_path": artifact_path,
        "artifact_kind": artifact_kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        manifest["rows"] = count_jsonl_rows(path)
    return manifest


def load_gapfill_requests(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "data_designer" / "gapfill_requests.jsonl"
    requests = read_jsonl(path)
    if not requests:
        raise FileNotFoundError(f"no Data Designer gap-fill requests found at {path}")
    return requests


def load_gap_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "provenance" / "gap_manifest.json"
    manifest = read_json_if_exists(path)
    if not manifest:
        raise FileNotFoundError(f"no gap manifest found at {path}")
    return manifest


def _as_json_text(value: Any) -> str:
    return json.dumps(value if value is not None else [], sort_keys=True)


def flatten_request_for_seed(request: dict[str, Any]) -> dict[str, Any]:
    input_payload = request.get("input") if isinstance(request.get("input"), dict) else {}
    return {
        "gap_id": request.get("gap_id") or input_payload.get("gap_id"),
        "gap_manifest_id": request.get("gap_manifest_id"),
        "dataset_version_id": request.get("dataset_version_id"),
        "recipe_name": request.get("recipe_name"),
        "product_family": input_payload.get("product_family"),
        "pairs_count": input_payload.get("pairs_count"),
        "pairs_needed": request.get("pairs_needed"),
        "num_records": request.get("num_records"),
        "pairs_per_record": request.get("pairs_per_record"),
        "seed_styles": input_payload.get("seed_styles") or "",
        "retrieved_chunks": input_payload.get("retrieved_chunks") or "",
        "generation_brief": input_payload.get("generation_brief") or "",
        "retrieved_urls": _as_json_text(request.get("retrieved_urls", [])),
        "seed_entailment_ids": _as_json_text(request.get("seed_entailment_ids", [])),
        "seed_chunk_ids": _as_json_text(request.get("seed_chunk_ids", [])),
    }


def write_seed_csv(requests: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(flatten_request_for_seed(requests[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for request in requests:
            writer.writerow(flatten_request_for_seed(request))


def build_submission_plan(
    config: DataDesignerConfig,
    gap_manifest: dict[str, Any],
    requests: list[dict[str, Any]],
    seed_csv: Path,
) -> dict[str, Any]:
    record_count = len(requests)
    records_requested = sum(int(request.get("num_records") or 0) for request in requests)
    pairs_requested = sum(int(request.get("pairs_needed") or 0) for request in requests)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "collection": config.collection,
        "gap_manifest_id": gap_manifest.get("gap_manifest_id"),
        "dataset_version_id": gap_manifest.get("dataset_version_id"),
        "status": "planned",
        "seed_dataset": {
            "local_path": str(seed_csv),
            "repo_id": config.seed_repo_id,
            "filename": config.seed_filename,
            "record_count": record_count,
            "datastore_endpoint": config.datastore_endpoint,
        },
        "data_designer": {
            "base_url": config.data_designer_url,
            "model": config.model,
            "model_alias": config.model_alias,
            "model_provider": config.model_provider,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "sampling_strategy": config.sampling_strategy,
            "num_records": records_requested or record_count,
            "prompt_column": "qa_pairs_json",
            "prompt_template": PROMPT_TEMPLATE,
        },
        "metrics": {
            "gaps_requested": len({request.get("gap_id") for request in requests}),
            "seed_records": record_count,
            "records_requested": records_requested,
            "pairs_requested": pairs_requested,
        },
    }


def _extract_job_id(value: Any) -> str:
    for attr in ("id", "job_id", "name"):
        item = getattr(value, attr, None)
        if item:
            return str(item)
    if isinstance(value, dict):
        for key in ("id", "job_id", "name"):
            if value.get(key):
                return str(value[key])
    return str(value)


def submit_with_nemo_microservices_sdk(
    config: DataDesignerConfig,
    seed_csv: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    if not config.data_designer_url:
        raise RuntimeError("NEMO_MICROSERVICES_BASE_URL or --data-designer-url is required")
    if not config.datastore_endpoint:
        raise RuntimeError("NEMO_MICROSERVICES_DATASTORE_ENDPOINT or --datastore-endpoint is required")
    try:
        from nemo_microservices.data_designer.essentials import (  # type: ignore
            DataDesignerConfigBuilder,
            LLMTextColumnConfig,
            ModelConfig,
            NeMoDataDesignerClient,
        )
        try:
            from nemo_microservices.data_designer.essentials import (  # type: ignore
                InferenceParameters,
            )
        except ImportError:
            from nemo_microservices.data_designer.essentials import (  # type: ignore
                ChatCompletionInferenceParams as InferenceParameters,
            )
    except ImportError as exc:
        raise RuntimeError(
            "NeMo Data Designer SDK is not installed in this image. Install the "
            "NVIDIA NeMo Microservices Python SDK or run --mode prepare/collect."
        ) from exc

    client = NeMoDataDesignerClient(base_url=config.data_designer_url)
    seed_reference = client.upload_seed_dataset(
        dataset=seed_csv,
        repo_id=config.seed_repo_id,
        datastore_settings={"endpoint": config.datastore_endpoint},
    )
    model_config = ModelConfig(
        alias=config.model_alias,
        model=config.model,
        inference_parameters=InferenceParameters(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        ),
    )
    builder = DataDesignerConfigBuilder(model_configs=[model_config])
    builder.with_seed_dataset(
        dataset_reference=seed_reference,
        sampling_strategy=config.sampling_strategy,
    )
    builder.add_column(
        LLMTextColumnConfig(
            name="qa_pairs_json",
            prompt=PROMPT_TEMPLATE,
            model_alias=config.model_alias,
        )
    )
    job = client.create(
        builder,
        num_records=plan["data_designer"]["num_records"],
        wait_until_done=config.wait,
    )
    return {
        "job_id": _extract_job_id(job),
        "status": "submitted",
        "submitted_at": utc_now(),
        "seed_dataset_reference": str(seed_reference),
    }


def _parse_json_object(value: str) -> Any:
    text = strip_think_tags(value)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return value
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return value


def _list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = _parse_json_object(value)
        if parsed is not value:
            return _list_values(parsed)
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def extract_pairs(record: dict[str, Any]) -> list[dict[str, str]]:
    candidates = [
        record.get("pairs"),
        record.get("qa_pairs_json"),
        record.get("generated_content"),
        record.get("generated"),
        record.get("response"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        parsed = _parse_json_object(candidate) if isinstance(candidate, str) else candidate
        if isinstance(parsed, dict) and isinstance(parsed.get("pairs"), list):
            return [
                {
                    "question": _clean_generated_text(item.get("question")),
                    "answer": _clean_generated_text(item.get("answer")),
                }
                for item in parsed["pairs"]
                if _clean_generated_text(item.get("question")) and _clean_generated_text(item.get("answer"))
            ]
        if isinstance(parsed, list):
            return [
                {
                    "question": _clean_generated_text(item.get("question")),
                    "answer": _clean_generated_text(item.get("answer")),
                }
                for item in parsed
                if (
                    isinstance(item, dict)
                    and _clean_generated_text(item.get("question"))
                    and _clean_generated_text(item.get("answer"))
                )
            ]
    if record.get("question") and record.get("answer"):
        return [{
            "question": _clean_generated_text(record["question"]),
            "answer": _clean_generated_text(record["answer"]),
        }]
    if record.get("prompt") and record.get("completion"):
        return [{
            "question": _clean_generated_text(record["prompt"]),
            "answer": _clean_generated_text(record["completion"]),
        }]
    return []


def load_generated_records(results_jsonl: Path | None, results_dir: Path | None) -> list[dict[str, Any]]:
    if results_jsonl:
        return read_jsonl(results_jsonl)
    if not results_dir:
        return []
    candidates = [
        results_dir / "dataset.jsonl",
        results_dir / "generated_samples.jsonl",
        results_dir / "results.jsonl",
        results_dir / "dataset.json",
        results_dir / "dataset.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            return read_jsonl(path)
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict) and isinstance(data.get("rows"), list):
                return [item for item in data["rows"] if isinstance(item, dict)]
        if path.suffix == ".csv":
            with path.open(newline="") as f:
                return list(csv.DictReader(f))
    return []


def _request_index(requests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(request.get("gap_id")): request for request in requests if request.get("gap_id")}


def _merged_seed_record(record: dict[str, Any], requests_by_gap: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gap_id = str(record.get("gap_id") or "")
    request = requests_by_gap.get(gap_id, {})
    seed = flatten_request_for_seed(request) if request else {}
    merged = dict(seed)
    merged.update({key: value for key, value in record.items() if value not in (None, "")})
    return merged


def normalize_generated_records(
    records: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    *,
    collection: str,
    data_designer_job_id: str,
) -> tuple[list[KVPRow], list[dict[str, Any]]]:
    requests_by_gap = _request_index(requests)
    rows: list[KVPRow] = []
    samples: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        seed = _merged_seed_record(record, requests_by_gap)
        gap_id = str(seed.get("gap_id") or stable_id("gap", collection, record_index))
        product_family = str(seed.get("product_family") or collection)
        retrieved_chunks = str(seed.get("retrieved_chunks") or "")
        retrieved_urls = _list_values(seed.get("retrieved_urls"))
        seed_entailment_ids = _list_values(seed.get("seed_entailment_ids"))
        seed_chunk_ids = _list_values(seed.get("seed_chunk_ids"))
        seed_sample_ids = _list_values(seed.get("seed_sample_ids"))
        for pair_index, pair in enumerate(extract_pairs(record)):
            sample_id = stable_id(
                "sample",
                "data_designer_gapfill",
                data_designer_job_id,
                gap_id,
                pair_index,
                pair["question"],
                pair["answer"],
            )
            row = KVPRow(
                passage_id=f"gapfill#{gap_id}",
                source_url="<data-designer-synthetic>",
                product_family=product_family,
                stage="1.5",
                target_product_family=product_family,
                sample_id=sample_id,
                question=pair["question"].strip(),
                answer=pair["answer"].strip(),
                context=retrieved_chunks[:5000],
                retrieved_urls=retrieved_urls,
                refined=False,
                entailment_id=None,
                source_revision_ids=[],
                source_chunk_ids=seed_chunk_ids,
                source_systems=["data_designer"],
                source_kinds=["synthetic_gapfill"],
                modalities=["text"],
            )
            rows.append(row)
            samples.append({
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "origin": "synthetic_gapfill",
                "task_type": "qa",
                "prompt": row.question,
                "completion": row.answer,
                "system": None,
                "lineage": {
                    "entailment_ids": seed_entailment_ids,
                    "source_revision_ids": [],
                    "source_chunk_ids": seed_chunk_ids,
                    "source_systems": ["data_designer"],
                    "source_kinds": ["synthetic_gapfill"],
                    "modalities": ["text"],
                    "gap_id": gap_id,
                    "data_designer_job_id": data_designer_job_id,
                    "seed_sample_ids": seed_sample_ids,
                },
                "quality": {
                    "curator_job_id": None,
                    "judge_model": None,
                    "grounding_score": None,
                },
                "metadata": {
                    "collection": collection,
                    "product_family": product_family,
                    "source_url": "<data-designer-synthetic>",
                    "retrieved_urls": retrieved_urls,
                    "gap_manifest_id": seed.get("gap_manifest_id"),
                    "recipe_name": seed.get("recipe_name"),
                    "record_index": record_index,
                    "pair_index": pair_index,
                },
            })
    return rows, samples


def write_result_artifacts(
    config: DataDesignerConfig,
    rows: list[KVPRow],
    samples: list[dict[str, Any]],
    job_info: dict[str, Any],
    gap_manifest: dict[str, Any],
) -> dict[str, Any]:
    dataset_dir = config.dataset_dir
    data_designer_dir = dataset_dir / "data_designer"
    provenance_dir = dataset_dir / "provenance"
    data_designer_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    stage_rows_path = dataset_dir / "stage1_5_gapfill.jsonl"
    with stage_rows_path.open("w") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    generated_path = data_designer_dir / "generated_samples.jsonl"
    provenance_samples_path = provenance_dir / "data_designer_samples.jsonl"
    write_jsonl(generated_path, samples)
    write_jsonl(provenance_samples_path, samples)

    result_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "collection": config.collection,
        "gap_manifest_id": gap_manifest.get("gap_manifest_id"),
        "dataset_version_id": gap_manifest.get("dataset_version_id"),
        "data_designer_job_id": job_info.get("job_id"),
        "status": job_info.get("status", "collected"),
        "stage1_5_rows_uri": "stage1_5_gapfill.jsonl",
        "generated_samples_uri": "data_designer/generated_samples.jsonl",
        "provenance_samples_uri": "provenance/data_designer_samples.jsonl",
        "generated_sample_count": len(samples),
        "stage1_5_row_count": len(rows),
    }
    write_json(data_designer_dir / "result_manifest.json", result_manifest)
    return result_manifest


def collect_artifact_paths(dataset_dir: Path, mode: str) -> list[tuple[Path, str]]:
    candidates = [
        (dataset_dir / "data_designer" / "seed_dataset.csv", "data_designer_seed_dataset"),
        (dataset_dir / "data_designer" / "submission_plan.json", "data_designer_submission_plan"),
        (dataset_dir / "data_designer" / "request_manifest.json", "data_designer_request_manifest"),
    ]
    if mode in {"collect", "submit-and-collect"}:
        candidates.extend([
            (dataset_dir / "data_designer" / "result_manifest.json", "data_designer_result_manifest"),
            (dataset_dir / "data_designer" / "generated_samples.jsonl", "data_designer_generated_samples"),
            (dataset_dir / "provenance" / "data_designer_samples.jsonl", "data_designer_provenance_samples"),
            (dataset_dir / "stage1_5_gapfill.jsonl", "stage1_5_synthetic_rows"),
        ])
    return [(path, kind) for path, kind in candidates if path.exists()]


def build_observability_documents(
    config: DataDesignerConfig,
    gap_manifest: dict[str, Any],
    requests: list[dict[str, Any]],
    plan: dict[str, Any],
    job_info: dict[str, Any],
    generated_count: int,
) -> dict[str, dict[str, Any]]:
    records_requested = plan["metrics"]["records_requested"]
    pairs_requested = plan["metrics"]["pairs_requested"]
    metrics: dict[str, int | float] = {
        "gapfill.gaps.requested": plan["metrics"]["gaps_requested"],
        "gapfill.seed_records.count": plan["metrics"]["seed_records"],
        "gapfill.records.requested": records_requested,
        "gapfill.pairs.requested": pairs_requested,
        "gapfill.samples.generated": generated_count,
        "gapfill.jobs.submitted.count": 1 if job_info.get("status") == "submitted" else 0,
    }
    metrics["gapfill.acceptance_rate"] = (
        round(generated_count / pairs_requested, 6) if pairs_requested else 0
    )
    artifacts = [
        artifact_manifest(path, config.dataset_dir, kind)
        for path, kind in collect_artifact_paths(config.dataset_dir, config.mode)
    ]
    run_context = {
        "schema_version": "observability.v1",
        "pipeline_stage": "data-designer-gapfill",
        "created_at": utc_now(),
        "collection": config.collection,
        "dataset_dir": str(config.dataset_dir),
        "mode": config.mode,
        "gap_manifest_id": gap_manifest.get("gap_manifest_id"),
        "dataset_version_id": gap_manifest.get("dataset_version_id"),
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
        "mlflow": {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
            "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME"),
            "parent_run_id": os.getenv("MLFLOW_PARENT_RUN_ID"),
        },
    }
    service_refs = {
        "schema_version": "observability.v1",
        "data_designer": {
            "base_url": config.data_designer_url,
            "job_id": job_info.get("job_id"),
            "job_status": job_info.get("status"),
            "seed_repo_id": config.seed_repo_id,
            "seed_filename": config.seed_filename,
        },
        "gap_manifest": {
            "gap_manifest_id": gap_manifest.get("gap_manifest_id"),
            "gap_count": len(gap_manifest.get("gaps", [])),
        },
        "requests": {
            "count": len(requests),
            "uri": "data_designer/gapfill_requests.jsonl",
        },
    }
    return {
        "run_context.json": run_context,
        "metrics.json": metrics,
        "artifacts_manifest.json": {"schema_version": "observability.v1", "artifacts": artifacts},
        "service_refs.json": service_refs,
    }


def write_observability_documents(out_dir: Path, documents: dict[str, dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in documents.items():
        write_json(out_dir / name, payload)


def run(config: DataDesignerConfig) -> dict[str, Any]:
    gap_manifest = load_gap_manifest(config.dataset_dir)
    requests = load_gapfill_requests(config.dataset_dir)
    data_designer_dir = config.dataset_dir / "data_designer"
    seed_csv = data_designer_dir / "seed_dataset.csv"
    write_seed_csv(requests, seed_csv)
    plan = build_submission_plan(config, gap_manifest, requests, seed_csv)
    write_json(data_designer_dir / "submission_plan.json", plan)

    job_info = {
        "job_id": config.job_id,
        "status": "planned",
        "created_at": utc_now(),
    }
    if config.mode in {"submit", "submit-and-collect"}:
        job_info = submit_with_nemo_microservices_sdk(config, seed_csv, plan)
        plan["status"] = job_info["status"]
        plan["data_designer"]["job_id"] = job_info["job_id"]
        write_json(data_designer_dir / "submission_plan.json", plan)

    generated_count = 0
    if config.mode in {"collect", "submit-and-collect"}:
        records = load_generated_records(config.results_jsonl, config.results_dir)
        if not records:
            raise FileNotFoundError(
                "no Data Designer generated records found; provide --results-jsonl or --results-dir"
            )
        job_id = config.job_id or job_info.get("job_id") or stable_id(
            "ddjob",
            gap_manifest.get("gap_manifest_id"),
            len(records),
        )
        job_info = {**job_info, "job_id": job_id, "status": "collected"}
        rows, samples = normalize_generated_records(
            records,
            requests,
            collection=config.collection,
            data_designer_job_id=job_id,
        )
        write_result_artifacts(config, rows, samples, job_info, gap_manifest)
        generated_count = len(samples)

    documents = build_observability_documents(
        config,
        gap_manifest,
        requests,
        plan,
        job_info,
        generated_count,
    )
    write_observability_documents(config.observability_dir, documents)
    return {
        "mode": config.mode,
        "gap_manifest_id": gap_manifest.get("gap_manifest_id"),
        "requests": len(requests),
        "job": job_info,
        "generated_samples": generated_count,
        "observability_dir": str(config.observability_dir),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--collection", default=os.getenv("COLLECTION"))
    ap.add_argument("--observability-dir", type=Path, default=DEFAULT_OBSERVABILITY_DIR)
    ap.add_argument(
        "--mode",
        choices=["prepare", "submit", "collect", "submit-and-collect"],
        default=os.getenv("DATA_DESIGNER_GAPFILL_MODE", "prepare"),
    )
    ap.add_argument("--data-designer-url", default=DEFAULT_DATA_DESIGNER_URL)
    ap.add_argument("--datastore-endpoint", default=DEFAULT_DATASTORE_ENDPOINT)
    ap.add_argument("--seed-repo-id", default=os.getenv("DATA_DESIGNER_SEED_REPO_ID"))
    ap.add_argument("--seed-filename", default=os.getenv("DATA_DESIGNER_SEED_FILENAME", "gapfill_requests.csv"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    ap.add_argument("--model-provider", default=DEFAULT_PROVIDER)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--sampling-strategy", default=DEFAULT_SAMPLING_STRATEGY)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--job-id", default=os.getenv("DATA_DESIGNER_JOB_ID"))
    ap.add_argument("--results-jsonl", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    collection = args.collection or args.dataset_dir.name
    seed_repo_id = args.seed_repo_id or f"default/{collection}-gapfill-seeds"
    config = DataDesignerConfig(
        dataset_dir=args.dataset_dir,
        collection=collection,
        observability_dir=args.observability_dir,
        mode=args.mode,
        data_designer_url=args.data_designer_url,
        datastore_endpoint=args.datastore_endpoint,
        seed_repo_id=seed_repo_id,
        seed_filename=args.seed_filename,
        model=args.model,
        model_alias=args.model_alias,
        model_provider=args.model_provider,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        sampling_strategy=args.sampling_strategy,
        wait=args.wait,
        job_id=args.job_id,
        results_jsonl=args.results_jsonl,
        results_dir=args.results_dir,
    )
    result = run(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
