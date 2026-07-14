#!/usr/bin/env python3
"""Normalize grounded Data Designer augmentation results.

This collector turns a native NeMo Data Designer result dataset into the
``accepted_samples.jsonl`` contract consumed by
``build_augmented_dataset_from_synthetic.py``. It preserves source-row lineage,
strips reasoning side channels from trainable text, rejects exact duplicates,
and writes observability artifacts for MLflow-style export.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.build_data_designer_seed_from_grounded import (  # noqa: E402
    count_jsonl_rows,
    sha256_file,
    strip_think_tags,
)
from scripts.pipeline.data_designer_gapfill import extract_pairs as _extract_pairs  # noqa: E402
from scripts.pipeline.provenance import SCHEMA_VERSION, stable_id, utc_now  # noqa: E402

DEFAULT_SOURCE_DATASET_DIR = Path(os.getenv("SOURCE_DATASET_DIR", "<DATASET_ROOT>/nim_curated"))
DEFAULT_EXPERIMENT_DIR = Path(
    os.getenv(
        "DATA_DESIGNER_AUGMENTATION_DIR",
        "<DATASET_ROOT>/experiments/nim_curated_dd_llm_5x",
    )
)
DEFAULT_SYSTEM_PROMPT = "You are a precise NVIDIA NIM technical assistant. Answer based on official documentation."

RESULT_RECORD_FILES = (
    "dataset.jsonl",
    "generated_samples.jsonl",
    "results.jsonl",
    "dataset.json",
    "dataset.csv",
)
SOURCE_PAIR_FILES = (
    "training.jsonl",
    "adapter_train.jsonl",
    "validation.jsonl",
    "adapter_val.jsonl",
    "test_set.jsonl",
    "test_set_with_context.jsonl",
    "stage2_eval.jsonl",
)


@dataclass(frozen=True)
class CollectConfig:
    source_dataset_dir: Path
    experiment_dir: Path
    collection: str
    job_id: str | None
    results_jsonl: Path | None
    results_dir: Path | None
    model_provider: str | None
    model: str | None
    system_prompt: str
    overwrite: bool = False


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def pair_key(prompt: Any, completion: Any) -> tuple[str, str]:
    return (normalize_text(prompt), normalize_text(completion))


def row_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("question") or "")


def row_completion(row: dict[str, Any]) -> str:
    return str(row.get("completion") or row.get("answer") or "")


def load_source_pairs(source_dataset_dir: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for rel in SOURCE_PAIR_FILES:
        for row in read_jsonl(source_dataset_dir / rel):
            key = pair_key(row_prompt(row), row_completion(row))
            if all(key):
                pairs.add(key)
    return pairs


def _json_loads_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _list_values(value: Any) -> list[str]:
    parsed = _json_loads_maybe(value)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [str(parsed)] if str(parsed) else []


def extract_pairs(record: dict[str, Any]) -> list[dict[str, str]]:
    pairs = _extract_pairs(record)
    if pairs:
        return pairs
    dotted_pairs = record.get("qa_pairs_json.pairs")
    if dotted_pairs is None:
        return []
    parsed = _json_loads_maybe(dotted_pairs)
    if not isinstance(parsed, list):
        return []
    output: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = strip_think_tags(str(item.get("question") or "")).strip()
        answer = strip_think_tags(str(item.get("answer") or "")).strip()
        if question and answer:
            output.append({"question": question, "answer": answer})
    return output


def _read_records_with_duckdb(paths: list[Path]) -> list[dict[str, Any]]:
    import duckdb

    con = duckdb.connect()
    path_args = ", ".join(json.dumps(str(path)) for path in paths)
    rel = con.sql(f"select * from read_parquet([{path_args}])")
    return rel.fetchdf().to_dict("records")


def _read_records_with_pandas(path: Path, engine: str) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_parquet(path, engine=engine).to_dict("records")


def load_parquet_records(parquet_paths: list[Path]) -> list[dict[str, Any]]:
    if not parquet_paths:
        return []
    errors: list[str] = []
    try:
        rows: list[dict[str, Any]] = []
        for path in parquet_paths:
            rows.extend(_read_records_with_duckdb([path]))
        return rows
    except Exception as exc:
        errors.append(f"duckdb: {type(exc).__name__}: {exc}")
    rows: list[dict[str, Any]] = []
    for engine in ("fastparquet", "pyarrow"):
        try:
            rows = []
            for path in parquet_paths:
                rows.extend(_read_records_with_pandas(path, engine))
            return rows
        except Exception as exc:
            errors.append(f"pandas/{engine}: {type(exc).__name__}: {exc}")
    raise RuntimeError("failed to read Data Designer parquet results: " + " | ".join(errors))


def load_generated_records(results_jsonl: Path | None, results_dir: Path | None) -> list[dict[str, Any]]:
    if results_jsonl:
        return read_jsonl(results_jsonl)
    if not results_dir:
        return []
    if results_dir.is_file():
        if results_dir.suffix == ".jsonl":
            return read_jsonl(results_dir)
        if results_dir.suffix == ".json":
            data = json.loads(results_dir.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else list(data.get("rows", []))
        if results_dir.suffix == ".csv":
            with results_dir.open(newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
    for rel in RESULT_RECORD_FILES:
        path = results_dir / rel
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            return read_jsonl(path)
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else list(data.get("rows", []))
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
    parquet_paths = sorted((results_dir / "dataset").glob("*.parquet"))
    if not parquet_paths:
        parquet_paths = sorted(results_dir.rglob("*.parquet"))
    return load_parquet_records(parquet_paths)


def result_repo_dir(config: CollectConfig, job_id: str) -> Path | None:
    if config.results_dir:
        return config.results_dir
    manifest = read_json(config.experiment_dir / "data_designer" / "download_manifest.json")
    direct = manifest.get("direct_result_repo") if isinstance(manifest.get("direct_result_repo"), dict) else {}
    path = direct.get("path")
    if path:
        return Path(path)
    candidate = config.experiment_dir / "data_designer" / "results" / job_id / "repo"
    return candidate if candidate.exists() else None


def resolve_job_id(config: CollectConfig) -> str:
    if config.job_id:
        return config.job_id
    response = read_json(config.experiment_dir / "data_designer" / "job_response.json")
    if response.get("id"):
        return str(response["id"])
    terminal = read_json(config.experiment_dir / "data_designer" / "job_terminal_status.json")
    if terminal.get("job_id"):
        return str(terminal["job_id"])
    raise RuntimeError("--job-id is required when job_response.json is unavailable")


def resolve_model_info(config: CollectConfig) -> tuple[str | None, str | None]:
    if config.model_provider or config.model:
        return config.model_provider, config.model
    request = read_json(config.experiment_dir / "data_designer" / "job_request.json")
    configs = (((request.get("spec") or {}).get("config") or {}).get("model_configs") or [])
    if configs:
        model_config = configs[0]
        return model_config.get("provider"), model_config.get("model")
    plan = read_json(config.experiment_dir / "data_designer" / "submission_plan.json")
    data_designer = plan.get("data_designer") if isinstance(plan.get("data_designer"), dict) else {}
    return data_designer.get("model_provider"), data_designer.get("model")


def read_analysis(result_dir: Path | None) -> dict[str, Any]:
    if not result_dir:
        return {}
    analysis = result_dir / "analysis"
    if not analysis.exists():
        return {}
    return read_json(analysis)


def token_metrics_from_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    records = int(analysis.get("num_records") or 0)
    for column in analysis.get("column_statistics") or []:
        if column.get("column_name") != "qa_pairs_json":
            continue
        prompt_mean = float(column.get("prompt_tokens_mean") or 0)
        completion_mean = float(column.get("completion_tokens_mean") or 0)
        return {
            "prompt_tokens_mean": column.get("prompt_tokens_mean"),
            "prompt_tokens_median": column.get("prompt_tokens_median"),
            "completion_tokens_mean": column.get("completion_tokens_mean"),
            "completion_tokens_median": column.get("completion_tokens_median"),
            "estimated_prompt_tokens": round(prompt_mean * records),
            "estimated_completion_tokens": round(completion_mean * records),
            "estimated_total_tokens": round((prompt_mean + completion_mean) * records),
        }
    return {}


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def seed_index(experiment_dir: Path) -> dict[str, dict[str, Any]]:
    seeds = read_jsonl(experiment_dir / "data_designer" / "llm_seed_requests.jsonl")
    return {str(seed.get("seed_id") or seed.get("gap_id")): seed for seed in seeds}


def merged_seed(record: dict[str, Any], seeds_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    seed_id = str(record.get("seed_id") or record.get("gap_id") or "")
    seed = dict(seeds_by_id.get(seed_id, {}))
    seed.update({key: value for key, value in record.items() if value not in (None, "")})
    return seed


def build_artifact(path: Path, base_dir: Path, artifact_kind: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "relative_path": str(path.relative_to(base_dir)),
        "artifact_kind": artifact_kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        item["rows"] = count_jsonl_rows(path)
    return item


def normalize_records(
    records: list[dict[str, Any]],
    *,
    config: CollectConfig,
    job_id: str,
    model_provider: str | None,
    model: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    grounded_pairs = load_source_pairs(config.source_dataset_dir)
    synthetic_pairs: set[tuple[str, str]] = set()
    seeds_by_id = seed_index(config.experiment_dir)
    raw_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for record_index, record in enumerate(records):
        seed = merged_seed(record, seeds_by_id)
        seed_id = str(seed.get("seed_id") or seed.get("gap_id") or stable_id("ddseed", config.collection, record_index))
        for pair_index, pair in enumerate(extract_pairs(record)):
            prompt = strip_think_tags(pair["question"]).strip()
            completion = strip_think_tags(pair["answer"]).strip()
            key = pair_key(prompt, completion)
            reasons: list[str] = []
            if not prompt or not completion:
                reasons.append("empty_prompt_or_completion")
            if key in grounded_pairs:
                reasons.append("duplicate_with_grounded_or_eval_dataset")
            if key in synthetic_pairs:
                reasons.append("duplicate_with_synthetic_dataset")
            status = "rejected" if reasons else "accepted"
            for reason in reasons:
                rejection_counts[reason] += 1
            if not reasons:
                synthetic_pairs.add(key)

            sample_id = stable_id(
                "sample",
                "data_designer_grounded_augmentation",
                job_id,
                seed_id,
                pair_index,
                prompt,
                completion,
            )
            source = {
                "source_dataset": seed.get("source_dataset") or config.collection,
                "source_split": seed.get("source_split") or "training",
                "source_row_index": _int_or_none(seed.get("source_row_index")),
                "source_sample_id": seed.get("source_sample_id"),
                "source_url": seed.get("source_url"),
                "passage_id": seed.get("passage_id"),
                "retrieved_urls": _list_values(seed.get("retrieved_urls")),
            }
            metadata = {
                "collection": seed.get("collection") or config.collection,
                "domain_slice": seed.get("domain_slice") or seed.get("product_family") or config.collection,
                "source_stage": seed.get("source_stage"),
                "qa_type": seed.get("qa_type"),
                "instr_type": seed.get("instr_type"),
                "pairs_count": _int_or_none(seed.get("pairs_count")),
                "coverage_axis": seed.get("coverage_axis"),
                "augmentation_intent": seed.get("augmentation_intent"),
                "generation_brief": seed.get("generation_brief"),
            }
            lineage = {
                "column": "qa_pairs_json",
                "job_id": job_id,
                "model_provider": model_provider,
                "model": model,
                "record_index": record_index,
                "pair_index": pair_index,
                "seed_id": seed_id,
                "reasoning_trace_excluded": "qa_pairs_json__reasoning_trace" in record,
            }
            trainable = {
                "prompt": prompt,
                "completion": completion,
                "system": str(seed.get("system") or config.system_prompt),
                "metadata": {
                    **metadata,
                    "source": source,
                    "lineage": lineage,
                    "sample_id": sample_id,
                },
            }
            quality = {
                "admission_checks": [
                    "non_empty",
                    "exact_duplicate_grounded_eval",
                    "exact_duplicate_synthetic",
                    "training_split_only",
                    "think_tag_stripped",
                ],
                "curator_job_id": None,
                "judge_model": None,
                "source_support_check": "not_run",
            }
            raw = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "status": status,
                "rejection_reasons": reasons,
                "prompt": prompt,
                "completion": completion,
                "system": trainable["system"],
                "source": source,
                "metadata": metadata,
                "data_designer": lineage,
                "quality": quality,
            }
            raw_rows.append(raw)
            if status == "accepted":
                accepted_rows.append(trainable)
                provenance_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "origin": "synthetic_augmentation",
                    "task_type": "qa",
                    "prompt": prompt,
                    "completion": completion,
                    "system": trainable["system"],
                    "lineage": {
                        "data_designer_job_id": job_id,
                        "seed_id": seed_id,
                        "seed_author": seed.get("seed_author"),
                        "source_dataset": source["source_dataset"],
                        "source_split": source["source_split"],
                        "source_row_index": source["source_row_index"],
                        "source_sample_id": source["source_sample_id"],
                        "source_url": source["source_url"],
                        "passage_id": source["passage_id"],
                        "retrieved_urls": source["retrieved_urls"],
                        "source_systems": ["data_designer", str(model_provider or "unknown")],
                        "source_kinds": ["synthetic_augmentation"],
                        "modalities": ["text"],
                    },
                    "metadata": metadata,
                    "quality": quality,
                })
    return raw_rows, accepted_rows, provenance_rows, rejection_counts


def write_observability(
    config: CollectConfig,
    result: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> None:
    out_dir = config.experiment_dir / "observability" / "data-designer-synthetic-generation"
    write_json(out_dir / "run_context.json", {
        "schema_version": "observability.v1",
        "pipeline_stage": "data-designer-synthetic-generation",
        "created_at": utc_now(),
        "collection": config.collection,
        "source_dataset_dir": str(config.source_dataset_dir),
        "experiment_dir": str(config.experiment_dir),
        "mlflow": {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
            "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME"),
            "parent_run_id": os.getenv("MLFLOW_PARENT_RUN_ID"),
        },
    })
    write_json(out_dir / "metrics.json", {
        "synthetic.seed_records.requested": result["seed_records_requested"],
        "synthetic.seed_records.returned": result["seed_records_returned"],
        "synthetic.seed_records.omitted": result["seed_records_omitted"],
        "synthetic.pairs.raw": result["raw_synthetic_pair_count"],
        "synthetic.pairs.accepted": result["accepted_synthetic_pair_count"],
        "synthetic.pairs.rejected": result["rejected_synthetic_pair_count"],
        "synthetic.tokens.estimated_prompt": result["token_metrics"].get("estimated_prompt_tokens", 0),
        "synthetic.tokens.estimated_completion": result["token_metrics"].get("estimated_completion_tokens", 0),
        "synthetic.tokens.estimated_total": result["token_metrics"].get("estimated_total_tokens", 0),
    })
    write_json(out_dir / "artifacts_manifest.json", {
        "schema_version": "observability.v1",
        "artifacts": artifacts,
    })


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing artifacts without --overwrite: " + ", ".join(existing))


def run(config: CollectConfig) -> dict[str, Any]:
    job_id = resolve_job_id(config)
    model_provider, model = resolve_model_info(config)
    result_dir = result_repo_dir(config, job_id)
    records = load_generated_records(config.results_jsonl, result_dir)
    if not records:
        raise FileNotFoundError("no Data Designer generated records found")

    raw_rows, accepted_rows, provenance_rows, rejection_counts = normalize_records(
        records,
        config=config,
        job_id=job_id,
        model_provider=model_provider,
        model=model,
    )
    data_designer_dir = config.experiment_dir / "data_designer"
    provenance_dir = config.experiment_dir / "provenance"
    generated_raw = data_designer_dir / "generated_raw.jsonl"
    generated_samples = data_designer_dir / "generated_samples.jsonl"
    accepted_samples = data_designer_dir / "accepted_samples.jsonl"
    omitted_seed_records = data_designer_dir / "omitted_seed_records.jsonl"
    provenance_samples = provenance_dir / "synthetic_samples.jsonl"
    stage_rows = config.experiment_dir / "stage1_5_data_designer_synthetic.jsonl"
    result_manifest_path = data_designer_dir / "result_manifest.json"
    ensure_writable(
        [generated_raw, generated_samples, accepted_samples, omitted_seed_records, provenance_samples, stage_rows, result_manifest_path],
        config.overwrite,
    )

    seed_rows = read_jsonl(data_designer_dir / "llm_seed_requests.jsonl")
    returned_seed_ids = {str(record.get("seed_id") or record.get("gap_id")) for record in records}
    omitted: list[dict[str, Any]] = []
    for idx, seed in enumerate(seed_rows):
        seed_id = str(seed.get("seed_id") or seed.get("gap_id") or "")
        if seed_id in returned_seed_ids:
            continue
        omitted.append({
            "schema_version": SCHEMA_VERSION,
            "data_designer_job_id": job_id,
            "model_provider": model_provider,
            "model": model,
            "reason": "provider_generation_failed_omitted_by_data_designer",
            "seed_id": seed_id,
            "seed_offset": idx,
            "source_dataset": seed.get("source_dataset"),
            "source_split": seed.get("source_split"),
            "source_row_index": seed.get("source_row_index"),
            "source_sample_id": seed.get("source_sample_id"),
            "source_url": seed.get("source_url"),
            "passage_id": seed.get("passage_id"),
        })

    write_jsonl(generated_raw, raw_rows)
    write_jsonl(generated_samples, accepted_rows)
    write_jsonl(accepted_samples, accepted_rows)
    write_jsonl(omitted_seed_records, omitted)
    write_jsonl(provenance_samples, provenance_rows)
    write_jsonl(stage_rows, raw_rows)

    analysis = read_analysis(result_dir)
    side_effect_columns = analysis.get("side_effect_column_names") or [
        "qa_pairs_json__reasoning_trace"
        if any("qa_pairs_json__reasoning_trace" in record for record in records)
        else None
    ]
    side_effect_columns = [column for column in side_effect_columns if column]
    raw_pair_count = len(raw_rows)
    accepted_pair_count = len(accepted_rows)
    token_metrics = token_metrics_from_analysis(analysis)
    download_manifest = read_json(data_designer_dir / "download_manifest.json")
    result_repo = {}
    if download_manifest:
        direct = download_manifest.get("direct_result_repo") if isinstance(download_manifest.get("direct_result_repo"), dict) else {}
        result_repo = {
            "repo_id": direct.get("repo_id"),
            "direct_clone_path": direct.get("path"),
            "artifact_url": (
                (download_manifest.get("result_list") or {}).get("data", [{}])[-1].get("artifact_url")
                if isinstance(download_manifest.get("result_list"), dict)
                and (download_manifest.get("result_list") or {}).get("data")
                else None
            ),
            "download_endpoint_status": (
                "failed_500_due_data_store_401_in_data_designer_helper"
                if download_manifest.get("download_endpoint_error")
                else "ok"
            ),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "collection": config.collection,
        "status": "collected",
        "data_designer_job_id": job_id,
        "model_provider": model_provider,
        "model": model,
        "seed_records_requested": len(seed_rows),
        "seed_records_returned": len(returned_seed_ids),
        "seed_records_omitted": len(omitted),
        "raw_synthetic_pair_count": raw_pair_count,
        "accepted_synthetic_pair_count": accepted_pair_count,
        "rejected_synthetic_pair_count": raw_pair_count - accepted_pair_count,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "source_support_check": "not_run",
        "curator_job_id": None,
        "think_hits_in_trainable_pairs_after_strip": sum(
            1
            for row in accepted_rows
            if "<think" in row["prompt"].lower() or "<think" in row["completion"].lower()
        ),
        "side_effect_columns_excluded": side_effect_columns,
        "token_metrics": token_metrics,
        "generated_raw_uri": "data_designer/generated_raw.jsonl",
        "generated_samples_uri": "data_designer/generated_samples.jsonl",
        "accepted_samples_uri": "data_designer/accepted_samples.jsonl",
        "omitted_seed_records_uri": "data_designer/omitted_seed_records.jsonl",
        "provenance_samples_uri": "provenance/synthetic_samples.jsonl",
        "stage_rows_uri": "stage1_5_data_designer_synthetic.jsonl",
        "result_repository": result_repo,
    }
    write_json(result_manifest_path, manifest)

    artifacts = [
        build_artifact(generated_raw, config.experiment_dir, "data_designer_generated_raw"),
        build_artifact(generated_samples, config.experiment_dir, "data_designer_generated_samples"),
        build_artifact(accepted_samples, config.experiment_dir, "data_designer_accepted_samples"),
        build_artifact(omitted_seed_records, config.experiment_dir, "data_designer_omitted_seed_records"),
        build_artifact(provenance_samples, config.experiment_dir, "synthetic_provenance_samples"),
        build_artifact(stage_rows, config.experiment_dir, "synthetic_stage_rows"),
        build_artifact(result_manifest_path, config.experiment_dir, "data_designer_result_manifest"),
    ]
    write_observability(config, manifest, artifacts)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dataset-dir", type=Path, default=DEFAULT_SOURCE_DATASET_DIR)
    ap.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    ap.add_argument("--collection", default=os.getenv("COLLECTION", "nim_curated"))
    ap.add_argument("--job-id", default=os.getenv("DATA_DESIGNER_JOB_ID"))
    ap.add_argument("--results-jsonl", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--model-provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> CollectConfig:
    return CollectConfig(
        source_dataset_dir=args.source_dataset_dir,
        experiment_dir=args.experiment_dir,
        collection=args.collection,
        job_id=args.job_id,
        results_jsonl=args.results_jsonl,
        results_dir=args.results_dir,
        model_provider=args.model_provider,
        model=args.model,
        system_prompt=args.system_prompt,
        overwrite=args.overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    result = run(config_from_args(parse_args(argv)))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
