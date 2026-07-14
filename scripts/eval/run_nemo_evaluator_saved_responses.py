"""Optionally score saved answer sets with NeMo Evaluator live RAGAS metrics.

This is a later diagnostic path, not the formal no-RAG winner evaluation. RAGAS
uses context-oriented metrics and may require a judge embedding endpoint. The
runner is batch-oriented because /v1/evaluation/live is synchronous and judge
endpoint concurrency limits can be low.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.evaluator_client import EvaluatorClient  # noqa: E402
from scripts.eval.run_live_ragas_smoke import (  # noqa: E402
    REASONING_GENERATION_BUDGET,
    _RAGAS_ROWS_INPUT_TEMPLATE,
    build_judge_model,
    extract_judge_token_counts,
    redact_secrets,
    split_context_baked_prompt,
)

log = logging.getLogger(__name__)

DEFAULT_EVALUATOR_URL = os.getenv("EVALUATOR_URL", "http://10.43.143.110:7331")
DEFAULT_JUDGE_API_URL = os.getenv("JUDGE_API_URL", "http://llm-judge.default.svc.cluster.local:8000/v1")
DEFAULT_JUDGE_MODEL_ID = os.getenv("EVALUATOR_JUDGE_MODEL", "azure/moonshotai/kimi-k2.6")
DEFAULT_JUDGE_API_KEY_ENV = os.getenv("JUDGE_API_KEY_ENV", "LLM_API_KEY")
DEFAULT_JUDGE_EMBEDDING_API_URL = os.getenv(
    "JUDGE_EMBEDDING_API_URL",
    "http://10.43.101.173:8000/v1/embeddings",
)
DEFAULT_JUDGE_EMBEDDING_MODEL_ID = os.getenv(
    "JUDGE_EMBEDDING_MODEL_ID",
    "nvidia/llama-3.2-nv-embedqa-1b-v2",
)
DEFAULT_COMPLETIONS_ROOT = Path(
    os.getenv(
        "GOLDEN_COMPLETIONS_ROOT",
        "<EVAL_ROOT>/completions-question-only",
    )
)
DEFAULT_GOLDEN_ROOT = _REPO_ROOT / "curator_dataset" / "experiments" / "20260709-curator-vs-le" / "golden_eval" / "golden-v1"
DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("NEMO_EVALUATOR_KIMI_OUTPUT_ROOT", "<EVAL_ROOT>/nemo-evaluator-kimi")
)
DEFAULT_REPO_SUMMARY_DIR = _REPO_ROOT / "curator_dataset" / "experiments" / "20260709-curator-vs-le" / "golden_eval" / "evaluator_kimi_20260712"
DEFAULT_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://10.43.102.80:5000")
DEFAULT_MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "docs-to-data-to-lora-golden-eval")
DEFAULT_MLFLOW_ARTIFACT_LOCATION = os.getenv(
    "MLFLOW_ARTIFACT_LOCATION",
    "file://<MLFLOW_ARTIFACT_ROOT>/golden-eval",
)
DEFAULT_METRICS = ["faithfulness", "response_relevancy", "answer_accuracy"]

_SECRET_KEYS = {"api_key", "authorization", "token", "access_token"}


@dataclass(frozen=True)
class ResponseFile:
    path: Path
    dataset_slug: str
    base_slug: str
    target_slug: str
    rank_slug: str
    answer_run_id: str
    corpus_slug: str

    @property
    def safe_id(self) -> str:
        return "__".join(
            [
                self.dataset_slug,
                self.base_slug,
                self.target_slug,
                self.rank_slug,
                self.answer_run_id,
            ]
        ).replace("/", "_")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_to_corpus(dataset_slug: str) -> str:
    if dataset_slug.startswith("nim_curated"):
        return "nim_curated"
    if dataset_slug.startswith("nemo_usvcs_curated"):
        return "nemo_usvcs_curated"
    raise ValueError(f"Cannot derive corpus from dataset slug: {dataset_slug}")


def discover_response_files(patterns: list[str], completions_root: Path) -> list[ResponseFile]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [str(candidate)]
        paths.extend(Path(match).resolve() for match in matches)
    unique = sorted(set(paths))
    out: list[ResponseFile] = []
    for path in unique:
        try:
            rel = path.relative_to(completions_root)
            parts = rel.parts
            dataset_slug, base_slug, target_slug, rank_slug, answer_run_id, filename = parts[:6]
        except Exception as exc:
            raise ValueError(
                f"Response path must be under {completions_root} with "
                "<dataset>/<base>/<target>/<rank>/<run>/responses.jsonl shape: {path}"
            ) from exc
        if filename != "responses.jsonl":
            raise ValueError(f"Expected responses.jsonl, got {path}")
        out.append(
            ResponseFile(
                path=path,
                dataset_slug=dataset_slug,
                base_slug=base_slug,
                target_slug=target_slug,
                rank_slug=rank_slug,
                answer_run_id=answer_run_id,
                corpus_slug=dataset_to_corpus(dataset_slug),
            )
        )
    return out


def load_context_rows(golden_root: Path, corpus_slug: str) -> dict[str, dict[str, Any]]:
    path = golden_root / corpus_slug / "test_set_with_context.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        golden_id = row.get("golden_id")
        if golden_id:
            rows[golden_id] = row
        source_index = (row.get("golden") or {}).get("source_row_index")
        if source_index is not None:
            rows[f"source:{source_index}"] = row
    return rows


def response_golden_id(row: dict[str, Any]) -> str | None:
    meta = row.get("row_metadata") or {}
    return meta.get("golden_id") or row.get("golden_id")


def response_golden_source_index(row: dict[str, Any]) -> Any:
    return ((row.get("row_metadata") or {}).get("golden") or {}).get("source_row_index")


def coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_evaluator_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_+.@:" else "-" for ch in value)


def _embedding_url(api_url: str) -> str:
    base = api_url.rstrip("/")
    if base.endswith("/embeddings"):
        return base
    if base.endswith("/v1"):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


def build_embedding_model(*, embedding_api_url: str, model_id: str, model_name: str | None = None) -> dict[str, Any]:
    return {
        "name": model_name or safe_evaluator_name(model_id),
        "namespace": "default",
        "api_endpoint": {
            "url": _embedding_url(embedding_api_url),
            "model_id": model_id,
            "format": "openai",
        },
    }


def build_eval_row(response_row: dict[str, Any], context_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    golden_id = response_golden_id(response_row)
    source_index = response_golden_source_index(response_row)
    context_row = context_rows.get(f"source:{source_index}") if source_index is not None else None
    if not context_row:
        context_row = context_rows.get(golden_id or "")
    if not context_row:
        raise KeyError(
            f"No context-baked row found for golden_source_row_index={source_index!r} "
            f"golden_id={golden_id!r}"
        )
    prompt_parts = split_context_baked_prompt(context_row["prompt"])
    model = response_row.get("model") or {}
    token_counts = response_row.get("token_counts") or {}
    return {
        "question": response_row["prompt"],
        "context": prompt_parts["context"],
        "completion": response_row.get("reference_completion") or context_row.get("completion"),
        "response": response_row.get("response") or "",
        "prompt": response_row["prompt"],
        "system": response_row.get("system"),
        "golden_id": golden_id,
        "golden_source_row_index": source_index,
        "response_source_row_index": response_row.get("source_row_index"),
        "row_ordinal": response_row.get("row_ordinal"),
        "dataset_slug": response_row.get("dataset_slug"),
        "response_model_id": model.get("model_id"),
        "response_base_slug": model.get("base_slug"),
        "response_target_slug": model.get("target_slug"),
        "response_rank_slug": model.get("rank_slug"),
        "response_target_type": model.get("target_type"),
        "target_token_counts": {
            "prompt_tokens": coerce_int(token_counts.get("prompt_tokens")),
            "completion_tokens_raw": coerce_int(token_counts.get("completion_tokens_raw")),
            "completion_tokens_cleaned_est": coerce_int(token_counts.get("completion_tokens_cleaned_est")),
            "think_tokens_est": coerce_int(token_counts.get("think_tokens_est")),
            "total_tokens_raw": coerce_int(token_counts.get("total_tokens_raw")),
        },
    }


def build_live_payload(
    *,
    rows: list[dict[str, Any]],
    judge_model: dict[str, Any],
    metric_types: list[str],
    parallelism: int,
    embedding_model: dict[str, Any] | None,
    description: str,
) -> dict[str, Any]:
    metrics = {
        metric_type: {
            "type": metric_type,
            "params": {
                "judge": {"model": judge_model},
                **({"judge_embeddings": {"model": embedding_model}} if embedding_model else {}),
                "input_template": _RAGAS_ROWS_INPUT_TEMPLATE,
            },
        }
        for metric_type in metric_types
    }
    return {
        "namespace": "default",
        "description": description,
        "target": {"type": "rows", "rows": rows},
        "config": {
            "type": "custom",
            "params": {
                "parallelism": parallelism,
                "limit_samples": len(rows),
                "temperature": 0.0001,
                "max_tokens": REASONING_GENERATION_BUDGET,
            },
            "tasks": {
                "ragas_saved_response_rubric": {
                    "type": "data",
                    "metrics": metrics,
                },
            },
        },
    }


def metric_summaries(result: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    tasks = ((result.get("result") or {}).get("tasks") or {})
    for task in tasks.values():
        for metric_name, metric_payload in (task.get("metrics") or {}).items():
            for score_name, score_payload in (metric_payload.get("scores") or {}).items():
                value = score_payload.get("value")
                stats = score_payload.get("stats") or {}
                key = score_name or metric_name
                out[key] = {
                    "value": value,
                    "mean": stats.get("mean", value),
                    "count": stats.get("count"),
                    "sum": stats.get("sum"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                }
    return out


def token_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = [
        "prompt_tokens",
        "completion_tokens_raw",
        "completion_tokens_cleaned_est",
        "think_tokens_est",
        "total_tokens_raw",
    ]
    totals = {key: 0 for key in keys}
    for row in rows:
        counts = row.get("target_token_counts") or {}
        for key in keys:
            totals[key] += coerce_int(counts.get(key))
    return totals


def summarize_result(rows: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    status_details = result.get("status_details") or {}
    judge_counts = extract_judge_token_counts(result)
    judge_totals = token_totals([{"target_token_counts": item} for item in judge_counts])
    return {
        "schema_version": "nemo-evaluator-saved-response-batch-summary/v1",
        "status": result.get("status"),
        "message": status_details.get("message"),
        "samples_processed": status_details.get("samples_processed"),
        "progress": status_details.get("progress"),
        "row_count": len(rows),
        "evaluator_result_id": ((result.get("result") or {}).get("id")),
        "evaluator_job_id": ((result.get("result") or {}).get("job")),
        "metrics": metric_summaries(result),
        "target_generation_tokens": token_totals(rows),
        "judge_scoring_tokens": {
            **judge_totals,
            "request_count_with_usage": len(judge_counts),
        },
    }


def aggregate_batch_summaries(batch_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, list[tuple[float, int]]] = {}
    for summary in batch_summaries:
        for metric_name, metric in (summary.get("metrics") or {}).items():
            metric_mean = metric.get("mean")
            if metric_mean is None:
                metric_mean = metric.get("value")
            if metric_mean is None:
                continue
            count = coerce_int(metric.get("count")) or coerce_int(summary.get("samples_processed")) or coerce_int(summary.get("row_count"))
            metrics.setdefault(metric_name, []).append((float(metric_mean), max(count, 1)))
    metric_out = {}
    for metric_name, values in metrics.items():
        weighted_sum = sum(value * count for value, count in values)
        total_count = sum(count for _, count in values)
        metric_out[metric_name] = {
            "mean": weighted_sum / total_count if total_count else mean(value for value, _ in values),
            "count": total_count,
        }
    return {
        "schema_version": "nemo-evaluator-saved-response-target-summary/v1",
        "completed_batches": sum(1 for item in batch_summaries if item.get("status") == "completed"),
        "failed_batches": sum(1 for item in batch_summaries if item.get("status") != "completed"),
        "row_count": sum(coerce_int(item.get("row_count")) for item in batch_summaries),
        "samples_processed": sum(coerce_int(item.get("samples_processed")) for item in batch_summaries),
        "metrics": metric_out,
        "status": "completed" if all(item.get("status") == "completed" for item in batch_summaries) else "partial",
    }


def mlflow_log_batch(
    *,
    enabled: bool,
    tracking_uri: str | None,
    experiment_name: str,
    artifact_location: str | None,
    run_name: str,
    tags: dict[str, str],
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_dir: Path,
) -> str | None:
    if not enabled:
        return None
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - operational fallback
        raise RuntimeError("MLflow export requested but mlflow is not importable") from exc
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    elif artifact_location and (experiment.artifact_location or "").startswith("file:///mlflow"):
        experiment_name = f"{experiment_name}-local-artifacts"
        if client.get_experiment_by_name(experiment_name) is None:
            client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        for key, value in params.items():
            if value is not None:
                mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_artifacts(str(artifact_dir))
        return run.info.run_id


def batch_rows(rows: list[dict[str, Any]], batch_size: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    if batch_size <= 0:
        yield 0, rows
        return
    for start in range(0, len(rows), batch_size):
        yield start // batch_size, rows[start : start + batch_size]


def run_one_response_file(
    spec: ResponseFile,
    *,
    args: argparse.Namespace,
    judge_model: dict[str, Any],
) -> dict[str, Any]:
    context_rows = load_context_rows(args.golden_root, spec.corpus_slug)
    eval_rows: list[dict[str, Any]] = []
    missing_context = 0
    for response_row in iter_jsonl(spec.path):
        try:
            eval_rows.append(build_eval_row(response_row, context_rows))
        except KeyError:
            missing_context += 1
        if args.limit and len(eval_rows) >= args.limit:
            break
    if not eval_rows:
        raise RuntimeError(f"No evaluable rows loaded from {spec.path}; missing_context={missing_context}")

    target_dir = args.output_root / "singleaxis" / spec.dataset_slug / spec.base_slug / spec.target_slug / spec.rank_slug / spec.answer_run_id / args.eval_run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        target_dir / "manifest.json",
        {
            "schema_version": "nemo-evaluator-saved-response-run/v1",
            "created_at": now_utc(),
            "eval_run_id": args.eval_run_id,
            "responses_path": str(spec.path),
            "dataset_slug": spec.dataset_slug,
            "corpus_slug": spec.corpus_slug,
            "base_slug": spec.base_slug,
            "target_slug": spec.target_slug,
            "rank_slug": spec.rank_slug,
            "answer_run_id": spec.answer_run_id,
            "row_count": len(eval_rows),
            "missing_context_rows": missing_context,
            "metrics": args.metrics,
            "evaluator_url": args.evaluator_url,
            "judge": {
                "endpoint": args.judge_api_url,
                "model": args.judge_model_id,
                "endpoint_format": args.judge_endpoint_format,
                "max_tokens": args.judge_max_tokens,
                "timeout_s": args.judge_timeout_s,
                "max_retries": args.judge_max_retries,
            },
            "judge_embeddings": {
                "enabled": args.use_judge_embeddings,
                "endpoint": args.judge_embedding_api_url if args.use_judge_embeddings else None,
                "model": args.judge_embedding_model_id if args.use_judge_embeddings else None,
            },
            "mlflow": {
                "enabled": not args.no_mlflow,
                "tracking_uri": args.mlflow_tracking_uri,
                "experiment": args.mlflow_experiment,
                "artifact_location": args.mlflow_artifact_location,
            },
        },
    )

    batch_summaries: list[dict[str, Any]] = []
    for batch_index, rows in batch_rows(eval_rows, args.batch_size):
        batch_dir = target_dir / f"batch-{batch_index:05d}"
        summary_path = batch_dir / "summary.json"
        if args.resume and summary_path.exists():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed":
                log.info("skip completed batch %s", batch_dir)
                batch_summaries.append(existing)
                continue

        batch_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(batch_dir / "rows_to_judge.jsonl", rows)
        payload = build_live_payload(
            rows=rows,
            judge_model=judge_model,
            metric_types=args.metrics,
            parallelism=args.parallelism,
            embedding_model=args.embedding_model,
            description=f"golden-v1 saved-response single-axis {spec.safe_id} batch {batch_index}",
        )
        write_json(batch_dir / "payload_redacted.json", redact_secrets(payload))
        with EvaluatorClient(
            args.evaluator_url,
            api_key=args.evaluator_api_key,
            timeout=args.judge_timeout_s + 240,
        ) as client:
            result = client.submit_live(payload)
        safe_result = redact_secrets(result)
        write_json(batch_dir / "result.json", safe_result)
        summary = summarize_result(rows, safe_result)
        write_json(summary_path, summary)

        metric_values = {
            f"score.{name}.mean": float(payload["mean"])
            for name, payload in (summary.get("metrics") or {}).items()
            if payload.get("mean") is not None
        }
        metric_values.update(
            {
                "rows.count": float(summary["row_count"]),
                "rows.samples_processed": float(coerce_int(summary.get("samples_processed"))),
                "judge.requests_with_usage": float(coerce_int((summary.get("judge_scoring_tokens") or {}).get("request_count_with_usage"))),
            }
        )
        mlflow_run_id = mlflow_log_batch(
            enabled=not args.no_mlflow,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
            artifact_location=args.mlflow_artifact_location,
            run_name=f"{args.eval_run_id}/{spec.base_slug}/{spec.target_slug}/{spec.rank_slug}/batch-{batch_index:05d}",
            tags={
                "pipeline": "docs-to-data-to-lora",
                "pipeline.stage": "golden-evaluator",
                "eval.engine": "nemo-evaluator",
                "eval.scope": "singleaxis",
                "eval.no_rag": "true",
                "eval.judge.model": args.judge_model_id,
                "eval.dataset_slug": spec.dataset_slug,
                "eval.base_slug": spec.base_slug,
                "eval.target_slug": spec.target_slug,
                "eval.rank_slug": spec.rank_slug,
                "eval.answer_run_id": spec.answer_run_id,
                "eval.run_id": args.eval_run_id,
            },
            params={
                "responses_path": str(spec.path),
                "evaluator_url": args.evaluator_url,
                "judge_api_url": args.judge_api_url,
                "metrics": ",".join(args.metrics),
                "parallelism": args.parallelism,
                "batch_size": args.batch_size,
                "batch_index": batch_index,
                "row_count": summary["row_count"],
                "evaluator_result_id": summary.get("evaluator_result_id"),
                "evaluator_job_id": summary.get("evaluator_job_id"),
            },
            metrics=metric_values,
            artifact_dir=batch_dir,
        )
        if mlflow_run_id:
            summary["mlflow_run_id"] = mlflow_run_id
            write_json(summary_path, summary)
        batch_summaries.append(summary)

        if result.get("status") != "completed":
            log.warning("batch %s ended with status=%s; stopping target", batch_dir, result.get("status"))
            break

    target_summary = aggregate_batch_summaries(batch_summaries)
    target_summary.update(
        {
            "completed_at": now_utc(),
            "eval_run_id": args.eval_run_id,
            "responses_path": str(spec.path),
            "dataset_slug": spec.dataset_slug,
            "base_slug": spec.base_slug,
            "target_slug": spec.target_slug,
            "rank_slug": spec.rank_slug,
            "answer_run_id": spec.answer_run_id,
            "target_output_dir": str(target_dir),
        }
    )
    write_json(target_dir / "summary.json", target_summary)
    return target_summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--responses",
        action="append",
        help="Path or glob for responses.jsonl. May be repeated.",
    )
    ap.add_argument("--all-responses", action="store_true", help="Evaluate every responses.jsonl under --completions-root.")
    ap.add_argument("--completions-root", type=Path, default=DEFAULT_COMPLETIONS_ROOT)
    ap.add_argument("--golden-root", type=Path, default=DEFAULT_GOLDEN_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--repo-summary-dir", type=Path, default=DEFAULT_REPO_SUMMARY_DIR)
    ap.add_argument("--eval-run-id", default=datetime.now(timezone.utc).strftime("golden-v1-nemo-evaluator-kimi-%Y%m%dT%H%M%SZ"))
    ap.add_argument("--evaluator-url", default=DEFAULT_EVALUATOR_URL)
    ap.add_argument("--evaluator-api-key", default=os.getenv("EVALUATOR_API_KEY"))
    ap.add_argument("--judge-api-url", default=DEFAULT_JUDGE_API_URL)
    ap.add_argument("--judge-model-id", default=DEFAULT_JUDGE_MODEL_ID)
    ap.add_argument("--judge-model-name", default=None, help="Evaluator-safe judge model entity name. Defaults to sanitized --judge-model-id.")
    ap.add_argument("--judge-api-key-env", default=DEFAULT_JUDGE_API_KEY_ENV)
    ap.add_argument("--judge-api-key", default=None)
    ap.add_argument("--judge-endpoint-format", default="openai")
    ap.add_argument("--judge-max-tokens", type=int, default=REASONING_GENERATION_BUDGET)
    ap.add_argument("--judge-timeout-s", type=int, default=600)
    ap.add_argument("--judge-max-retries", type=int, default=3)
    ap.add_argument("--judge-embedding-api-url", default=DEFAULT_JUDGE_EMBEDDING_API_URL)
    ap.add_argument("--judge-embedding-model-id", default=DEFAULT_JUDGE_EMBEDDING_MODEL_ID)
    ap.add_argument("--judge-embedding-model-name", default=None)
    ap.add_argument("--no-judge-embeddings", action="store_true", help="Do not attach judge_embeddings even when embedding-backed metrics are requested.")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--limit", type=int, default=0, help="Limit rows per response file; 0 means all.")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--parallelism", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    ap.add_argument("--mlflow-experiment", default=DEFAULT_MLFLOW_EXPERIMENT)
    ap.add_argument("--mlflow-artifact-location", default=DEFAULT_MLFLOW_ARTIFACT_LOCATION)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    patterns = list(args.responses or [])
    if args.all_responses:
        patterns.append(str(args.completions_root / "**" / "responses.jsonl"))
    if not patterns:
        raise ValueError("Pass --responses or --all-responses")

    judge_api_key = args.judge_api_key or os.getenv(args.judge_api_key_env)
    if args.judge_api_url.startswith("https://") and not judge_api_key:
        raise ValueError(f"set ${args.judge_api_key_env} or pass --judge-api-key")

    judge_model = build_judge_model(
        judge_api_url=args.judge_api_url,
        model_id=args.judge_model_id,
        max_tokens=args.judge_max_tokens,
        request_timeout_s=args.judge_timeout_s,
        max_retries=args.judge_max_retries,
        api_key=judge_api_key,
        endpoint_format=args.judge_endpoint_format,
    )
    judge_model["name"] = args.judge_model_name or safe_evaluator_name(args.judge_model_id)
    embedding_backed_metrics = {"response_relevancy", "answer_similarity", "semantic_similarity"}
    args.use_judge_embeddings = bool(set(args.metrics) & embedding_backed_metrics) and not args.no_judge_embeddings
    args.embedding_model = (
        build_embedding_model(
            embedding_api_url=args.judge_embedding_api_url,
            model_id=args.judge_embedding_model_id,
            model_name=args.judge_embedding_model_name,
        )
        if args.use_judge_embeddings
        else None
    )

    response_files = discover_response_files(patterns, args.completions_root.resolve())
    log.info("discovered %d response files", len(response_files))
    run_summaries: list[dict[str, Any]] = []
    for spec in response_files:
        log.info("evaluating %s", spec.path)
        run_summaries.append(run_one_response_file(spec, args=args, judge_model=judge_model))

    repo_summary = {
        "schema_version": "nemo-evaluator-kimi-run-summary/v1",
        "created_at": now_utc(),
        "eval_run_id": args.eval_run_id,
        "mode": "singleaxis",
        "engine": "nemo-evaluator",
        "judge": {
            "endpoint": args.judge_api_url,
            "model": args.judge_model_id,
        },
        "judge_embeddings": {
            "enabled": args.use_judge_embeddings,
            "endpoint": args.judge_embedding_api_url if args.use_judge_embeddings else None,
            "model": args.judge_embedding_model_id if args.use_judge_embeddings else None,
        },
        "mlflow": {
            "enabled": not args.no_mlflow,
            "tracking_uri": args.mlflow_tracking_uri,
            "experiment": args.mlflow_experiment,
            "artifact_location": args.mlflow_artifact_location,
        },
        "response_file_count": len(response_files),
        "metrics": args.metrics,
        "parallelism": args.parallelism,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "results": run_summaries,
    }
    args.repo_summary_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.repo_summary_dir / f"{args.eval_run_id}_summary.json", repo_summary)
    return 0 if all(item.get("status") == "completed" for item in run_summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
