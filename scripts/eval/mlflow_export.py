"""Small MLflow helpers for evaluation scripts."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://10.43.102.80:5000")
DEFAULT_MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "docs-to-data-to-lora-golden-eval")
DEFAULT_MLFLOW_ARTIFACT_LOCATION = os.getenv(
    "MLFLOW_ARTIFACT_LOCATION",
    "file://<MLFLOW_ARTIFACT_ROOT>/golden-eval",
)


def _coerce_metric(value: Any) -> float | None:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    return metric if math.isfinite(metric) else None


def _safe_param(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 500 else text[:497] + "..."


def ensure_experiment(
    *,
    mlflow: Any,
    experiment_name: str,
    artifact_location: str | None,
) -> str:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=artifact_location)
        return experiment_name
    if artifact_location and (experiment.artifact_location or "").startswith("file:///mlflow"):
        fallback = f"{experiment_name}-local-artifacts"
        if client.get_experiment_by_name(fallback) is None:
            client.create_experiment(fallback, artifact_location=artifact_location)
        return fallback
    return experiment_name


def log_eval_artifacts(
    *,
    enabled: bool,
    tracking_uri: str | None,
    experiment_name: str,
    artifact_location: str | None,
    run_name: str,
    artifact_dir: Path,
    tags: dict[str, Any],
    params: dict[str, Any],
    metrics: dict[str, Any],
) -> str | None:
    if not enabled:
        return None
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - operational dependency.
        raise RuntimeError("MLflow export requested but mlflow is not importable") from exc
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    resolved_experiment = ensure_experiment(
        mlflow=mlflow,
        experiment_name=experiment_name,
        artifact_location=artifact_location,
    )
    mlflow.set_experiment(resolved_experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({key: _safe_param(value) for key, value in tags.items() if value is not None})
        for key, value in params.items():
            if value is not None:
                mlflow.log_param(key, _safe_param(value))
        for key, value in metrics.items():
            metric = _coerce_metric(value)
            if metric is not None:
                mlflow.log_metric(key, metric)
        mlflow.log_artifacts(str(artifact_dir))
        return run.info.run_id


def write_repo_summary(
    *,
    repo_summary_dir: Path,
    eval_run_id: str,
    scope: str,
    artifact_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    repo_summary_dir = repo_summary_dir / eval_run_id
    repo_summary_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(artifact_dir)).strip("-")
    slug = slug[-180:] if len(slug) > 180 else slug
    out = repo_summary_dir / f"{scope}__{slug}.json"
    record = {
        "schema_version": "golden-eval-repo-summary/v1",
        "eval_run_id": eval_run_id,
        "scope": scope,
        "artifact_dir": str(artifact_dir),
        "manifest_path": str(artifact_dir / "manifest.json"),
        "summary_path": str(artifact_dir / "summary.json"),
        "mlflow": manifest.get("mlflow"),
        "rubric": manifest.get("rubric"),
        "judge": manifest.get("judge"),
        "summary": summary,
    }
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
