"""Tests for MLflow evaluation export helpers."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from scripts.eval.mlflow_export import ensure_experiment, log_eval_artifacts, write_repo_summary


class _FakeClient:
    def __init__(self) -> None:
        self.experiments: dict[str, SimpleNamespace] = {}
        self.created: list[tuple[str, str | None]] = []

    def get_experiment_by_name(self, name: str):
        return self.experiments.get(name)

    def create_experiment(self, name: str, artifact_location: str | None = None):
        self.created.append((name, artifact_location))
        self.experiments[name] = SimpleNamespace(name=name, artifact_location=artifact_location)
        return name


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMlflow:
    def __init__(self) -> None:
        self.client = _FakeClient()
        self.tracking = SimpleNamespace(MlflowClient=lambda: self.client)
        self.tracking_uri = None
        self.experiment = None
        self.run_names: list[str] = []
        self.tags: dict[str, str] = {}
        self.params: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.artifact_paths: list[str] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment = experiment_name

    def start_run(self, run_name: str):
        self.run_names.append(run_name)
        return _FakeRun("run-123")

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_param(self, key: str, value: str) -> None:
        self.params[key] = value

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = value

    def log_artifacts(self, path: str) -> None:
        self.artifact_paths.append(path)


def test_ensure_experiment_creates_missing_experiment():
    fake = _FakeMlflow()

    resolved = ensure_experiment(
        mlflow=fake,
        experiment_name="golden-eval",
        artifact_location="file:///tmp/mlflow-artifacts",
    )

    assert resolved == "golden-eval"
    assert fake.client.created == [("golden-eval", "file:///tmp/mlflow-artifacts")]


def test_ensure_experiment_uses_local_artifact_fallback_for_container_default():
    fake = _FakeMlflow()
    fake.client.experiments["golden-eval"] = SimpleNamespace(
        name="golden-eval",
        artifact_location="file:///mlflow/artifacts",
    )

    resolved = ensure_experiment(
        mlflow=fake,
        experiment_name="golden-eval",
        artifact_location="file:///tmp/mlflow-artifacts",
    )

    assert resolved == "golden-eval-local-artifacts"
    assert fake.client.created == [
        ("golden-eval-local-artifacts", "file:///tmp/mlflow-artifacts")
    ]


def test_log_eval_artifacts_records_tags_params_metrics_and_artifacts(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "summary.json").write_text("{}", encoding="utf-8")
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    run_id = log_eval_artifacts(
        enabled=True,
        tracking_uri="http://mlflow:5000",
        experiment_name="golden-eval",
        artifact_location="file:///tmp/mlflow-artifacts",
        run_name="single-axis",
        artifact_dir=artifact_dir,
        tags={"pipeline": "docs-to-data-to-lora", "skip": None},
        params={"model": "llama", "long": "x" * 600, "skip": None},
        metrics={"accuracy": "4.5", "bad": "not-a-number", "nan": float("nan")},
    )

    assert run_id == "run-123"
    assert fake.tracking_uri == "http://mlflow:5000"
    assert fake.experiment == "golden-eval"
    assert fake.run_names == ["single-axis"]
    assert fake.tags == {"pipeline": "docs-to-data-to-lora"}
    assert fake.params["model"] == "llama"
    assert fake.params["long"].endswith("...")
    assert len(fake.params["long"]) == 500
    assert fake.metrics == {"accuracy": 4.5}
    assert fake.artifact_paths == [str(artifact_dir)]


def test_log_eval_artifacts_disabled_does_not_require_mlflow(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)

    assert log_eval_artifacts(
        enabled=False,
        tracking_uri=None,
        experiment_name="golden-eval",
        artifact_location=None,
        run_name="disabled",
        artifact_dir=tmp_path,
        tags={},
        params={},
        metrics={},
    ) is None


def test_write_repo_summary_captures_manifest_summary_and_mlflow_ref(tmp_path):
    artifact_dir = tmp_path / "eval" / "run"
    artifact_dir.mkdir(parents=True)
    manifest = {
        "mlflow": {"run_id": "run-123"},
        "rubric": {"scale": "1-5"},
        "judge": {"model": "judge-model"},
    }
    summary = {"rows_scored": 10}

    out = write_repo_summary(
        repo_summary_dir=tmp_path / "repo-summaries",
        eval_run_id="eval-001",
        scope="singleaxis",
        artifact_dir=artifact_dir,
        manifest=manifest,
        summary=summary,
    )

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["schema_version"] == "golden-eval-repo-summary/v1"
    assert record["eval_run_id"] == "eval-001"
    assert record["scope"] == "singleaxis"
    assert record["mlflow"] == {"run_id": "run-123"}
    assert record["summary"] == {"rows_scored": 10}
