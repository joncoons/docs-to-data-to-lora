#!/usr/bin/env python3
"""Submit and collect the grounded-dataset Data Designer augmentation experiment.

This complements ``build_data_designer_seed_from_grounded.py``. It uploads the
LLM-authored seed CSV to NeMo Data Store, submits a native NeMo Data Designer
job, and can poll/download results for later admission and merge steps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.build_data_designer_seed_from_grounded import (  # noqa: E402
    DATA_DESIGNER_PROMPT_TEMPLATE,
    count_jsonl_rows,
    sha256_file,
)
from scripts.pipeline.provenance import SCHEMA_VERSION, utc_now  # noqa: E402

DEFAULT_EXPERIMENT_DIR = Path(
    os.getenv(
        "DATA_DESIGNER_AUGMENTATION_DIR",
        "<DATASET_ROOT>/experiments/nim_curated_dd_llm_1b",
    )
)
DEFAULT_DATA_DESIGNER_URL = os.getenv("DATA_DESIGNER_URL", "http://192.168.1.187:30812")
DEFAULT_DATA_STORE_URL = os.getenv("DATA_STORE_URL", "http://192.168.1.187:30912")
DEFAULT_DATA_STORE_GIT_BASE = os.getenv("DATA_STORE_GIT_BASE", DEFAULT_DATA_STORE_URL)
DEFAULT_NAMESPACE = os.getenv("DATASET_NAMESPACE", "default")
DEFAULT_SEED_REPO_NAME = os.getenv("DATA_DESIGNER_SEED_REPO_NAME", "stage3-nim-curated-dd-llm-seeds")
DEFAULT_MODEL_PROVIDER = os.getenv("DATA_DESIGNER_MODEL_PROVIDER", "frontier-llm-provider")
DEFAULT_MODEL = os.getenv("DATA_DESIGNER_MODEL", "frontier-llm-model")
DEFAULT_MODEL_ALIAS = os.getenv("DATA_DESIGNER_MODEL_ALIAS", "generation_model")

SYSTEM_PROMPT = (
    "You generate source-grounded synthetic training data for NVIDIA technical "
    "documentation. Return JSON only. Do not include reasoning, markdown, or "
    "<think> tags. Every answer must be supported by the provided source text."
)


QA_PAIRS_OUTPUT_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pairs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
            },
        }
    },
    "required": ["pairs"],
}


SEED_DATASET_COLUMNS = (
    "seed_id",
    "gap_id",
    "collection",
    "source_dataset",
    "source_split",
    "source_row_index",
    "source_sample_id",
    "source_url",
    "passage_id",
    "product_family",
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
)


@dataclass(frozen=True)
class ServiceConfig:
    namespace: str
    data_designer_url: str
    data_store_url: str
    data_store_git_base: str
    data_store_user: str | None
    data_store_password: str | None
    git_user_email: str = "data-designer-augmentation@nemo-peft.local"
    git_user_name: str = "data-designer-augmentation"

    @property
    def data_store_auth(self) -> tuple[str, str] | None:
        if not self.data_store_user:
            return None
        return (self.data_store_user, self.data_store_password or "")

    @property
    def authenticated_git_base(self) -> str:
        return inject_basic_auth(
            self.data_store_git_base,
            self.data_store_user,
            self.data_store_password,
        )


def inject_basic_auth(base_url: str, user: str | None, password: str | None) -> str:
    base_url = base_url.rstrip("/")
    if not user:
        return base_url
    parts = urlsplit(base_url)
    if parts.username:
        return base_url
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"Data Store Git base must be an absolute URL, got: {base_url!r}")
    auth = quote(user, safe="")
    if password is not None:
        auth = f"{auth}:{quote(password, safe='')}"
    return urlunsplit((parts.scheme, f"{auth}@{parts.netloc}", parts.path.rstrip("/"), parts.query, parts.fragment))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def job_artifact_dir(experiment_dir: Path, job_id: str) -> Path:
    return experiment_dir / "data_designer" / "jobs" / job_id


def result_repo_id(namespace: str, job_id: str) -> str:
    return f"{namespace}/job-results-{job_id}"


def collect_file_artifacts(root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if not root.exists():
        return artifacts
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        artifacts.append({
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return artifacts


def clone_result_repo(config: ServiceConfig, job_id: str, experiment_dir: Path) -> dict[str, Any]:
    repo_id = result_repo_id(config.namespace, job_id)
    clone_url = f"{config.authenticated_git_base.rstrip('/')}/{repo_id}.git"
    repo_dir = experiment_dir / "data_designer" / "results" / job_id / "repo"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    git_status = "cloned"
    if (repo_dir / ".git").exists():
        pull = run_cmd(["git", "pull", "--ff-only", "origin", "main"], cwd=repo_dir, allow_fail=True)
        if pull.returncode != 0:
            sys.stderr.write(pull.stderr.decode())
            raise subprocess.CalledProcessError(pull.returncode, pull.args)
        git_status = "updated"
    elif repo_dir.exists() and any(repo_dir.iterdir()):
        raise RuntimeError(f"result repo directory already exists and is not a git checkout: {repo_dir}")
    else:
        run_cmd(["git", "clone", clone_url, str(repo_dir)])
    commit_sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir).stdout.decode().strip()
    return {
        "repo_id": repo_id,
        "path": str(repo_dir),
        "git_status": git_status,
        "commit_sha": commit_sha,
        "artifacts": collect_file_artifacts(repo_dir),
    }


def run_cmd(args: list[str], cwd: Path | None = None, allow_fail: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=cwd, capture_output=True)
    if result.returncode != 0 and not allow_fail:
        sys.stderr.write(result.stderr.decode())
        raise subprocess.CalledProcessError(result.returncode, args)
    return result


def create_dataset_repo(repo_name: str, config: ServiceConfig) -> str:
    response = httpx.post(
        f"{config.data_store_url.rstrip('/')}/v1/hf/api/repos/create",
        auth=config.data_store_auth,
        json={
            "type": "dataset",
            "name": repo_name,
            "organization": config.namespace,
            "private": False,
        },
        timeout=30,
    )
    if response.status_code in (200, 201):
        return "created"
    if response.status_code == 409 or "already created" in response.text:
        return "exists"
    raise RuntimeError(f"create seed repo failed: {response.status_code} {response.text}")


def push_seed_repo(
    experiment_dir: Path,
    repo_name: str,
    config: ServiceConfig,
) -> dict[str, Any]:
    repo_id = f"{config.namespace}/{repo_name}"
    clone_url = f"{config.authenticated_git_base.rstrip('/')}/{repo_id}.git"
    seed_csv = experiment_dir / "data_designer" / "seed_dataset.csv"
    seed_requests = experiment_dir / "data_designer" / "llm_seed_requests.jsonl"
    submission_plan = experiment_dir / "data_designer" / "submission_plan.json"
    if not seed_csv.exists():
        raise FileNotFoundError(seed_csv)
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        run_cmd(["git", "clone", clone_url, str(repo)])
        for src, dst_rel in (
            (seed_csv, "seed_dataset.csv"),
            (seed_requests, "llm_seed_requests.jsonl"),
            (submission_plan, "submission_plan.json"),
        ):
            if src.exists():
                shutil.copyfile(src, repo / dst_rel)
        run_cmd(["git", "config", "user.email", config.git_user_email], cwd=repo)
        run_cmd(["git", "config", "user.name", config.git_user_name], cwd=repo)
        run_cmd(["git", "add", "seed_dataset.csv", "llm_seed_requests.jsonl", "submission_plan.json"], cwd=repo)
        with seed_csv.open(newline="", encoding="utf-8") as fh:
            rows = sum(1 for _ in csv.DictReader(fh))
        commit = run_cmd(
            ["git", "commit", "-m", f"upload grounded Data Designer seeds: {rows} records"],
            cwd=repo,
            allow_fail=True,
        )
        git_status = "pushed"
        if commit.returncode != 0:
            if b"nothing to commit" in commit.stdout or b"nothing to commit" in commit.stderr:
                git_status = "unchanged"
            else:
                sys.stderr.write(commit.stderr.decode())
                raise subprocess.CalledProcessError(commit.returncode, commit.args)
        if git_status == "pushed":
            run_cmd(["git", "push", "origin", "main"], cwd=repo)
    return {
        "repo_id": repo_id,
        "repo_status": create_dataset_repo(repo_name, config),
        "git_status": git_status,
        "seed_rows": rows,
        "seed_csv_sha256": sha256_file(seed_csv),
    }


def default_seed_dataset_refs(namespace: str, repo_name: str, filename: str) -> list[str]:
    repo_id = f"{namespace}/{repo_name}"
    return [
        f"hf://datasets/{repo_id}/{filename}",
        f"hf://datasets/{repo_id}/resolve/main/{filename}",
        f"{repo_id}#{filename}",
        f"{repo_id}/{filename}",
    ]


def build_job_payload(
    *,
    name: str,
    namespace: str,
    seed_dataset_ref: str,
    num_records: int,
    model_provider: str,
    model: str,
    model_alias: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_parallel_requests: int,
    timeout_s: int,
    description: str,
    experiment_label: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "namespace": namespace,
        "description": description,
        "spec": {
            "num_records": num_records,
            "config": {
                "seed_config": {
                    "dataset": seed_dataset_ref,
                    "sampling_strategy": "ordered",
                },
                "model_configs": [
                    {
                        "alias": model_alias,
                        "model": model,
                        "provider": model_provider,
                        "inference_parameters": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_tokens": max_tokens,
                            "max_parallel_requests": max_parallel_requests,
                            "timeout": timeout_s,
                        },
                    }
                ],
                "columns": [
                    *[
                        {"name": column_name, "column_type": "seed-dataset"}
                        for column_name in SEED_DATASET_COLUMNS
                    ],
                    {
                        "name": "qa_pairs_json",
                        "column_type": "llm-structured",
                        "prompt": DATA_DESIGNER_PROMPT_TEMPLATE,
                        "model_alias": model_alias,
                        "system_prompt": SYSTEM_PROMPT,
                        "output_format": QA_PAIRS_OUTPUT_FORMAT,
                    },
                ],
            },
        },
        "custom_fields": {
            "experiment": experiment_label,
            "seed_dataset_ref": seed_dataset_ref,
            "model_provider": model_provider,
            "model": model,
        },
    }


def _decode_response_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except json.JSONDecodeError:
        rows = []
        for line in response.text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return {"object": "jsonl", "data": rows}


def data_designer_post(config: ServiceConfig, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(
        f"{config.data_designer_url.rstrip('/')}{path}",
        json=payload,
        timeout=900,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Data Designer POST {path} failed: {response.status_code} {response.text}")
    decoded = _decode_response_body(response)
    if not isinstance(decoded, dict):
        return {"object": "response", "data": decoded}
    return decoded


def data_designer_get(config: ServiceConfig, path: str, timeout: float = 60.0) -> httpx.Response:
    response = httpx.get(f"{config.data_designer_url.rstrip('/')}{path}", timeout=timeout)
    response.raise_for_status()
    return response


def preview_seed_ref(config: ServiceConfig, payload: dict[str, Any]) -> dict[str, Any]:
    preview_payload = json.loads(json.dumps(payload))
    preview_payload["spec"]["num_records"] = 1
    # /preview accepts the job spec shape without job metadata.
    return data_designer_post(config, "/v1/data-designer/preview", preview_payload["spec"])


def submit_job(config: ServiceConfig, payload: dict[str, Any]) -> dict[str, Any]:
    return data_designer_post(config, "/v1/data-designer/jobs", payload)


def job_status(config: ServiceConfig, job_id: str) -> dict[str, Any]:
    return data_designer_get(config, f"/v1/data-designer/jobs/{job_id}/status").json()


def wait_for_job(config: ServiceConfig, job_id: str, interval_s: int, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = job_status(config, job_id)
        status = last.get("status")
        print(json.dumps({"job_id": job_id, "status": status, "updated_at": utc_now()}), flush=True)
        if status in {"completed", "error", "cancelled"}:
            return last
        time.sleep(interval_s)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s; last status={last}")


def download_results(config: ServiceConfig, job_id: str, experiment_dir: Path) -> dict[str, Any]:
    out_dir = experiment_dir / "data_designer" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_list = data_designer_get(config, f"/v1/data-designer/jobs/{job_id}/results").json()
    write_json(experiment_dir / "data_designer" / "results_manifest.json", result_list)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "data_designer_job_id": job_id,
        "result_list": result_list,
    }
    try:
        dataset_resp = data_designer_get(
            config,
            f"/v1/data-designer/jobs/{job_id}/results/dataset/download",
            timeout=300,
        )
    except httpx.HTTPStatusError as exc:
        result["dataset_download"] = None
        result["download_endpoint_error"] = {
            "status_code": exc.response.status_code,
            "url": str(exc.request.url),
            "body": exc.response.text[:2000],
        }
        try:
            result["direct_result_repo"] = clone_result_repo(config, job_id, experiment_dir)
            result["download_strategy"] = "direct_data_store_git_clone"
        except Exception as clone_exc:
            result["download_strategy"] = "failed"
            result["direct_result_repo_error"] = str(clone_exc)
            write_json(experiment_dir / "data_designer" / "download_manifest.json", result)
            raise RuntimeError(
                "Data Designer result download failed and direct NeMo Data Store "
                "recovery failed. Set DATA_STORE_USER/DATA_STORE_PASSWORD if the "
                "result repo requires authentication."
            ) from exc
    else:
        dataset_path = out_dir / "dataset_download.bin"
        dataset_path.write_bytes(dataset_resp.content)
        result["download_strategy"] = "data_designer_download_endpoint"
        result["dataset_download"] = {
            "path": str(dataset_path),
            "bytes": dataset_path.stat().st_size,
            "sha256": sha256_file(dataset_path),
            "content_type": dataset_resp.headers.get("content-type"),
        }
    write_json(experiment_dir / "data_designer" / "download_manifest.json", result)
    return result


def write_submission_artifacts(
    experiment_dir: Path,
    upload_result: dict[str, Any],
    payload: dict[str, Any],
    job: dict[str, Any] | None,
) -> None:
    dd_dir = experiment_dir / "data_designer"
    write_json(dd_dir / "seed_upload_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        **upload_result,
    })
    write_json(dd_dir / "job_request.json", payload)
    if job:
        write_json(dd_dir / "job_response.json", job)
        plan_path = dd_dir / "submission_plan.json"
        if plan_path.exists():
            plan = read_json(plan_path)
            plan["status"] = "submitted"
            data_designer = plan.setdefault("data_designer", {})
            if not isinstance(data_designer, dict):
                data_designer = {}
                plan["data_designer"] = data_designer
            data_designer.update({
                "job_id": job.get("id"),
                "job_name": job.get("name"),
                "model": payload["spec"]["config"]["model_configs"][0]["model"],
                "model_provider": payload["spec"]["config"]["model_configs"][0].get("provider"),
                "max_tokens": payload["spec"]["config"]["model_configs"][0]["inference_parameters"].get("max_tokens"),
                "seed_dataset_ref": payload["spec"]["config"]["seed_config"]["dataset"],
            })
            write_json(plan_path, plan)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    ap.add_argument("--data-designer-url", default=DEFAULT_DATA_DESIGNER_URL)
    ap.add_argument("--data-store-url", default=DEFAULT_DATA_STORE_URL)
    ap.add_argument("--data-store-git-base", default=DEFAULT_DATA_STORE_GIT_BASE)
    ap.add_argument("--data-store-user", default=os.getenv("DATA_STORE_USER"))
    ap.add_argument("--data-store-password", default=os.getenv("DATA_STORE_PASSWORD"))
    ap.add_argument("--seed-repo-name", default=DEFAULT_SEED_REPO_NAME)
    ap.add_argument("--seed-filename", default="seed_dataset.csv")
    ap.add_argument("--seed-dataset-ref", default=None)
    ap.add_argument("--mode", choices=["upload", "preview", "submit", "status", "wait", "collect"], required=True)
    ap.add_argument("--job-id", default=os.getenv("DATA_DESIGNER_JOB_ID"))
    ap.add_argument("--job-name", default="nim-curated-dd-llm-1b-v1")
    ap.add_argument("--experiment-label", default=None)
    ap.add_argument("--description", default=None)
    ap.add_argument("--model-provider", default=DEFAULT_MODEL_PROVIDER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-parallel-requests", type=int, default=4)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--wait-timeout-s", type=int, default=7200)
    ap.add_argument("--poll-interval-s", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ServiceConfig(
        namespace=args.namespace,
        data_designer_url=args.data_designer_url,
        data_store_url=args.data_store_url,
        data_store_git_base=args.data_store_git_base,
        data_store_user=args.data_store_user,
        data_store_password=args.data_store_password,
    )

    if args.mode in {"upload", "preview", "submit"}:
        repo_status = create_dataset_repo(args.seed_repo_name, config)
        upload_result = push_seed_repo(args.experiment_dir, args.seed_repo_name, config)
        upload_result["repo_status"] = repo_status
        seed_refs = default_seed_dataset_refs(args.namespace, args.seed_repo_name, args.seed_filename)
        seed_ref = args.seed_dataset_ref or seed_refs[0]
        seed_rows = upload_result["seed_rows"]
        payload = build_job_payload(
            name=args.job_name,
            namespace=args.namespace,
            seed_dataset_ref=seed_ref,
            num_records=seed_rows,
            model_provider=args.model_provider,
            model=args.model,
            model_alias=args.model_alias,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_parallel_requests=args.max_parallel_requests,
            timeout_s=args.timeout_s,
            description=(
                args.description
                or f"{args.experiment_label or args.job_name}: Data Designer synthetic QA generation "
                "from LLM-authored public documentation seeds."
            ),
            experiment_label=args.experiment_label or args.job_name,
        )
        write_submission_artifacts(args.experiment_dir, upload_result, payload, None)
        if args.dry_run or args.mode == "upload":
            print(json.dumps({"upload": upload_result, "seed_ref_candidates": seed_refs}, indent=2, sort_keys=True))
            return 0
        if args.mode == "preview":
            result = preview_seed_ref(config, payload)
            write_json(args.experiment_dir / "data_designer" / "preview_response.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        job = submit_job(config, payload)
        write_submission_artifacts(args.experiment_dir, upload_result, payload, job)
        print(json.dumps(job, indent=2, sort_keys=True))
        if args.wait_timeout_s > 0:
            terminal = wait_for_job(config, job["id"], args.poll_interval_s, args.wait_timeout_s)
            write_json(args.experiment_dir / "data_designer" / "job_terminal_status.json", terminal)
            write_json(job_artifact_dir(args.experiment_dir, job["id"]) / "job_terminal_status.json", terminal)
        return 0

    if args.mode in {"status", "wait", "collect"}:
        if not args.job_id:
            response_path = args.experiment_dir / "data_designer" / "job_response.json"
            if response_path.exists():
                args.job_id = read_json(response_path).get("id")
        if not args.job_id:
            raise RuntimeError("--job-id is required when job_response.json is unavailable")
        if args.mode == "status":
            print(json.dumps(job_status(config, args.job_id), indent=2, sort_keys=True))
            return 0
        if args.mode == "wait":
            terminal = wait_for_job(config, args.job_id, args.poll_interval_s, args.wait_timeout_s)
            write_json(args.experiment_dir / "data_designer" / "job_terminal_status.json", terminal)
            write_json(job_artifact_dir(args.experiment_dir, args.job_id) / "job_terminal_status.json", terminal)
            return 0
        result = download_results(config, args.job_id, args.experiment_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    raise ValueError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
