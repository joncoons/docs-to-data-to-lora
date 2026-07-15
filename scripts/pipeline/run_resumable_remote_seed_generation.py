#!/usr/bin/env python3
"""Resumable LLM seed generation for grounded Data Designer augmentation."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.build_data_designer_seed_from_grounded import (  # noqa: E402
    SeedConfig,
    _token_usage_metrics,
    artifact_manifest,
    build_seed_record,
    build_submission_plan,
    load_candidate_rows,
    select_candidates,
    source_file_manifest,
    stable_id,
    utc_now,
    write_json,
    write_jsonl,
    write_observability,
    write_seed_csv,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        fh.flush()


def expected_seed_id(config: SeedConfig, candidate: dict[str, Any], sequence_index: int) -> str:
    return stable_id(
        "ddseed",
        config.collection,
        candidate["source_row_index"],
        candidate["prompt"],
        candidate["completion"],
        sequence_index,
    )


def load_key(path: Path | None, env_name: str) -> str | None:
    value = os.getenv(env_name)
    if value:
        return value
    if not path:
        return None
    text = path.read_text(errors="replace")
    match = re.search(r"(sk-[A-Za-z0-9_\-.]+|nvapi-[A-Za-z0-9_\-.]+)", text)
    return match.group(1) if match else None


def run(config: SeedConfig, *, key_file: Path | None, fallback_on_error: bool, progress_interval: int) -> dict[str, Any]:
    if config.mode != "llm":
        raise RuntimeError("resumable runner is intended for --mode llm")
    if not config.allow_external_llm:
        raise RuntimeError("--allow-external-llm is required")
    api_key = load_key(key_file, config.judge_api_key_env)
    if not api_key:
        raise RuntimeError(f"{config.judge_api_key_env} or --key-file is required")

    candidates, metrics = load_candidate_rows(config)
    selected = select_candidates(candidates, config.seed_count, config.random_seed)
    data_designer_dir = config.output_dir / "data_designer"
    seed_requests = data_designer_dir / "llm_seed_requests.jsonl"
    failures_path = data_designer_dir / "llm_seed_failures.jsonl"
    progress_path = data_designer_dir / "llm_seed_progress.json"

    existing_rows = read_jsonl(seed_requests)
    existing_by_id = {str(row.get("seed_id")): row for row in existing_rows if row.get("seed_id")}
    seeds_by_index: dict[int, dict[str, Any]] = {}
    remaining: list[tuple[int, dict[str, Any], str]] = []
    for idx, candidate in enumerate(selected):
        seed_id = expected_seed_id(config, candidate, idx)
        if seed_id in existing_by_id:
            seeds_by_index[idx] = existing_by_id[seed_id]
        else:
            remaining.append((idx, candidate, seed_id))

    started_at = time.time()
    completed = len(seeds_by_index)
    failures = 0

    def write_progress(status: str) -> None:
        elapsed = max(time.time() - started_at, 0.001)
        write_json(progress_path, {
            "schema_version": "resumable-llm-seeds.v1",
            "updated_at": utc_now(),
            "status": status,
            "collection": config.collection,
            "output_dir": str(config.output_dir),
            "seed_records_requested": len(selected),
            "seed_records_completed": completed,
            "seed_records_remaining": len(selected) - completed,
            "seed_records_failed_with_fallback": failures,
            "elapsed_s": round(elapsed, 3),
            "records_per_minute": round(completed / elapsed * 60, 3),
        })

    write_progress("running")

    def build_one(idx: int, candidate: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
        try:
            return idx, build_seed_record(config, candidate, idx, api_key=api_key), None
        except Exception as exc:
            failure = {
                "schema_version": "resumable-llm-seeds.v1",
                "created_at": utc_now(),
                "collection": config.collection,
                "source_row_index": candidate.get("source_row_index"),
                "source_sample_id": candidate.get("source_sample_id"),
                "seed_id": expected_seed_id(config, candidate, idx),
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }
            if not fallback_on_error:
                raise
            fallback_config = replace(config, mode="prepare")
            seed = build_seed_record(fallback_config, candidate, idx, api_key=None)
            seed["seed_author"] = "heuristic_fallback_after_llm_error"
            seed["llm"]["failure"] = failure
            return idx, seed, failure

    if remaining:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {pool.submit(build_one, idx, candidate): idx for idx, candidate, _ in remaining}
            for future in as_completed(futures):
                idx, seed, failure = future.result()
                seeds_by_index[idx] = seed
                append_jsonl(seed_requests, seed)
                completed += 1
                if failure:
                    failures += 1
                    append_jsonl(failures_path, failure)
                if completed % max(1, progress_interval) == 0 or completed == len(selected):
                    write_progress("running")
                    print(json.dumps({
                        "collection": config.collection,
                        "completed": completed,
                        "total": len(selected),
                        "failures_with_fallback": failures,
                        "updated_at": utc_now(),
                    }), flush=True)

    seeds = [seeds_by_index[idx] for idx in range(len(selected))]
    metrics.update(_token_usage_metrics(seeds))

    source_manifest_path = config.output_dir / "source_snapshot" / "manifest.json"
    write_json(source_manifest_path, source_file_manifest(config.dataset_dir))
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
    if failures_path.exists():
        artifacts.append(artifact_manifest(failures_path, config.output_dir, "llm_seed_failures"))
    write_observability(config.output_dir, config, plan, artifacts)
    write_progress("completed")
    return {
        "collection": config.collection,
        "mode": config.mode,
        "candidate_rows": metrics["candidate_rows"],
        "seed_records": len(seeds),
        "synthetic_pairs_requested": plan["metrics"]["synthetic_pairs_requested"],
        "fallback_failures": failures,
        "output_dir": str(config.output_dir),
        "seed_requests": str(seed_requests),
        "seed_csv": str(seed_csv),
        "submission_plan": str(submission_plan),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--source-dataset-name", default=None)
    ap.add_argument("--seed-count", type=int, default=0)
    ap.add_argument("--pairs-per-seed", type=int, default=5)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--judge-api-url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--judge-model", default="azure/anthropic/claude-sonnet-4-6")
    ap.add_argument("--judge-api-key-env", default="LLM_API_KEY")
    ap.add_argument("--key-file", type=Path, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--fallback-on-error", action="store_true")
    ap.add_argument("--progress-interval", type=int, default=25)
    return ap.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SeedConfig:
    return SeedConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        collection=args.collection,
        source_dataset_name=args.source_dataset_name or args.collection,
        mode="llm",
        seed_count=args.seed_count,
        pairs_per_seed=args.pairs_per_seed,
        random_seed=args.random_seed,
        judge_api_url=args.judge_api_url,
        judge_model=args.judge_model,
        judge_api_key_env=args.judge_api_key_env,
        allow_external_llm=True,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        max_workers=max(1, args.max_workers),
        max_retries=max(1, args.max_retries),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        config_from_args(args),
        key_file=args.key_file,
        fallback_on_error=args.fallback_on_error,
        progress_interval=args.progress_interval,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
