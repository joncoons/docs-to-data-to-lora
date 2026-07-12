#!/usr/bin/env python3
"""Run no-RAG direct Kimi K2 single-axis judging over saved completions.

This consumes durable ``responses.jsonl`` files from ``collect_completions.py``
and compares each saved model answer to the immutable golden reference answer.
No retrieval context is sent to the judge for the formal winner evaluation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - handled in minimal test envs.
    httpx = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.mlflow_export import (  # noqa: E402
    DEFAULT_MLFLOW_ARTIFACT_LOCATION,
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_MLFLOW_TRACKING_URI,
    log_eval_artifacts,
    write_repo_summary,
)


_THINK_BALANCED = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TAIL = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_PRELUDE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Strip reasoning-model think blocks before parsing or scoring."""
    if not text:
        return text
    cleaned = _THINK_BALANCED.sub("", text)
    cleaned = _THINK_PRELUDE.sub("", cleaned)
    cleaned = _THINK_OPEN_TAIL.sub("", cleaned)
    return cleaned.strip()


def split_context_baked_prompt(prompt: str) -> dict[str, str]:
    """Split Stage 3 context-baked prompts into question/context fields."""
    marker = "\nQuestion:"
    if marker not in prompt:
        return {"context": prompt, "question": prompt}
    context, question = prompt.rsplit(marker, 1)
    return {"context": context.strip(), "question": question.strip()}

log = logging.getLogger(__name__)

DEFAULT_JUDGE_API_URL = os.getenv(
    "JUDGE_API_URL",
    "https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/kimi-k2-6/v1",
)
DEFAULT_JUDGE_MODEL = os.getenv("EVALUATOR_JUDGE_MODEL", "kimi-k2-6")
DEFAULT_JUDGE_API_KEY_ENV = os.getenv("JUDGE_API_KEY_ENV", "KIMI_KEY")
DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("SINGLEAXIS_OUTPUT_ROOT", "/mnt/nvme2/peft/evals/singleaxis-kimi")
)
DEFAULT_REPO_SUMMARY_DIR = _REPO_ROOT / "curator_dataset" / "experiments" / "20260709-curator-vs-le" / "golden_eval" / "kimi_norag_20260712"
DEFAULT_COMPLETIONS_ROOT = Path(
    os.getenv("COMPLETIONS_OUTPUT_ROOT", "/mnt/nvme2/peft/evals/completions")
)
REASONING_GENERATION_BUDGET = 8192

JUDGE_SYSTEM = (
    "You are an independent evaluation judge for NVIDIA technical QA. "
    "Grade only from the supplied question, immutable reference answer, and "
    "model answer. Do not use retrieval context or external knowledge. Return "
    "JSON only."
)

JUDGE_USER_TEMPLATE = """\
Evaluate the model answer on four 1-5 axes.

Question:
{question}

Immutable reference answer:
{reference}

Model answer:
{response}

Scoring rubric:
- accuracy: factual match to the reference answer.
- completeness: coverage of the reference answer's important details.
- faithfulness: avoids contradictions or unsupported additions relative to the reference answer.
- clarity: concise, readable, and well formed.

Return JSON only:
{{"accuracy": int, "completeness": int, "faithfulness": int, "clarity": int, "reason": "short reason"}}
"""

SCORE_FIELDS = ("accuracy", "completeness", "faithfulness", "clarity")
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class JudgeConfig:
    judge_api_url: str
    judge_model: str
    api_key: str
    max_tokens: int
    temperature: float
    timeout_s: float
    max_attempts: int
    retry_backoff_s: float
    retry_backoff_max_s: float
    retry_jitter_s: float


def chat_url(api_url: str) -> str:
    base = api_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path, *, limit: int | None = None) -> Iterable[dict[str, Any]]:
    yielded = 0
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                row.setdefault("_source_line_number", line_number)
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    prompt = _coerce_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion = _coerce_int(usage.get("completion_tokens") or usage.get("output_tokens"))
    total = _coerce_int(usage.get("total_tokens")) or prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens_raw": completion,
        "total_tokens_raw": total,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_think_tags(text)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge response was not a JSON object")
    return parsed


def clamp_score(value: Any) -> int:
    score = _coerce_int(value)
    if score < 1:
        return 1
    if score > 5:
        return 5
    return score


def coerce_scores(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": clamp_score(payload.get("accuracy")),
        "completeness": clamp_score(payload.get("completeness")),
        "faithfulness": clamp_score(payload.get("faithfulness")),
        "clarity": clamp_score(payload.get("clarity")),
        "reason": str(payload.get("reason") or "").strip(),
    }


def build_judge_user(row: dict[str, Any]) -> str:
    prompt_parts = split_context_baked_prompt(str(row.get("prompt") or ""))
    return JUDGE_USER_TEMPLATE.format(
        question=prompt_parts["question"],
        reference=str(row.get("reference_completion") or ""),
        response=str(row.get("response") or ""),
    )


def source_row_index(row: dict[str, Any]) -> int | None:
    try:
        return int(row["source_row_index"])
    except (KeyError, TypeError, ValueError):
        return None


def source_line_number(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("source_line_number") or row["_source_line_number"])
    except (KeyError, TypeError, ValueError):
        return None


def load_completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indices: set[int] = set()
    for row in iter_jsonl(path):
        idx = source_row_index(row)
        if idx is not None:
            indices.add(idx)
    return indices


def latest_rows_by_source_index(path: Path) -> dict[int, dict[str, Any]]:
    """Return the latest score row for each source row index."""
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        idx = source_row_index(row)
        if idx is not None:
            rows[idx] = row
    return rows


def score_duplicate_count(path: Path) -> int:
    if not path.exists():
        return 0
    counts: Counter[int] = Counter()
    for row in iter_jsonl(path):
        idx = source_row_index(row)
        if idx is not None:
            counts[idx] += 1
    return sum(count - 1 for count in counts.values() if count > 1)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def write_deduped_scores(raw_scores_path: Path, deduped_scores_path: Path) -> int:
    rows = latest_rows_by_source_index(raw_scores_path)
    return write_jsonl(deduped_scores_path, (rows[idx] for idx in sorted(rows)))


def load_error_attempts(path: Path) -> dict[int, int]:
    """Return cross-invocation exhausted attempts by source row index."""
    attempts: dict[int, int] = {}
    if not path.exists():
        return attempts
    for row in iter_jsonl(path):
        idx = source_row_index(row)
        if idx is None:
            continue
        attempts[idx] = attempts.get(idx, 0) + max(1, _coerce_int(row.get("max_attempts")))
    return attempts


def completion_relative_dir(responses_path: Path, completions_root: Path) -> Path:
    try:
        return responses_path.parent.relative_to(completions_root)
    except ValueError:
        return Path(responses_path.parent.name)


def output_dir_for(
    responses_path: Path,
    *,
    completions_root: Path,
    output_root: Path,
    eval_run_id: str,
) -> Path:
    return output_root / completion_relative_dir(responses_path, completions_root) / eval_run_id


def retry_delay_s(config: JudgeConfig, attempt: int, response: Any | None = None) -> float:
    """Exponential backoff with bounded jitter; attempt is zero-based."""
    retry_after = None
    if response is not None:
        try:
            retry_after_header = response.headers.get("retry-after")
            retry_after = float(retry_after_header) if retry_after_header else None
        except (TypeError, ValueError):
            retry_after = None
    base = min(config.retry_backoff_s * (2 ** attempt), config.retry_backoff_max_s)
    if retry_after is not None:
        base = max(base, min(retry_after, config.retry_backoff_max_s))
    jitter = random.uniform(0.0, config.retry_jitter_s) if config.retry_jitter_s > 0 else 0.0
    return base + jitter


def is_retryable_exception(exc: Exception) -> bool:
    if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return True


async def judge_one(
    client: Any,
    config: JudgeConfig,
    row: dict[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    user = build_judge_user(row)
    payload = {
        "model": config.judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": max(config.temperature, 0.0),
        "max_tokens": config.max_tokens,
    }
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(config.max_attempts):
        started = time.perf_counter()
        try:
            resp = await client.post(chat_url(config.judge_api_url), headers=headers, json=payload)
            latency_s = time.perf_counter() - started
            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < config.max_attempts - 1:
                delay = retry_delay_s(config, attempt, resp)
                log.warning(
                    "judge retry row=%s line=%s attempt=%d/%d status=%d backoff=%.1fs",
                    source_row_index(row),
                    source_line_number(row),
                    attempt + 1,
                    config.max_attempts,
                    resp.status_code,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            body = resp.json()
            choice = body["choices"][0]
            raw = choice.get("message", {}).get("content") or choice.get("text") or ""
            cleaned = strip_think_tags(raw)
            parsed = parse_json_object(cleaned)
            scores = coerce_scores(parsed)
            return {
                "schema_version": "direct-kimi-singleaxis-row/v1",
                "source_row_index": source_row_index(row),
                "source_line_number": source_line_number(row),
                "row_ordinal": ordinal,
                "dataset_slug": row.get("dataset_slug"),
                "model": row.get("model"),
                "scores": scores,
                "judge": {
                    "provider": "kimi-k2",
                    "endpoint": config.judge_api_url,
                    "model": config.judge_model,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": body.get("usage") or {},
                    "token_counts": normalize_usage(body.get("usage") or {}),
                    "raw_response_chars": len(raw),
                    "cleaned_response_chars": len(cleaned),
                    "think_chars_stripped": max(len(raw) - len(cleaned), 0),
                    "attempts": attempt + 1,
                    "latency_s": latency_s,
                    "max_tokens": config.max_tokens,
                },
                "target_token_counts": row.get("token_counts") or {},
            }
        except Exception as exc:  # noqa: BLE001 - row-level retry and error capture.
            last_error = exc
            if attempt < config.max_attempts - 1 and is_retryable_exception(exc):
                delay = retry_delay_s(config, attempt)
                log.warning(
                    "judge retry row=%s line=%s attempt=%d/%d error=%s backoff=%.1fs",
                    source_row_index(row),
                    source_line_number(row),
                    attempt + 1,
                    config.max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                break
    assert last_error is not None
    raise last_error


async def judge_rows(
    rows: list[dict[str, Any]],
    *,
    config: JudgeConfig,
    scores_path: Path,
    errors_path: Path,
    failed_lines_path: Path,
    responses_path: Path,
    concurrency: int,
    log_every: int,
) -> int:
    if httpx is None:
        raise RuntimeError("httpx is required to call the judge endpoint")
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    error_count = 0

    async with httpx.AsyncClient(timeout=config.timeout_s, limits=limits) as client:
        async def run_one(item: tuple[int, dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
            ordinal, row = item
            async with semaphore:
                try:
                    return True, await judge_one(client, config, row, ordinal)
                except Exception as exc:  # noqa: BLE001 - durable batch records row-level failures.
                    return False, {
                        "schema_version": "direct-kimi-singleaxis-error/v1",
                        "source_row_index": source_row_index(row),
                        "source_line_number": source_line_number(row),
                        "row_ordinal": ordinal,
                        "dataset_slug": row.get("dataset_slug"),
                        "model": row.get("model"),
                        "responses_path": str(responses_path),
                        "max_attempts": config.max_attempts,
                        "retry_backoff_s": config.retry_backoff_s,
                        "retry_backoff_max_s": config.retry_backoff_max_s,
                        "retry_jitter_s": config.retry_jitter_s,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failed_line": {
                            "responses_path": str(responses_path),
                            "source_line_number": source_line_number(row),
                            "source_row_index": source_row_index(row),
                            "row": row,
                        },
                    }

        tasks = [asyncio.create_task(run_one((idx, row))) for idx, row in enumerate(rows, start=1)]
        with scores_path.open("a", encoding="utf-8") as out_fh, errors_path.open("a", encoding="utf-8") as err_fh, failed_lines_path.open("a", encoding="utf-8") as failed_fh:
            for task in asyncio.as_completed(tasks):
                ok, record = await task
                if ok:
                    out_fh.write(json.dumps(record, sort_keys=True) + "\n")
                    out_fh.flush()
                    if log_every and record["row_ordinal"] % log_every == 0:
                        log.info("judged row=%s", record["row_ordinal"])
                else:
                    error_count += 1
                    err_fh.write(json.dumps(record, sort_keys=True) + "\n")
                    err_fh.flush()
                    failed_fh.write(json.dumps(record["failed_line"], sort_keys=True) + "\n")
                    failed_fh.flush()
                    log.error(
                        "judge failed row=%s line=%s source_row_index=%s: %s",
                        record["row_ordinal"],
                        record.get("source_line_number"),
                        record.get("source_row_index"),
                        record["error"],
                    )
    return error_count


def sum_token_counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens_raw": 0,
        "completion_tokens_cleaned_est": 0,
        "think_tokens_est": 0,
        "total_tokens_raw": 0,
    }
    for row in rows:
        counts = row.get(key) or {}
        totals["prompt_tokens"] += _coerce_int(counts.get("prompt_tokens"))
        totals["completion_tokens_raw"] += _coerce_int(counts.get("completion_tokens_raw"))
        totals["completion_tokens_cleaned_est"] += _coerce_int(counts.get("completion_tokens_cleaned_est"))
        totals["think_tokens_est"] += _coerce_int(counts.get("think_tokens_est"))
        totals["total_tokens_raw"] += _coerce_int(counts.get("total_tokens_raw"))
    return totals


def summarize_scores(scores_path: Path, errors_path: Path) -> dict[str, Any]:
    score_rows_raw = list(iter_jsonl(scores_path)) if scores_path.exists() else []
    score_rows = list(latest_rows_by_source_index(scores_path).values())
    error_rows = list(iter_jsonl(errors_path)) if errors_path.exists() else []
    scored_indices = {source_row_index(row) for row in score_rows if source_row_index(row) is not None}
    unresolved_errors = [
        row for row in error_rows
        if source_row_index(row) not in scored_indices
    ]
    unresolved_indices = sorted({
        idx for idx in (source_row_index(row) for row in unresolved_errors)
        if idx is not None
    })
    unresolved_without_index = sum(1 for row in unresolved_errors if source_row_index(row) is None)
    means: dict[str, float | None] = {}
    for field in SCORE_FIELDS:
        values = [_coerce_int((row.get("scores") or {}).get(field)) for row in score_rows]
        values = [value for value in values if value]
        means[f"mean_{field}"] = sum(values) / len(values) if values else None
    target_totals = sum_token_counts(score_rows, "target_token_counts")
    judge_totals = sum_token_counts(
        (
            {
                "judge_token_counts": {
                    **((row.get("judge") or {}).get("token_counts") or {}),
                    "completion_tokens_cleaned_est": 0,
                    "think_tokens_est": 0,
                }
            }
            for row in score_rows
        ),
        "judge_token_counts",
    )
    return {
        "rows_scored": len(score_rows),
        "rows_scored_raw": len(score_rows_raw),
        "score_duplicate_rows": max(0, len(score_rows_raw) - len(score_rows)),
        "rows_failed": len(unresolved_indices) + unresolved_without_index,
        "unresolved_source_row_indices": unresolved_indices,
        "row_error_attempts": len(error_rows),
        "score_means": means,
        "target_generation": target_totals,
        "judge_scoring": judge_totals,
        "combined_total_tokens_raw": (
            target_totals["total_tokens_raw"] + judge_totals["total_tokens_raw"]
        ),
    }


def run_one_responses_file(
    responses_path: Path,
    *,
    completions_root: Path,
    output_root: Path,
    eval_run_id: str,
    judge_config: JudgeConfig,
    limit: int | None,
    resume: bool,
    concurrency: int,
    log_every: int,
    max_total_attempts_per_row: int,
) -> Path:
    out_dir = output_dir_for(
        responses_path,
        completions_root=completions_root,
        output_root=output_root,
        eval_run_id=eval_run_id,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.jsonl"
    deduped_scores_path = out_dir / "scores_deduped.jsonl"
    errors_path = out_dir / "errors.jsonl"
    failed_lines_path = out_dir / "failed_lines.jsonl"
    missing_rows_path = out_dir / "missing_rows.jsonl"
    rows_to_judge_path = out_dir / "rows_to_judge.jsonl"
    retry_exhausted_path = out_dir / "retry_exhausted_rows.jsonl"
    manifest_path = out_dir / "manifest.json"
    summary_path = out_dir / "summary.json"

    completed = load_completed_indices(scores_path) if resume else set()
    error_attempts = load_error_attempts(errors_path) if resume else {}
    missing_rows = [
        row
        for row in iter_jsonl(responses_path, limit=limit)
        if source_row_index(row) not in completed
    ]
    rows = []
    retry_exhausted_rows = []
    for row in missing_rows:
        idx = source_row_index(row)
        attempts = error_attempts.get(idx, 0) if idx is not None else 0
        if resume and idx is not None and attempts >= max_total_attempts_per_row:
            retry_exhausted_rows.append({
                "source_row_index": idx,
                "source_line_number": source_line_number(row),
                "prior_attempts": attempts,
                "max_total_attempts_per_row": max_total_attempts_per_row,
                "row": row,
            })
        else:
            rows.append(row)
    write_jsonl(missing_rows_path, missing_rows)
    write_jsonl(rows_to_judge_path, rows)
    write_jsonl(retry_exhausted_path, retry_exhausted_rows)
    write_deduped_scores(scores_path, deduped_scores_path)
    manifest = {
        "schema_version": "direct-kimi-singleaxis-run/v1",
        "eval_run_id": eval_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "responses_path": str(responses_path),
        "responses_manifest": read_json(responses_path.parent / "manifest.json"),
        "output_dir": str(out_dir),
        "judge": {
            "endpoint": judge_config.judge_api_url,
            "model": judge_config.judge_model,
            "max_tokens": judge_config.max_tokens,
            "temperature": judge_config.temperature,
            "max_attempts": judge_config.max_attempts,
            "max_total_attempts_per_row": max_total_attempts_per_row,
            "retry_backoff_s": judge_config.retry_backoff_s,
            "retry_backoff_max_s": judge_config.retry_backoff_max_s,
            "retry_jitter_s": judge_config.retry_jitter_s,
        },
        "rubric": {
            "mode": "golden_reference_no_rag",
            "uses_source_context": False,
            "score_fields": list(SCORE_FIELDS),
        },
        "resume": {
            "enabled": resume,
            "existing_scores": len(completed),
            "raw_score_duplicate_rows": score_duplicate_count(scores_path),
            "missing_rows": len(missing_rows),
            "retry_exhausted_rows": len(retry_exhausted_rows),
            "rows_to_judge": len(rows),
        },
        "output_files": {
            "scores": str(scores_path),
            "scores_deduped": str(deduped_scores_path),
            "errors": str(errors_path),
            "failed_lines": str(failed_lines_path),
            "missing_rows": str(missing_rows_path),
            "rows_to_judge": str(rows_to_judge_path),
            "retry_exhausted_rows": str(retry_exhausted_path),
            "summary": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    if rows:
        asyncio.run(
            judge_rows(
                rows,
                config=judge_config,
                scores_path=scores_path,
                errors_path=errors_path,
                failed_lines_path=failed_lines_path,
                responses_path=responses_path,
                concurrency=concurrency,
                log_every=log_every,
            )
        )
    write_deduped_scores(scores_path, deduped_scores_path)
    summary = summarize_scores(scores_path, errors_path)
    write_json(summary_path, summary)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["summary"] = summary
    write_json(manifest_path, manifest)
    return out_dir


def mlflow_metrics_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "rows.scored": summary.get("rows_scored"),
        "rows.failed": summary.get("rows_failed"),
        "rows.error_attempts": summary.get("row_error_attempts"),
        "tokens.target_total": (summary.get("target_generation") or {}).get("total_tokens_raw"),
        "tokens.judge_total": (summary.get("judge_scoring") or {}).get("total_tokens_raw"),
        "tokens.combined_total": summary.get("combined_total_tokens_raw"),
    }
    for key, value in (summary.get("score_means") or {}).items():
        metrics[f"score.{key.replace('mean_', '')}.mean"] = value
    return metrics


def log_singleaxis_to_mlflow(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    judge_config: JudgeConfig,
) -> str | None:
    manifest = read_json(out_dir / "manifest.json")
    summary = read_json(out_dir / "summary.json")
    response_manifest = manifest.get("responses_manifest") or {}
    model = response_manifest.get("model") or {}
    run_id = log_eval_artifacts(
        enabled=not args.no_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        artifact_location=args.mlflow_artifact_location,
        run_name=f"{args.eval_run_id}/singleaxis/{out_dir.parent.name}",
        artifact_dir=out_dir,
        tags={
            "pipeline": "docs-to-data-to-lora",
            "pipeline.stage": "golden-evaluation",
            "eval.engine": "direct-kimi",
            "eval.scope": "singleaxis",
            "eval.no_rag": "true",
            "eval.rubric": "golden_reference_no_rag",
            "eval.run_id": args.eval_run_id,
            "eval.judge.model": judge_config.judge_model,
            "eval.dataset_slug": response_manifest.get("dataset_slug"),
            "eval.base_slug": model.get("base_slug"),
            "eval.target_slug": model.get("target_slug"),
            "eval.rank_slug": model.get("rank_slug"),
        },
        params={
            "responses_path": manifest.get("responses_path"),
            "output_dir": manifest.get("output_dir"),
            "judge_api_url": judge_config.judge_api_url,
            "judge_max_tokens": judge_config.max_tokens,
            "judge_temperature": judge_config.temperature,
            "rows_scored": summary.get("rows_scored"),
            "rows_failed": summary.get("rows_failed"),
        },
        metrics=mlflow_metrics_from_summary(summary),
    )
    if run_id:
        manifest["mlflow"] = {
            "tracking_uri": args.mlflow_tracking_uri,
            "experiment": args.mlflow_experiment,
            "artifact_location": args.mlflow_artifact_location,
            "run_id": run_id,
        }
        write_json(out_dir / "manifest.json", manifest)
    if args.repo_summary_dir:
        write_repo_summary(
            repo_summary_dir=args.repo_summary_dir,
            eval_run_id=args.eval_run_id,
            scope="singleaxis",
            artifact_dir=out_dir,
            manifest=manifest,
            summary=summary,
        )
    return run_id


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses", action="append", required=True, type=Path)
    ap.add_argument("--completions-root", type=Path, default=DEFAULT_COMPLETIONS_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--eval-run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--judge-api-url", default=DEFAULT_JUDGE_API_URL)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--judge-api-key-env", default=DEFAULT_JUDGE_API_KEY_ENV)
    ap.add_argument("--judge-api-key", default=None)
    ap.add_argument("--judge-max-tokens", type=int, default=REASONING_GENERATION_BUDGET)
    ap.add_argument("--judge-temperature", type=float, default=0.0)
    ap.add_argument("--judge-timeout-s", type=float, default=600.0)
    ap.add_argument("--judge-max-attempts", type=int, default=5)
    ap.add_argument("--retry-backoff-s", type=float, default=5.0)
    ap.add_argument("--retry-backoff-max-s", type=float, default=120.0)
    ap.add_argument("--retry-jitter-s", type=float, default=2.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument(
        "--max-total-attempts-per-row",
        type=int,
        default=None,
        help="Cross-invocation attempt cap for resume cleanup; defaults to --judge-max-attempts.",
    )
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    ap.add_argument("--mlflow-experiment", default=DEFAULT_MLFLOW_EXPERIMENT)
    ap.add_argument("--mlflow-artifact-location", default=DEFAULT_MLFLOW_ARTIFACT_LOCATION)
    ap.add_argument("--repo-summary-dir", type=Path, default=DEFAULT_REPO_SUMMARY_DIR)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    api_key = args.judge_api_key or os.getenv(args.judge_api_key_env)
    if not api_key:
        raise ValueError(f"judge API key is required; set ${args.judge_api_key_env}")
    judge_config = JudgeConfig(
        judge_api_url=args.judge_api_url,
        judge_model=args.judge_model,
        api_key=api_key,
        max_tokens=args.judge_max_tokens,
        temperature=args.judge_temperature,
        timeout_s=args.judge_timeout_s,
        max_attempts=max(1, args.judge_max_attempts),
        retry_backoff_s=args.retry_backoff_s,
        retry_backoff_max_s=max(args.retry_backoff_s, args.retry_backoff_max_s),
        retry_jitter_s=max(0.0, args.retry_jitter_s),
    )
    out_dirs = []
    for responses_path in args.responses:
        out_dirs.append(
            run_one_responses_file(
                responses_path,
                completions_root=args.completions_root,
                output_root=args.output_root,
                eval_run_id=args.eval_run_id,
                judge_config=judge_config,
                limit=args.limit,
                resume=args.resume,
                concurrency=max(1, args.concurrency),
                log_every=args.log_every,
                max_total_attempts_per_row=max(
                    1,
                    args.max_total_attempts_per_row
                    if args.max_total_attempts_per_row is not None
                    else args.judge_max_attempts,
                ),
            )
        )
        log_singleaxis_to_mlflow(
            out_dirs[-1],
            args=args,
            judge_config=judge_config,
        )
    for out_dir in out_dirs:
        print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
