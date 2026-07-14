#!/usr/bin/env python3
"""Run direct Kimi K2 pairwise judging over saved completion files.

This consumes durable ``responses.jsonl`` files from ``collect_completions.py``.
It is intended for pairwise matrix evaluation after completions have already
been collected, so no model target has to be redeployed just to re-answer the
same held-out rows.
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

log = logging.getLogger(__name__)

DEFAULT_JUDGE_API_URL = os.getenv(
    "JUDGE_API_URL",
    "https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/kimi-k2-6/v1",
)
DEFAULT_JUDGE_MODEL = os.getenv("EVALUATOR_JUDGE_MODEL", "kimi-k2-6")
DEFAULT_JUDGE_API_KEY_ENV = os.getenv("JUDGE_API_KEY_ENV", "KIMI_KEY")
DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("PAIRWISE_OUTPUT_ROOT", "<EVAL_ROOT>/pairwise-kimi")
)
DEFAULT_REPO_SUMMARY_DIR = _REPO_ROOT / "curator_dataset" / "experiments" / "20260709-curator-vs-le" / "golden_eval" / "kimi_norag_20260712"
REASONING_GENERATION_BUDGET = 8192
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

_THINK_BALANCED = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TAIL = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_PRELUDE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)

JUDGE_SYSTEM = (
    "You are an independent pairwise evaluation judge for NVIDIA technical QA. "
    "Compare only the supplied question, immutable reference answer, and model "
    "answers. Do not use retrieval context or external knowledge. Return JSON only."
)

JUDGE_USER_TEMPLATE = """\
Compare two answers (A and B) to the same question.

Question:
{question}

Immutable reference answer:
{reference}

Answer A:
{answer_a}

Answer B:
{answer_b}

Choose the better answer using this priority order:
- factual accuracy against the reference answer
- completeness against the reference answer
- lack of contradictions or unsupported additions relative to the reference answer
- clarity and concision

Return JSON only:
{{"winner": "A"|"B"|"TIE", "reason": "short reason"}}
"""


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


@dataclass(frozen=True)
class PairSpec:
    left_label: str
    left_path: Path
    right_label: str
    right_path: Path


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
        start = cleaned.find("{")
        if start < 0:
            raise
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge response was not a JSON object")
    return parsed


def coerce_winner(value: Any) -> str:
    winner = str(value or "").strip().upper()
    if winner in {"A", "B", "TIE"}:
        return winner
    return "TIE"


def coerce_pairwise_payload(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "winner": coerce_winner(payload.get("winner")),
        "reason": str(payload.get("reason") or "").strip(),
    }


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


def load_response_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        idx = source_row_index(row)
        if idx is not None:
            rows[idx] = row
    return rows


def sanitize_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", value).strip("-") or "unknown"


def pair_output_dir(pair: PairSpec, output_root: Path, eval_run_id: str) -> Path:
    left_rows = load_response_rows(pair.left_path)
    dataset = "unknown_dataset"
    if left_rows:
        first = left_rows[sorted(left_rows)[0]]
        dataset = str(first.get("dataset_slug") or dataset)
    pair_slug = f"{sanitize_path_part(pair.left_label)}__vs__{sanitize_path_part(pair.right_label)}"
    return output_root / sanitize_path_part(dataset) / pair_slug / eval_run_id


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


def build_judge_user(left: dict[str, Any], right: dict[str, Any], *, swapped: bool) -> str:
    prompt_parts = split_context_baked_prompt(str(left.get("prompt") or right.get("prompt") or ""))
    answer_a = strip_think_tags(str(right.get("response") or "")) if swapped else strip_think_tags(
        str(left.get("response") or "")
    )
    answer_b = strip_think_tags(str(left.get("response") or "")) if swapped else strip_think_tags(
        str(right.get("response") or "")
    )
    return JUDGE_USER_TEMPLATE.format(
        question=prompt_parts["question"],
        reference=str(left.get("reference_completion") or right.get("reference_completion") or ""),
        answer_a=answer_a,
        answer_b=answer_b,
    )


def map_winner_to_side(winner: str, *, swapped: bool) -> str:
    winner = coerce_winner(winner)
    if winner == "TIE":
        return "tie"
    if not swapped:
        return "left" if winner == "A" else "right"
    return "right" if winner == "A" else "left"


def resolve_position_swap(mapped_winners: list[str]) -> tuple[str, str]:
    """Resolve one or two mapped position-swap winners to left/right/tie."""
    non_empty = [winner for winner in mapped_winners if winner]
    if not non_empty:
        return "tie", "empty"
    unique = set(non_empty)
    if len(unique) == 1:
        return non_empty[0], "agree"
    non_ties = [winner for winner in non_empty if winner != "tie"]
    if len(set(non_ties)) == 1:
        return non_ties[0], "weak"
    return "tie", "conflict"


async def judge_once(
    client: Any,
    config: JudgeConfig,
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    swapped: bool,
    row_ordinal: int,
) -> dict[str, Any]:
    user = build_judge_user(left, right, swapped=swapped)
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
                    "pairwise judge retry row=%s left_line=%s right_line=%s position=%s attempt=%d/%d status=%d backoff=%.1fs",
                    row_ordinal,
                    source_line_number(left),
                    source_line_number(right),
                    "right_as_a" if swapped else "left_as_a",
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
            parsed = coerce_pairwise_payload(parse_json_object(cleaned))
            mapped_winner = map_winner_to_side(parsed["winner"], swapped=swapped)
            return {
                "position": "right_as_a" if swapped else "left_as_a",
                "raw_winner": parsed["winner"],
                "mapped_winner": mapped_winner,
                "reason": parsed["reason"],
                "judge": {
                    "provider": "direct-llm-judge",
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
            }
        except Exception as exc:  # noqa: BLE001 - row-level retry and error capture.
            last_error = exc
            if attempt < config.max_attempts - 1 and is_retryable_exception(exc):
                delay = retry_delay_s(config, attempt)
                log.warning(
                    "pairwise judge retry row=%s left_line=%s right_line=%s position=%s attempt=%d/%d error=%s backoff=%.1fs",
                    row_ordinal,
                    source_line_number(left),
                    source_line_number(right),
                    "right_as_a" if swapped else "left_as_a",
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


async def judge_pair_row(
    client: Any,
    config: JudgeConfig,
    pair: PairSpec,
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    row_ordinal: int,
    position_swap: bool,
) -> dict[str, Any]:
    outcomes = [await judge_once(client, config, left, right, swapped=False, row_ordinal=row_ordinal)]
    if position_swap:
        outcomes.append(await judge_once(client, config, left, right, swapped=True, row_ordinal=row_ordinal))
    winner, agreement = resolve_position_swap([row["mapped_winner"] for row in outcomes])
    return {
        "schema_version": "direct-kimi-pairwise-row/v1",
        "source_row_index": source_row_index(left),
        "left_source_line_number": source_line_number(left),
        "right_source_line_number": source_line_number(right),
        "row_ordinal": row_ordinal,
        "dataset_slug": left.get("dataset_slug") or right.get("dataset_slug"),
        "left_label": pair.left_label,
        "right_label": pair.right_label,
        "left_model": left.get("model"),
        "right_model": right.get("model"),
        "winner": winner,
        "agreement": agreement,
        "outcomes": outcomes,
        "target_token_counts": {
            "left": left.get("token_counts") or {},
            "right": right.get("token_counts") or {},
        },
    }


async def judge_rows(
    rows: list[tuple[int, dict[str, Any], dict[str, Any]]],
    *,
    pair: PairSpec,
    config: JudgeConfig,
    pairwise_path: Path,
    errors_path: Path,
    failed_lines_path: Path,
    concurrency: int,
    log_every: int,
    position_swap: bool,
) -> int:
    if httpx is None:
        raise RuntimeError("httpx is required to call the judge endpoint")
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    error_count = 0

    async with httpx.AsyncClient(timeout=config.timeout_s, limits=limits) as client:
        async def run_one(item: tuple[int, dict[str, Any], dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
            row_ordinal, left, right = item
            async with semaphore:
                try:
                    return True, await judge_pair_row(
                        client,
                        config,
                        pair,
                        left,
                        right,
                        row_ordinal=row_ordinal,
                        position_swap=position_swap,
                    )
                except Exception as exc:  # noqa: BLE001 - durable batch records row-level failures.
                    return False, {
                        "schema_version": "direct-kimi-pairwise-error/v1",
                        "source_row_index": source_row_index(left),
                        "left_source_line_number": source_line_number(left),
                        "right_source_line_number": source_line_number(right),
                        "row_ordinal": row_ordinal,
                        "dataset_slug": left.get("dataset_slug") or right.get("dataset_slug"),
                        "left_label": pair.left_label,
                        "right_label": pair.right_label,
                        "left_responses_path": str(pair.left_path),
                        "right_responses_path": str(pair.right_path),
                        "max_attempts": config.max_attempts,
                        "retry_backoff_s": config.retry_backoff_s,
                        "retry_backoff_max_s": config.retry_backoff_max_s,
                        "retry_jitter_s": config.retry_jitter_s,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failed_line": {
                            "left": {
                                "responses_path": str(pair.left_path),
                                "source_line_number": source_line_number(left),
                                "source_row_index": source_row_index(left),
                                "row": left,
                            },
                            "right": {
                                "responses_path": str(pair.right_path),
                                "source_line_number": source_line_number(right),
                                "source_row_index": source_row_index(right),
                                "row": right,
                            },
                        },
                    }

        tasks = [asyncio.create_task(run_one(item)) for item in rows]
        with pairwise_path.open("a", encoding="utf-8") as out_fh, errors_path.open(
            "a", encoding="utf-8"
        ) as err_fh, failed_lines_path.open("a", encoding="utf-8") as failed_fh:
            for task in asyncio.as_completed(tasks):
                ok, record = await task
                if ok:
                    out_fh.write(json.dumps(record, sort_keys=True) + "\n")
                    out_fh.flush()
                    if log_every and record["row_ordinal"] % log_every == 0:
                        log.info("judged pair row=%s", record["row_ordinal"])
                else:
                    error_count += 1
                    err_fh.write(json.dumps(record, sort_keys=True) + "\n")
                    err_fh.flush()
                    failed_fh.write(json.dumps(record["failed_line"], sort_keys=True) + "\n")
                    failed_fh.flush()
                    log.error(
                        "judge failed pair row=%s left_line=%s right_line=%s source_row_index=%s: %s",
                        record["row_ordinal"],
                        record.get("left_source_line_number"),
                        record.get("right_source_line_number"),
                        record.get("source_row_index"),
                        record["error"],
                    )
    return error_count


def sum_token_counts(rows: Iterable[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens_raw": 0,
        "completion_tokens_cleaned_est": 0,
        "think_tokens_est": 0,
        "total_tokens_raw": 0,
    }
    for row in rows:
        counts: Any = row
        for key in key_path:
            counts = counts.get(key) if isinstance(counts, dict) else {}
        counts = counts or {}
        totals["prompt_tokens"] += _coerce_int(counts.get("prompt_tokens"))
        totals["completion_tokens_raw"] += _coerce_int(counts.get("completion_tokens_raw"))
        totals["completion_tokens_cleaned_est"] += _coerce_int(
            counts.get("completion_tokens_cleaned_est")
        )
        totals["think_tokens_est"] += _coerce_int(counts.get("think_tokens_est"))
        totals["total_tokens_raw"] += _coerce_int(counts.get("total_tokens_raw"))
    return totals


def summarize_pairwise(pairwise_path: Path, errors_path: Path) -> dict[str, Any]:
    pair_rows = list(iter_jsonl(pairwise_path)) if pairwise_path.exists() else []
    error_rows = list(iter_jsonl(errors_path)) if errors_path.exists() else []
    scored_indices = {source_row_index(row) for row in pair_rows if source_row_index(row) is not None}
    unresolved_errors = [
        row for row in error_rows
        if source_row_index(row) not in scored_indices
    ]
    wins = {"left": 0, "right": 0, "tie": 0}
    agreements: dict[str, int] = {}
    judge_totals = {
        "prompt_tokens": 0,
        "completion_tokens_raw": 0,
        "completion_tokens_cleaned_est": 0,
        "think_tokens_est": 0,
        "total_tokens_raw": 0,
    }
    for row in pair_rows:
        wins[row.get("winner") if row.get("winner") in wins else "tie"] += 1
        agreement = str(row.get("agreement") or "unknown")
        agreements[agreement] = agreements.get(agreement, 0) + 1
        for outcome in row.get("outcomes") or []:
            counts = ((outcome.get("judge") or {}).get("token_counts") or {})
            judge_totals["prompt_tokens"] += _coerce_int(counts.get("prompt_tokens"))
            judge_totals["completion_tokens_raw"] += _coerce_int(counts.get("completion_tokens_raw"))
            judge_totals["total_tokens_raw"] += _coerce_int(counts.get("total_tokens_raw"))
    left_target = sum_token_counts(pair_rows, ("target_token_counts", "left"))
    right_target = sum_token_counts(pair_rows, ("target_token_counts", "right"))
    target_total_raw = left_target["total_tokens_raw"] + right_target["total_tokens_raw"]
    return {
        "rows_compared": len(pair_rows),
        "rows_failed": len(unresolved_errors),
        "row_error_attempts": len(error_rows),
        "wins": wins,
        "agreement_counts": agreements,
        "target_generation": {
            "left": left_target,
            "right": right_target,
            "combined_total_tokens_raw": target_total_raw,
        },
        "judge_scoring": judge_totals,
        "combined_total_tokens_raw": target_total_raw + judge_totals["total_tokens_raw"],
    }


def build_pair_rows(
    left_rows: dict[int, dict[str, Any]],
    right_rows: dict[int, dict[str, Any]],
    *,
    completed: set[int],
    limit: int | None,
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    common = sorted(set(left_rows) & set(right_rows))
    items: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for idx in common:
        if idx in completed:
            continue
        items.append((len(items) + 1, left_rows[idx], right_rows[idx]))
        if limit is not None and len(items) >= limit:
            break
    return items


def run_one_pair(
    pair: PairSpec,
    *,
    output_root: Path,
    eval_run_id: str,
    judge_config: JudgeConfig,
    limit: int | None,
    resume: bool,
    concurrency: int,
    log_every: int,
    position_swap: bool,
) -> Path:
    out_dir = pair_output_dir(pair, output_root, eval_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairwise_path = out_dir / "pairwise.jsonl"
    errors_path = out_dir / "errors.jsonl"
    failed_lines_path = out_dir / "failed_lines.jsonl"
    manifest_path = out_dir / "manifest.json"
    summary_path = out_dir / "summary.json"

    left_rows = load_response_rows(pair.left_path)
    right_rows = load_response_rows(pair.right_path)
    completed = load_completed_indices(pairwise_path) if resume else set()
    rows = build_pair_rows(left_rows, right_rows, completed=completed, limit=limit)
    missing_left = len(set(right_rows) - set(left_rows))
    missing_right = len(set(left_rows) - set(right_rows))
    manifest = {
        "schema_version": "direct-kimi-pairwise-run/v1",
        "eval_run_id": eval_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "left": {
            "label": pair.left_label,
            "responses_path": str(pair.left_path),
            "responses_manifest": read_json(pair.left_path.parent / "manifest.json"),
        },
        "right": {
            "label": pair.right_label,
            "responses_path": str(pair.right_path),
            "responses_manifest": read_json(pair.right_path.parent / "manifest.json"),
        },
        "row_alignment": {
            "left_rows": len(left_rows),
            "right_rows": len(right_rows),
            "common_rows": len(set(left_rows) & set(right_rows)),
            "missing_from_left": missing_left,
            "missing_from_right": missing_right,
        },
        "output_dir": str(out_dir),
        "judge": {
            "endpoint": judge_config.judge_api_url,
            "model": judge_config.judge_model,
            "max_tokens": judge_config.max_tokens,
            "temperature": judge_config.temperature,
            "position_swap": position_swap,
            "max_attempts": judge_config.max_attempts,
            "retry_backoff_s": judge_config.retry_backoff_s,
            "retry_backoff_max_s": judge_config.retry_backoff_max_s,
            "retry_jitter_s": judge_config.retry_jitter_s,
        },
        "rubric": {
            "mode": "golden_reference_no_rag",
            "uses_source_context": False,
        },
        "resume": {
            "enabled": resume,
            "existing_pairwise": len(completed),
            "rows_to_judge": len(rows),
        },
        "output_files": {
            "pairwise": str(pairwise_path),
            "errors": str(errors_path),
            "failed_lines": str(failed_lines_path),
            "summary": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    if rows:
        asyncio.run(
            judge_rows(
                rows,
                pair=pair,
                config=judge_config,
                pairwise_path=pairwise_path,
                errors_path=errors_path,
                failed_lines_path=failed_lines_path,
                concurrency=concurrency,
                log_every=log_every,
                position_swap=position_swap,
            )
        )
    summary = summarize_pairwise(pairwise_path, errors_path)
    write_json(summary_path, summary)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["summary"] = summary
    write_json(manifest_path, manifest)
    return out_dir


def mlflow_metrics_from_pairwise(summary: dict[str, Any]) -> dict[str, Any]:
    wins = summary.get("wins") or {}
    total = summary.get("rows_compared") or 0
    metrics: dict[str, Any] = {
        "rows.compared": summary.get("rows_compared"),
        "rows.failed": summary.get("rows_failed"),
        "rows.error_attempts": summary.get("row_error_attempts"),
        "wins.left": wins.get("left"),
        "wins.right": wins.get("right"),
        "wins.tie": wins.get("tie"),
        "tokens.target_total": (summary.get("target_generation") or {}).get("combined_total_tokens_raw"),
        "tokens.judge_total": (summary.get("judge_scoring") or {}).get("total_tokens_raw"),
        "tokens.combined_total": summary.get("combined_total_tokens_raw"),
    }
    if total:
        metrics["win_rate.left"] = (wins.get("left") or 0) / total
        metrics["win_rate.right"] = (wins.get("right") or 0) / total
        metrics["tie_rate"] = (wins.get("tie") or 0) / total
    return metrics


def log_pairwise_to_mlflow(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    judge_config: JudgeConfig,
) -> str | None:
    manifest = read_json(out_dir / "manifest.json")
    summary = read_json(out_dir / "summary.json")
    run_id = log_eval_artifacts(
        enabled=not args.no_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        artifact_location=args.mlflow_artifact_location,
        run_name=f"{args.eval_run_id}/pairwise/{out_dir.parent.name}",
        artifact_dir=out_dir,
        tags={
            "pipeline": "docs-to-data-to-lora",
            "pipeline.stage": "golden-evaluation",
            "eval.engine": "direct-llm-judge",
            "eval.scope": "pairwise",
            "eval.no_rag": "true",
            "eval.rubric": "golden_reference_no_rag",
            "eval.run_id": args.eval_run_id,
            "eval.judge.model": judge_config.judge_model,
            "eval.dataset_slug": out_dir.parent.parent.name if len(out_dir.parts) >= 2 else None,
            "eval.left_label": (manifest.get("left") or {}).get("label"),
            "eval.right_label": (manifest.get("right") or {}).get("label"),
            "eval.position_swap": str(args.position_swap),
        },
        params={
            "left_responses_path": (manifest.get("left") or {}).get("responses_path"),
            "right_responses_path": (manifest.get("right") or {}).get("responses_path"),
            "output_dir": manifest.get("output_dir"),
            "judge_api_url": judge_config.judge_api_url,
            "judge_max_tokens": judge_config.max_tokens,
            "judge_temperature": judge_config.temperature,
            "rows_compared": summary.get("rows_compared"),
            "rows_failed": summary.get("rows_failed"),
        },
        metrics=mlflow_metrics_from_pairwise(summary),
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
            scope="pairwise",
            artifact_dir=out_dir,
            manifest=manifest,
            summary=summary,
        )
    return run_id


def parse_pair(value: list[str]) -> PairSpec:
    if len(value) != 4:
        raise argparse.ArgumentTypeError("--pair requires LABEL_A RESPONSES_A LABEL_B RESPONSES_B")
    return PairSpec(value[0], Path(value[1]), value[2], Path(value[3]))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pair",
        action="append",
        nargs=4,
        metavar=("LEFT_LABEL", "LEFT_RESPONSES", "RIGHT_LABEL", "RIGHT_RESPONSES"),
        required=True,
        help="Compare two saved responses files. May be repeated.",
    )
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
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--position-swap", action=argparse.BooleanOptionalAction, default=True)
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
    pairs = [parse_pair(pair_args) for pair_args in args.pair]
    out_dirs = []
    for pair in pairs:
        out_dirs.append(
            run_one_pair(
                pair,
                output_root=args.output_root,
                eval_run_id=args.eval_run_id,
                judge_config=judge_config,
                limit=args.limit,
                resume=args.resume,
                concurrency=max(1, args.concurrency),
                log_every=args.log_every,
                position_swap=args.position_swap,
            )
        )
        log_pairwise_to_mlflow(
            out_dirs[-1],
            args=args,
            judge_config=judge_config,
        )
    for out_dir in out_dirs:
        print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
