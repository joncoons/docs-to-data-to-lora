"""Collect durable model completions from an OpenAI-compatible endpoint.

This intentionally separates completion generation from judge/evaluator scoring.
Outputs are organized by dataset, base model, target kind, rank, and run id so
base, r16, and r32 dense adapters can be compared without artifact collisions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

DEFAULT_TARGET_API_URL = os.getenv("TARGET_API_URL", "http://rag-oai-proxy.runai-rag:8080")
DEFAULT_TARGET_API_KEY_ENV = os.getenv("TARGET_API_KEY_ENV", "TARGET_API_KEY")
DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("COMPLETIONS_OUTPUT_ROOT", "/mnt/nvme2/peft/evals/completions")
)
DEFAULT_MAX_TOKENS = 8192

_THINK_BALANCED = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TAIL = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_PRELUDE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_LORA_LLAMA_RE = re.compile(
    r"^lora-(?P<corpus>.+)-llama-(?P<base>\d+(?:\.\d+)?-\d+b)-r(?P<rank>\d+)$"
)
_LORA_NEMOTRON_RE = re.compile(
    r"^lora-(?P<corpus>.+)-nemotron-nano-30b-r(?P<rank>\d+)(?P<variant>-.+)?$"
)


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    cleaned = _THINK_BALANCED.sub("", text)
    cleaned = _THINK_PRELUDE.sub("", cleaned)
    cleaned = _THINK_OPEN_TAIL.sub("", cleaned)
    return cleaned.strip()


def _chat_url(api_url: str) -> str:
    base = api_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _safe_slug(value: str) -> str:
    value = value.strip().replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9._+-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value.lower() or "unknown"


def infer_dataset_slug(path: Path) -> str:
    if path.name in {"test_set_with_context.jsonl", "test_set.jsonl", "dataset.jsonl"}:
        return _safe_slug(path.parent.name)
    return _safe_slug(path.stem)


def parse_model_descriptor(model_id: str) -> dict[str, Any]:
    model = model_id.split("/", 1)[1] if "/" in model_id else model_id
    llama_lora = _LORA_LLAMA_RE.match(model)
    if llama_lora:
        rank = int(llama_lora.group("rank"))
        return {
            "model_id": model_id,
            "target_type": "lora",
            "corpus_slug": _safe_slug(llama_lora.group("corpus")),
            "base_slug": f"llama-{_safe_slug(llama_lora.group('base'))}",
            "target_slug": f"lora-{_safe_slug(llama_lora.group('corpus'))}",
            "rank": rank,
            "rank_slug": f"r{rank}",
        }
    nemotron_lora = _LORA_NEMOTRON_RE.match(model)
    if nemotron_lora:
        rank = int(nemotron_lora.group("rank"))
        variant = nemotron_lora.group("variant")
        return {
            "model_id": model_id,
            "target_type": "lora",
            "corpus_slug": _safe_slug(nemotron_lora.group("corpus")),
            "base_slug": "nemotron-nano-30b",
            "target_slug": f"lora-{_safe_slug(nemotron_lora.group('corpus'))}",
            "rank": rank,
            "rank_slug": f"r{rank}",
            "adapter_variant": _safe_slug(variant[1:]) if variant else None,
        }
    base_slug = model
    if base_slug.startswith("llama-") and base_slug.endswith("-instruct"):
        base_slug = base_slug[: -len("-instruct")]
    return {
        "model_id": model_id,
        "target_type": "base",
        "corpus_slug": None,
        "base_slug": _safe_slug(base_slug),
        "target_slug": "base",
        "rank": None,
        "rank_slug": "base",
    }


def build_run_dir(
    output_root: Path,
    *,
    dataset_slug: str,
    descriptor: dict[str, Any],
    run_id: str,
) -> Path:
    return (
        output_root
        / _safe_slug(dataset_slug)
        / descriptor["base_slug"]
        / descriptor["target_slug"]
        / descriptor["rank_slug"]
        / _safe_slug(run_id)
    )


def iter_jsonl_rows(path: Path, *, start_index: int = 0, limit: int | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    yielded = 0
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx < start_index or not line.strip():
                continue
            yield idx, json.loads(line)
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _token_count_estimate(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, round(len(text) / 4))


def build_token_counts(usage: dict[str, Any], raw_text: str, cleaned_text: str) -> dict[str, Any]:
    prompt_tokens = _coerce_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    usage_completion_tokens = _coerce_int(
        usage.get("completion_tokens") or usage.get("output_tokens")
    )
    raw_completion_tokens = usage_completion_tokens or _token_count_estimate(raw_text)
    raw_total_tokens = _coerce_int(usage.get("total_tokens"))
    stripped_think = raw_text != cleaned_text
    if stripped_think:
        cleaned_completion_tokens_est = min(
            raw_completion_tokens,
            _token_count_estimate(cleaned_text),
        )
    else:
        cleaned_completion_tokens_est = raw_completion_tokens
    if not raw_total_tokens:
        raw_total_tokens = prompt_tokens + raw_completion_tokens
    think_tokens_est = max(raw_completion_tokens - cleaned_completion_tokens_est, 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens_raw": raw_completion_tokens,
        "completion_tokens_cleaned_est": cleaned_completion_tokens_est,
        "think_tokens_est": think_tokens_est,
        "total_tokens_raw": raw_total_tokens,
        "raw_response_chars": len(raw_text),
        "cleaned_response_chars": len(cleaned_text),
        "think_chars_stripped": max(len(raw_text) - len(cleaned_text), 0),
        "completion_tokens_saved_by_stripping_est": think_tokens_est,
        "cleaned_completion_ratio_est": (
            cleaned_completion_tokens_est / raw_completion_tokens
            if raw_completion_tokens else None
        ),
        "reasoning_overhead_ratio_est": (
            think_tokens_est / raw_completion_tokens if raw_completion_tokens else None
        ),
    }


def summarize_counts(rows: list[dict[str, Any]], error_count: int) -> dict[str, Any]:
    totals = {
        "rows_completed": len(rows),
        "rows_failed": error_count,
        "prompt_tokens": 0,
        "completion_tokens_raw": 0,
        "completion_tokens_cleaned_est": 0,
        "think_tokens_est": 0,
        "total_tokens_raw": 0,
        "raw_response_chars": 0,
        "cleaned_response_chars": 0,
        "think_chars_stripped": 0,
    }
    for row in rows:
        counts = row.get("token_counts") or {}
        for key in list(totals):
            if key.startswith("rows_"):
                continue
            totals[key] += _coerce_int(counts.get(key))
    raw_completion = totals["completion_tokens_raw"]
    totals["reasoning_overhead_ratio_est"] = (
        totals["think_tokens_est"] / raw_completion if raw_completion else None
    )
    totals["cleaned_completion_ratio_est"] = (
        totals["completion_tokens_cleaned_est"] / raw_completion if raw_completion else None
    )
    return totals


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_retry_status_codes(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _source_row_index(row: dict[str, Any]) -> int | None:
    try:
        return int(row["source_row_index"])
    except (KeyError, TypeError, ValueError):
        return None


def _load_source_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indices: set[int] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            idx = _source_row_index(json.loads(line))
            if idx is not None:
                indices.add(idx)
    return indices


def _load_response_rows_by_index(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = _source_row_index(row)
            if idx is not None:
                rows[idx] = row
    return rows


def summarize_run_files(responses_path: Path, errors_path: Path) -> dict[str, Any]:
    responses_by_index = _load_response_rows_by_index(responses_path)
    error_indices = _load_source_indices(errors_path)
    unresolved_error_count = len(error_indices - set(responses_by_index))
    return summarize_counts(list(responses_by_index.values()), unresolved_error_count)


def should_retry_exception(exc: Exception, retry_status_codes: set[int]) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return int(status_code) in retry_status_codes
    return type(exc).__name__ in {
        "ConnectError",
        "ConnectTimeout",
        "NetworkError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutException",
        "WriteError",
        "WriteTimeout",
    }



async def _collect_one_row(
    *,
    client: Any,
    target_api_url: str,
    target_api_key: str | None,
    model_id: str,
    descriptor: dict[str, Any],
    dataset_slug: str,
    run_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    prompt_field: str,
    system_field: str,
    reference_field: str,
    max_attempts: int,
    retry_backoff_s: float,
    retry_backoff_max_s: float,
    retry_status_codes: set[int],
    ordinal: int,
    source_index: int,
    row: dict[str, Any],
) -> tuple[bool, int, dict[str, Any]]:
    messages = []
    if row.get(system_field):
        messages.append({"role": "system", "content": row[system_field]})
    messages.append({"role": "user", "content": row[prompt_field]})
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": max(temperature, 0.01),
        "top_p": top_p,
    }
    headers = {"Authorization": f"Bearer {target_api_key}"} if target_api_key else None
    row_metadata = {
        key: value
        for key, value in row.items()
        if key not in {prompt_field, system_field, reference_field}
    }
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        started = time.perf_counter()
        try:
            resp = await client.post(_chat_url(target_api_url), json=payload, headers=headers)
            latency_s = time.perf_counter() - started
            resp.raise_for_status()
            body = resp.json()
            choice = body["choices"][0]
            raw = choice.get("message", {}).get("content") or choice.get("text") or ""
            cleaned = strip_think_tags(raw)
            result = {
                "schema_version": "completion-row/v1",
                "run_id": run_id,
                "source_row_index": source_index,
                "row_ordinal": ordinal,
                "dataset_slug": dataset_slug,
                "model": descriptor,
                "prompt": row[prompt_field],
                "system": row.get(system_field),
                "reference_completion": row.get(reference_field),
                "response_raw": raw,
                "response": cleaned,
                "stripped_think_tags": raw != cleaned,
                "finish_reason": choice.get("finish_reason"),
                "usage": body.get("usage") or {},
                "token_counts": build_token_counts(body.get("usage") or {}, raw, cleaned),
                "latency_s": latency_s,
                "attempts": attempts,
                "generation_max_tokens": max_tokens,
                "target_api_url": target_api_url,
            }
            if row_metadata:
                result["row_metadata"] = row_metadata
            return True, ordinal, result
        except Exception as exc:  # noqa: BLE001 - durable batch collection records row-level failures.
            latency_s = time.perf_counter() - started
            retryable = should_retry_exception(exc, retry_status_codes)
            if retryable and attempts < max_attempts:
                delay_s = min(
                    retry_backoff_max_s,
                    retry_backoff_s * (2 ** (attempts - 1)),
                )
                log.warning(
                    "%s row=%d attempt=%d/%d failed; retrying in %.1fs: %s",
                    model_id,
                    ordinal,
                    attempts,
                    max_attempts,
                    delay_s,
                    exc,
                )
                await asyncio.sleep(delay_s)
                continue
            error = {
                "schema_version": "completion-error/v1",
                "run_id": run_id,
                "source_row_index": source_index,
                "row_ordinal": ordinal,
                "dataset_slug": dataset_slug,
                "model": descriptor,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "latency_s": latency_s,
                "attempts": attempts,
                "retryable": retryable,
            }
            if row_metadata:
                error["row_metadata"] = row_metadata
            return False, ordinal, error
    raise RuntimeError("unreachable retry loop state")


async def _collect_rows_async(
    *,
    rows_to_collect: list[tuple[int, int, dict[str, Any]]],
    responses_path: Path,
    errors_path: Path,
    target_api_url: str,
    target_api_key: str | None,
    model_id: str,
    descriptor: dict[str, Any],
    dataset_slug: str,
    run_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
    prompt_field: str,
    system_field: str,
    reference_field: str,
    fail_fast: bool,
    log_every: int,
    max_attempts: int,
    retry_backoff_s: float,
    retry_backoff_max_s: float,
    retry_status_codes: set[int],
    concurrency: int,
) -> int:
    import httpx

    limits = httpx.Limits(
        max_connections=max(concurrency, 1),
        max_keepalive_connections=max(concurrency, 1),
    )
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        async def run_one(
            item: tuple[int, int, dict[str, Any]],
        ) -> tuple[bool, int, dict[str, Any]]:
            ordinal, source_index, row = item
            async with semaphore:
                return await _collect_one_row(
                    client=client,
                    target_api_url=target_api_url,
                    target_api_key=target_api_key,
                    model_id=model_id,
                    descriptor=descriptor,
                    dataset_slug=dataset_slug,
                    run_id=run_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_field=prompt_field,
                    system_field=system_field,
                    reference_field=reference_field,
                    max_attempts=max_attempts,
                    retry_backoff_s=retry_backoff_s,
                    retry_backoff_max_s=retry_backoff_max_s,
                    retry_status_codes=retry_status_codes,
                    ordinal=ordinal,
                    source_index=source_index,
                    row=row,
                )

        tasks = [asyncio.create_task(run_one(item)) for item in rows_to_collect]
        error_count = 0
        with responses_path.open("a", encoding="utf-8") as out_fh, errors_path.open("a", encoding="utf-8") as err_fh:
            for task in asyncio.as_completed(tasks):
                ok, ordinal, record = await task
                if ok:
                    out_fh.write(json.dumps(record, sort_keys=True) + "\n")
                    out_fh.flush()
                    if log_every and ordinal % log_every == 0:
                        log.info("%s row=%d completed", model_id, ordinal)
                else:
                    error_count += 1
                    err_fh.write(json.dumps(record, sort_keys=True) + "\n")
                    err_fh.flush()
                    log.error("%s row=%d failed: %s", model_id, ordinal, record["error"])
                    if fail_fast:
                        for pending in tasks:
                            if not pending.done():
                                pending.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        break
        return error_count


def collect_for_model(
    *,
    dataset_path: Path,
    dataset_slug: str,
    output_root: Path,
    target_api_url: str,
    target_api_key: str | None,
    model_id: str,
    run_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
    start_index: int,
    limit: int | None,
    prompt_field: str,
    system_field: str,
    reference_field: str,
    fail_fast: bool,
    log_every: int,
    resume: bool,
    retry_failed: bool,
    max_attempts: int,
    retry_backoff_s: float,
    retry_backoff_max_s: float,
    retry_status_codes: set[int],
    concurrency: int,
) -> Path:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    descriptor = parse_model_descriptor(model_id)
    run_dir = build_run_dir(
        output_root,
        dataset_slug=dataset_slug,
        descriptor=descriptor,
        run_id=run_id,
    )
    run_dir.mkdir(parents=True, exist_ok=resume)
    responses_path = run_dir / "responses.jsonl"
    errors_path = run_dir / "errors.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "token_summary.json"
    completed_indices = _load_source_indices(responses_path) if resume else set()
    failed_indices = _load_source_indices(errors_path) if resume else set()
    skip_indices = completed_indices if retry_failed else completed_indices | failed_indices
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "completion-run/v1",
        "run_id": run_id,
        "started_at": started_at,
        "dataset_path": str(dataset_path),
        "dataset_slug": dataset_slug,
        "target_api_url": target_api_url,
        "chat_url": _chat_url(target_api_url),
        "target_api_key_supplied": bool(target_api_key),
        "model": descriptor,
        "generation": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "timeout_s": timeout_s,
            "concurrency": concurrency,
        },
        "retry": {
            "max_attempts": max_attempts,
            "retry_backoff_s": retry_backoff_s,
            "retry_backoff_max_s": retry_backoff_max_s,
            "retry_status_codes": sorted(retry_status_codes),
        },
        "resume": {
            "enabled": resume,
            "retry_failed": retry_failed,
            "existing_completed_rows": len(completed_indices),
            "existing_failed_rows": len(failed_indices),
        },
        "output_files": {
            "responses": str(responses_path),
            "errors": str(errors_path),
            "token_summary": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)

    rows_to_collect: list[tuple[int, int, dict[str, Any]]] = []
    skipped_count = 0
    for ordinal, (source_index, row) in enumerate(
        iter_jsonl_rows(dataset_path, start_index=start_index, limit=limit), start=1
    ):
        if source_index in skip_indices:
            skipped_count += 1
            continue
        rows_to_collect.append((ordinal, source_index, row))

    error_count = asyncio.run(
        _collect_rows_async(
            rows_to_collect=rows_to_collect,
            responses_path=responses_path,
            errors_path=errors_path,
            target_api_url=target_api_url,
            target_api_key=target_api_key,
            model_id=model_id,
            descriptor=descriptor,
            dataset_slug=dataset_slug,
            run_id=run_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
            prompt_field=prompt_field,
            system_field=system_field,
            reference_field=reference_field,
            fail_fast=fail_fast,
            log_every=log_every,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            retry_backoff_max_s=retry_backoff_max_s,
            retry_status_codes=retry_status_codes,
            concurrency=concurrency,
        )
    )
    summary = summarize_run_files(responses_path, errors_path)
    summary["rows_skipped_existing"] = skipped_count
    summary["row_errors_this_invocation"] = error_count
    write_json(summary_path, summary)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["summary"] = summary
    write_json(manifest_path, manifest)
    if error_count and fail_fast:
        raise RuntimeError(f"failed after {error_count} row errors; see {errors_path}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-slug")
    parser.add_argument("--model", action="append", required=True, help="Target model id. Repeat for multiple targets.")
    parser.add_argument("--target-api-url", default=DEFAULT_TARGET_API_URL)
    parser.add_argument("--target-api-key-env", default=DEFAULT_TARGET_API_KEY_ENV)
    parser.add_argument("--target-api-key", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0001)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--system-field", default="system")
    parser.add_argument("--reference-field", default="completion")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue an existing run directory and skip rows already recorded.")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, retry rows that only have prior error records.")
    parser.add_argument("--max-attempts", type=int, default=1, help="Total attempts per row, including the first attempt.")
    parser.add_argument("--retry-backoff-s", type=float, default=5.0)
    parser.add_argument("--retry-backoff-max-s", type=float, default=60.0)
    parser.add_argument("--retry-status-codes", default="408,429,500,502,503,504")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum in-flight completion requests for each model.")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    dataset_slug = args.dataset_slug or infer_dataset_slug(args.dataset)
    target_api_key = args.target_api_key or os.getenv(args.target_api_key_env)
    run_dirs = []
    for model_id in args.model:
        run_dirs.append(
            collect_for_model(
                dataset_path=args.dataset,
                dataset_slug=dataset_slug,
                output_root=args.output_root,
                target_api_url=args.target_api_url,
                target_api_key=target_api_key,
                model_id=model_id,
                run_id=args.run_id,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout_s=args.timeout_s,
                start_index=args.start_index,
                limit=args.limit,
                prompt_field=args.prompt_field,
                system_field=args.system_field,
                reference_field=args.reference_field,
                fail_fast=args.fail_fast,
                log_every=args.log_every,
                resume=args.resume,
                retry_failed=args.retry_failed,
                max_attempts=args.max_attempts,
                retry_backoff_s=args.retry_backoff_s,
                retry_backoff_max_s=args.retry_backoff_max_s,
                retry_status_codes=parse_retry_status_codes(args.retry_status_codes),
                concurrency=args.concurrency,
            )
        )
    for run_dir in run_dirs:
        print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
