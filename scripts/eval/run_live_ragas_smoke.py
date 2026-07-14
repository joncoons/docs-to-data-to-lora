"""Run a small NeMo Evaluator live RAGAS smoke test over precomputed rows.

Evaluator 1.5 only supports RAGAS metrics through /v1/evaluation/live, and
that endpoint accepts rows/dataset targets rather than model targets. This
script precomputes target answers through an OpenAI-compatible NIM endpoint,
strips reasoning tags defensively, then submits a rows target to live eval.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.evaluator_client import EvaluatorClient  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_EVALUATOR_URL = os.getenv("EVALUATOR_URL", "http://nemo-evaluator:7331")
DEFAULT_TARGET_API_URL = os.getenv("TARGET_API_URL", "http://rag-oai-proxy.runai-rag:8080")
DEFAULT_JUDGE_API_URL = os.getenv(
    "JUDGE_API_URL",
    "http://llm-judge.default.svc.cluster.local:8000/v1",
)
DEFAULT_MODEL_ID = os.getenv("EVALUATOR_SMOKE_MODEL", "llama-3.3-70b-instruct")
DEFAULT_JUDGE_MODEL_ID = os.getenv("EVALUATOR_JUDGE_MODEL", "frontier-judge")
DEFAULT_JUDGE_API_KEY_ENV = os.getenv("JUDGE_API_KEY_ENV", "LLM_API_KEY")
DEFAULT_DATASET_PATH = Path(
    os.getenv(
        "EVALUATOR_SMOKE_DATASET_PATH",
        "<DATASET_ROOT>/nim_curated/test_set_with_context.jsonl",
    )
)
DEFAULT_OUT = Path(
    os.getenv("EVALUATOR_SMOKE_OUT", str(_REPO_ROOT / "evals" / "live_ragas_smoke.json"))
)

REASONING_GENERATION_BUDGET = 8192
_THINK_BALANCED = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TAIL = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_PRELUDE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_RAGAS_ROWS_INPUT_TEMPLATE = (
    '{\n'
    '  "user_input":         {{ item.question | tojson }},\n'
    '  "retrieved_contexts": [{{ item.context | tojson }}],\n'
    '  "response":           {{ item.response | tojson }},\n'
    '  "reference":          {{ item.completion | tojson }}\n'
    '}'
)

_TOKEN_KEYS = (
    "prompt_tokens",
    "completion_tokens_raw",
    "completion_tokens_cleaned_est",
    "think_tokens_est",
    "total_tokens_raw",
)


def strip_think_tags(text: str) -> str:
    """Strip reasoning-model <think> blocks while preserving final answers."""
    if not text:
        return text
    cleaned = _THINK_BALANCED.sub("", text)
    cleaned = _THINK_PRELUDE.sub("", cleaned)
    cleaned = _THINK_OPEN_TAIL.sub("", cleaned)
    return cleaned.strip()


def load_jsonl_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _chat_url(api_url: str) -> str:
    base = api_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"




def split_context_baked_prompt(prompt: str) -> dict[str, str]:
    """Split Stage 3 context-baked prompts into RAGAS question/context fields."""
    marker = "\nQuestion:"
    if marker not in prompt:
        return {"context": prompt, "question": prompt}
    context, question = prompt.rsplit(marker, 1)
    return {"context": context.strip(), "question": question.strip()}


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _token_count_estimate(text: str) -> int:
    """Best-effort token count for cleaned answers when NIM only reports raw usage."""
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, round(len(text) / 4))


def build_token_counts(usage: dict[str, Any], raw_text: str, cleaned_text: str) -> dict[str, Any]:
    prompt_tokens = _coerce_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    raw_completion_tokens = _coerce_int(
        usage.get("completion_tokens") or usage.get("output_tokens")
    )
    raw_total_tokens = _coerce_int(usage.get("total_tokens"))
    cleaned_completion_tokens_est = min(
        raw_completion_tokens or _token_count_estimate(cleaned_text),
        _token_count_estimate(cleaned_text),
    )
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


def precompute_target_rows(
    rows: list[dict[str, Any]],
    *,
    target_api_url: str,
    model_id: str,
    max_tokens: int = REASONING_GENERATION_BUDGET,
    temperature: float = 0.0001,
    timeout_s: float = 900.0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_s) as client:
        for idx, row in enumerate(rows):
            messages = []
            if row.get("system"):
                messages.append({"role": "system", "content": row["system"]})
            messages.append({"role": "user", "content": row["prompt"]})
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": max(temperature, 0.0001),
                "top_p": 0.95,
            }
            resp = client.post(_chat_url(target_api_url), json=payload)
            resp.raise_for_status()
            body = resp.json()
            raw = body["choices"][0]["message"].get("content") or ""
            cleaned = strip_think_tags(raw)
            token_counts = build_token_counts(body.get("usage") or {}, raw, cleaned)
            log.info(
                "precomputed row=%d raw_completion_tokens=%d cleaned_est=%d think_est=%d",
                idx,
                token_counts["completion_tokens_raw"],
                token_counts["completion_tokens_cleaned_est"],
                token_counts["think_tokens_est"],
            )
            prompt_parts = split_context_baked_prompt(row["prompt"])
            out.append(
                {
                    "prompt": row["prompt"],
                    "context": prompt_parts["context"],
                    "question": prompt_parts["question"],
                    "completion": row["completion"],
                    "response": cleaned,
                    "system": row.get("system"),
                    "source_row_index": idx,
                    "response_model": model_id,
                    "response_usage": body.get("usage") or {},
                    "target_token_counts": token_counts,
                    "generation_max_tokens": max_tokens,
                }
            )
    return out


def build_judge_model(
    *,
    judge_api_url: str,
    model_id: str,
    max_tokens: int = REASONING_GENERATION_BUDGET,
    request_timeout_s: int = 600,
    max_retries: int = 3,
    api_key: str | None = None,
    endpoint_format: str = "openai",
) -> dict[str, Any]:
    endpoint = {
        "url": _chat_url(judge_api_url),
        "model_id": model_id,
        "format": endpoint_format,
    }
    if api_key:
        endpoint["api_key"] = api_key
    return {
        "name": model_id,
        "namespace": "default",
        "api_endpoint": endpoint,
        "prompt": {
            "inference_params": {
                "temperature": 0.0001,
                "max_tokens": max_tokens,
                "request_timeout": request_timeout_s,
                "max_retries": max_retries,
                "max_workers": 1,
            },
        },
    }


def build_live_payload(
    *,
    rows: list[dict[str, Any]],
    judge_model: dict[str, Any],
    metric_types: list[str],
    parallelism: int = 1,
) -> dict[str, Any]:
    metrics = {
        metric_type: {
            "type": metric_type,
            "params": {
                "judge": {"model": judge_model},
                "input_template": _RAGAS_ROWS_INPUT_TEMPLATE,
            },
        }
        for metric_type in metric_types
    }
    return {
        "namespace": "default",
        "description": "Stage 3 live RAGAS smoke over precomputed rows",
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
                "ragas_data_rubric": {
                    "type": "data",
                    "metrics": metrics,
                },
            },
        },
    }



_SECRET_KEYS = {"api_key", "authorization", "token", "access_token"}


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            if key.lower() in _SECRET_KEYS:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_secrets(nested)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def _normalize_usage(candidate: dict[str, Any]) -> dict[str, int] | None:
    nested = candidate.get("usage") or candidate.get("token_usage") or candidate.get("usage_metadata")
    if isinstance(nested, dict):
        candidate = nested
    prompt = _coerce_int(candidate.get("prompt_tokens") or candidate.get("input_tokens"))
    completion = _coerce_int(candidate.get("completion_tokens") or candidate.get("output_tokens"))
    total = _coerce_int(candidate.get("total_tokens")) or prompt + completion
    if not any((prompt, completion, total)):
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens_raw": completion,
        "total_tokens_raw": total,
    }


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)


def extract_judge_token_counts(live_result: dict[str, Any]) -> list[dict[str, int]]:
    """Extract judge token usage from Evaluator/RAGAS logs when LangChain exposes it."""
    seen: set[tuple[int, int, int]] = set()
    out: list[dict[str, int]] = []
    for task_rows in (live_result.get("logs") or {}).values():
        for task_row in task_rows or []:
            for request in task_row.get("requests") or []:
                for candidate in _walk_dicts(request):
                    usage = _normalize_usage(candidate)
                    if not usage:
                        continue
                    key = (
                        usage["prompt_tokens"],
                        usage["completion_tokens_raw"],
                        usage["total_tokens_raw"],
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(usage)
    return out


def _sum_counts(items: list[dict[str, Any]], key: str | None = None) -> dict[str, int]:
    totals = {token_key: 0 for token_key in _TOKEN_KEYS}
    for item in items:
        source = item.get(key, {}) if key else item
        for token_key in _TOKEN_KEYS:
            totals[token_key] += _coerce_int(source.get(token_key))
    return totals


def summarize_token_roi(target_rows: list[dict[str, Any]], live_result: dict[str, Any]) -> dict[str, Any]:
    target_totals = _sum_counts(target_rows, key="target_token_counts")
    judge_counts = extract_judge_token_counts(live_result)
    judge_totals = _sum_counts(judge_counts)
    combined_total_tokens = target_totals["total_tokens_raw"] + judge_totals["total_tokens_raw"]
    return {
        "target_generation": target_totals,
        "judge_scoring": {
            **judge_totals,
            "request_count_with_usage": len(judge_counts),
        },
        "combined_total_tokens_raw": combined_total_tokens,
        "target_reasoning_overhead_ratio_est": (
            target_totals["think_tokens_est"] / target_totals["completion_tokens_raw"]
            if target_totals["completion_tokens_raw"] else None
        ),
        "target_cleaned_completion_ratio_est": (
            target_totals["completion_tokens_cleaned_est"]
            / target_totals["completion_tokens_raw"]
            if target_totals["completion_tokens_raw"] else None
        ),
        "notes": [
            "Target prompt/completion/total tokens come from NIM usage when available.",
            "Cleaned completion and think-token counts are estimates after <think> stripping.",
            "Judge token usage is extracted from Evaluator live logs when exposed by LangChain/RAGAS.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator-url", default=DEFAULT_EVALUATOR_URL)
    ap.add_argument("--evaluator-api-key", default=os.getenv("EVALUATOR_API_KEY"))
    ap.add_argument(
        "--target-api-url",
        default=DEFAULT_TARGET_API_URL,
        help="OAI-compatible endpoint reachable from where this script runs.",
    )
    ap.add_argument(
        "--judge-api-url",
        default=DEFAULT_JUDGE_API_URL,
        help="OAI-compatible endpoint reachable from the Evaluator pod.",
    )
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help="Target model ID used for precomputed responses.")
    ap.add_argument("--judge-model-id", default=DEFAULT_JUDGE_MODEL_ID,
                    help="LLM judge model ID. Defaults to the configured LLM judge.")
    ap.add_argument("--judge-api-key-env", default=DEFAULT_JUDGE_API_KEY_ENV,
                    help="Environment variable containing the judge API key.")
    ap.add_argument("--judge-api-key", default=None,
                    help="Judge API key override. Prefer --judge-api-key-env to avoid shell history.")
    ap.add_argument("--judge-endpoint-format", default="openai",
                    help="Evaluator API endpoint format for the judge model.")
    ap.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--target-max-tokens", type=int, default=REASONING_GENERATION_BUDGET)
    ap.add_argument("--judge-max-tokens", type=int, default=REASONING_GENERATION_BUDGET)
    ap.add_argument("--judge-timeout-s", type=int, default=600)
    ap.add_argument("--judge-max-retries", type=int, default=3)
    ap.add_argument("--metrics", nargs="+", default=["faithfulness"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    judge_api_key = args.judge_api_key
    if not judge_api_key and args.judge_api_key_env:
        judge_api_key = os.getenv(args.judge_api_key_env)
    if args.judge_api_url.startswith("https://") and not judge_api_key:
        raise ValueError(
            f"judge API key is required for {args.judge_api_url}; "
            f"set ${args.judge_api_key_env} or pass --judge-api-key"
        )

    dataset_rows = load_jsonl_rows(args.dataset_path, args.limit)
    target_rows = precompute_target_rows(
        dataset_rows,
        target_api_url=args.target_api_url,
        model_id=args.model_id,
        max_tokens=args.target_max_tokens,
    )
    judge_model = build_judge_model(
        judge_api_url=args.judge_api_url,
        model_id=args.judge_model_id,
        max_tokens=args.judge_max_tokens,
        request_timeout_s=args.judge_timeout_s,
        max_retries=args.judge_max_retries,
        api_key=judge_api_key,
        endpoint_format=args.judge_endpoint_format,
    )
    payload = build_live_payload(rows=target_rows, judge_model=judge_model, metric_types=args.metrics)

    with EvaluatorClient(
        args.evaluator_url,
        api_key=args.evaluator_api_key,
        timeout=args.judge_timeout_s + 180,
    ) as client:
        result = client.submit_live(payload)

    artifact = {
        "payload": redact_secrets(payload),
        "result": redact_secrets(result),
        "token_roi_summary": summarize_token_roi(target_rows, result),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    log.info("live RAGAS status=%s wrote %s", result.get("status"), args.out)
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
