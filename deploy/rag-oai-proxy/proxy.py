"""OAI-compatible proxy fronting ALL Stage 3 eval targets.

Single sidecar that the Evaluator calls for every eval job. Two responsibilities:

  1. Routing — derive the upstream {URL, model_id} from the inbound `model`:
       Adapter targets (lora-*) and base targets (bare base model names)
       map to the corresponding LoRA-enabled NIM Service. The 49B comparator
       maps to nim-llm. Base inference vs LoRA inference is selected by what
       `model` we forward to the upstream NIM (NIM routes via NIM_PEFT_SOURCE).
  2. <think>...</think> scrubbing — applied to every response. Idempotent on
     responses without think tags (dense Llama path is a passthrough no-op).
     Keeps thinking ENABLED upstream so reasoning quality is preserved, but
     strips the raw blocks so pairwise judges compare only final answers.

Retrieval is NOT done here. Stage 3 test sets have retrieved context
pre-baked into each row's prompt (see bake_context_into_testset.py); the proxy
just forwards (system + user-message-with-context) to the upstream NIM.

Routes:
  GET  /v1/models              declared model_ids the proxy serves
  POST /v1/chat/completions    OAI shim for every Evaluator target
  GET  /health                 liveness
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException


# --- upstream NIM Service names per base model ----------------------------

_NIM_SERVICE_FOR_BASE: dict[str, str] = {
    "meta/llama-3.2-1b-instruct":              "nim-llama-3.2-1b",
    "meta/llama-3.2-3b-instruct":              "nim-llama-3.2-3b",
    "meta/llama-3.1-8b-instruct":              "nim-llama-3.1-8b",
    "nvidia/nemotron-3-nano-30b-a3b":          "nim-nemotron-nano",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": "nim-llm",
}

# Stage 3 LoRA adapter inventory: (corpus_slug, base_model, rank)
_ADAPTERS: list[tuple[str, str, int]] = [
    ("nim",        "meta/llama-3.2-1b-instruct",     16),
    ("nim",        "meta/llama-3.2-1b-instruct",     32),
    ("nim",        "meta/llama-3.2-3b-instruct",     16),
    ("nim",        "meta/llama-3.2-3b-instruct",     32),
    ("nim",        "meta/llama-3.1-8b-instruct",     16),
    ("nim",        "meta/llama-3.1-8b-instruct",     32),
    ("nim",        "nvidia/nemotron-3-nano-30b-a3b", 16),
    ("nemo-usvcs", "meta/llama-3.2-1b-instruct",     16),
    ("nemo-usvcs", "meta/llama-3.2-1b-instruct",     32),
    ("nemo-usvcs", "meta/llama-3.2-3b-instruct",     16),
    ("nemo-usvcs", "meta/llama-3.2-3b-instruct",     32),
    ("nemo-usvcs", "meta/llama-3.1-8b-instruct",     16),
    ("nemo-usvcs", "meta/llama-3.1-8b-instruct",     32),
    ("nemo-usvcs", "nvidia/nemotron-3-nano-30b-a3b", 16),
]


def _adapter_name(corpus_slug: str, base: str, rank: int) -> str:
    if base == "nvidia/nemotron-3-nano-30b-a3b":
        return f"lora-{corpus_slug}-nemotron-nano-30b-r{rank}"
    size_slug = base.replace("meta/llama-", "").replace("-instruct", "")
    return f"lora-{corpus_slug}-llama-{size_slug}-r{rank}"


def _base_target_name(base: str) -> str:
    """Drop the org/ prefix, slashes aren't allowed in Evaluator names.
    'meta/llama-3.2-1b-instruct' -> 'llama-3.2-1b-instruct'
    'nvidia/llama-3.3-nemotron-super-49b-v1.5' -> 'llama-3.3-nemotron-super-49b-v1.5'
    """
    return base.split("/", 1)[1]


def _build_routes() -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}

    # Adapter targets (14): each LoRA-enabled NIM serves its own LoRA names.
    for corpus_slug, base, rank in _ADAPTERS:
        nim_svc = _NIM_SERVICE_FOR_BASE[base]
        adapter = _adapter_name(corpus_slug, base, rank)
        routes[adapter] = {
            "url":            f"http://{nim_svc}.runai-rag:8000/v1/chat/completions",
            "upstream_model": adapter,
        }

    # Base targets — same upstream NIM as the matching LoRA names, but
    # `upstream_model` is the bare base model id (no LoRA applied at inference).
    # The 49B is registered as a base too (it's not LoRA-modified in Stage 3).
    for base, nim_svc in _NIM_SERVICE_FOR_BASE.items():
        if base == "nvidia/nemotron-3-nano-30b-a3b":
            # Nano base intentionally excluded from Stage 3 base-target sweep
            # (only its LoRA variants are in scope).
            continue
        target_name = _base_target_name(base)
        routes[target_name] = {
            "url":            f"http://{nim_svc}.runai-rag:8000/v1/chat/completions",
            "upstream_model": base,
        }

    return routes


_ROUTES: dict[str, dict[str, Any]] = _build_routes()


# --- <think>...</think> tag scrubbing (universal post-process) ----------

_THINK_BALANCED = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_TAIL = re.compile(r"<think>.*$", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks. No-op on tag-free input."""
    if not text:
        return text
    cleaned = _THINK_BALANCED.sub("", text)
    cleaned = _THINK_OPEN_TAIL.sub("", cleaned)
    return cleaned.strip()


# --- FastAPI app -----------------------------------------------------------

app = FastAPI(title="eval-oai-proxy", description=__doc__)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "routes": len(_ROUTES)}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": now, "owned_by": "eval-oai-proxy"}
            for mid in sorted(_ROUTES)
        ],
    }


def _wrap_as_chat_completion(
    model: str,
    content: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap final response as OAI chat.completion, preserving upstream usage stats."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _wrap_as_text_completion(
    model: str,
    content: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap final response as legacy OAI text_completion (for /v1/completions).

    Evaluator's openai-python client calls client.completions.create() which
    targets /v1/completions and expects choices[0].text — not the chat shape.
    """
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:16]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": content,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _log_request(payload: dict[str, Any]) -> None:
    """Single-line JSON to stdout per request — kubectl logs is the audit trail."""
    print(json.dumps(payload), flush=True, file=sys.stdout)


async def _run_inference(
    model: str,
    messages: list[dict[str, Any]],
    body: dict[str, Any],
    api_kind: str,
) -> tuple[str, dict[str, Any]]:
    """Shared inference path: route by model_id → upstream NIM chat/completions,
    strip <think>, log usage, return (cleaned_content, upstream_usage)."""
    route = _ROUTES.get(model)
    if route is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model!r}. Known: {sorted(_ROUTES)}",
        )

    nim_body: dict[str, Any] = {
        "stream":      False,
        "model":       route["upstream_model"],
        "messages":    messages,
        # 8192 default for Stage 3: reasoning models (Nemotron-Super-49B,
        # Nano MoE) emit hundreds-of-tokens <think> blocks; need budget for
        # reasoning + final answer. Caller (Evaluator) typically overrides
        # via the eval config's params.max_tokens.
        "max_tokens":  body.get("max_tokens", 8192),
        # rag-server / NIM enforce temperature > 0 (greedy-equivalent floor).
        "temperature": max(body.get("temperature") or 0.0001, 0.0001),
        "top_p":       body.get("top_p", 0.95),
    }
    if "stop" in body and body["stop"] is not None:
        nim_body["stop"] = body["stop"]

    timeout = httpx.Timeout(300.0, connect=15.0)
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(route["url"], json=nim_body)
    except httpx.HTTPError as e:
        _log_request({
            "ts": time.time(), "api_kind": api_kind, "model": model,
            "upstream_url": route["url"], "upstream_model": route["upstream_model"],
            "status": "upstream_error", "error": repr(e),
            "latency_ms": int((time.monotonic() - t0) * 1000),
        })
        raise HTTPException(status_code=502, detail=f"upstream error: {e!r}") from e

    latency_ms = int((time.monotonic() - t0) * 1000)
    if resp.status_code != 200:
        _log_request({
            "ts": time.time(), "api_kind": api_kind, "model": model,
            "upstream_url": route["url"], "upstream_model": route["upstream_model"],
            "status": "upstream_http_error", "http_status": resp.status_code,
            "body_snippet": resp.text[:200], "latency_ms": latency_ms,
        })
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"upstream NIM error: {resp.text[:600]!r}",
        )
    data = resp.json()
    try:
        raw = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        _log_request({
            "ts": time.time(), "api_kind": api_kind, "model": model,
            "upstream_url": route["url"], "upstream_model": route["upstream_model"],
            "status": "malformed_response", "error": repr(e),
            "latency_ms": latency_ms,
        })
        raise HTTPException(
            status_code=502,
            detail=f"malformed NIM response: {e!r} body={resp.text[:400]!r}",
        ) from e

    cleaned = strip_think_tags(raw)
    upstream_usage = data.get("usage") or {}
    raw_completion_tokens = int(upstream_usage.get("completion_tokens") or 0)
    # Estimate the post-scrub completion token count by character-ratio scaling
    # (Triton/vLLM gives us raw completion_tokens for the full string including
    # <think>; we don't re-tokenize the cleaned text). Off by a few percent but
    # adequate for ratio analysis of reasoning-vs-answer cost.
    if raw_completion_tokens and len(raw) > 0:
        cleaned_completion_tokens_est = round(raw_completion_tokens * len(cleaned) / len(raw))
    else:
        cleaned_completion_tokens_est = raw_completion_tokens

    _log_request({
        "ts": time.time(),
        "api_kind": api_kind,
        "model": model,
        "upstream_url": route["url"],
        "upstream_model": route["upstream_model"],
        "status": "ok",
        "latency_ms": latency_ms,
        "prompt_tokens": int(upstream_usage.get("prompt_tokens") or 0),
        "completion_tokens_raw": raw_completion_tokens,
        "completion_tokens_cleaned_est": cleaned_completion_tokens_est,
        "raw_chars": len(raw),
        "cleaned_chars": len(cleaned),
        "think_chars_stripped": len(raw) - len(cleaned),
    })

    return cleaned, upstream_usage


@app.post("/v1/chat/completions")
async def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
    model = body.get("model", "")
    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")
    cleaned, usage = await _run_inference(model, messages, body, api_kind="chat")
    return _wrap_as_chat_completion(model, cleaned, usage=usage)


@app.post("/v1/completions")
async def completions(body: dict[str, Any]) -> dict[str, Any]:
    """Legacy OAI completions endpoint. The NeMo Evaluator uses
    `client.completions.create()` which hits this path. We wrap the legacy
    `prompt` (string or list of strings) as a single user-role message,
    delegate to the shared chat-completions inference path, then return in
    the legacy text_completion shape."""
    model = body.get("model", "")
    prompt = body.get("prompt")
    if prompt is None:
        raise HTTPException(status_code=400, detail="prompt is required")
    # Some clients send `prompt` as a list (for n-way batching). We only
    # support n=1, so take the first element if it's a list.
    if isinstance(prompt, list):
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt list is empty")
        prompt = prompt[0]
    if not isinstance(prompt, str):
        raise HTTPException(status_code=400, detail=f"prompt must be string; got {type(prompt).__name__}")

    messages = [{"role": "user", "content": prompt}]
    cleaned, usage = await _run_inference(model, messages, body, api_kind="legacy")
    return _wrap_as_text_completion(model, cleaned, usage=usage)
