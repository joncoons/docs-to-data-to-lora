#!/usr/bin/env python3
"""Durable uncapped batched Stage 1A runner for the 2026-07-09 LE rerun.

This wraps the production batched Stage 1A implementation with per-passage
checkpointing so long NVIDIA endpoint calls can be retried/resumed safely.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

from scripts.pipeline.config import get_external_judge_api_key
from scripts.pipeline.models import KVPRow, Passage
from scripts.pipeline.provenance import entailments_from_kvp_rows
from scripts.pipeline.provenance_io import write_jsonl
from scripts.pipeline.prompts import LE_SYSTEM, LE_USER
from scripts.pipeline.stage1a_batched_kvp import (
    BATCHED_PROMPT_HASH,
    FALLBACK_PROMPT_HASH,
    _call_batched_kvp,
    _chunks,
    _entailment_items,
    _fallback_one_item,
    _rows_from_batched_response,
    parse_le_response,
)

log = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_PRELUDE_RE = re.compile(r"^.*?</think>", re.DOTALL)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def strip_think_blocks(content: str) -> str:
    if not content:
        return content
    content = _THINK_BLOCK_RE.sub("", content)
    content = _THINK_PRELUDE_RE.sub("", content)
    return content.strip()


@dataclass(frozen=True)
class LLMTarget:
    endpoint: str
    model: str
    api_key: str
    max_model_len: int | None = None


class TimeoutLLMClient:
    def __init__(
        self,
        *,
        targets: list[LLMTarget],
        canonical_model: str,
        max_workers: int,
        min_interval_s: float,
        retry_attempts: int,
        retry_base_delay_s: float,
        no_think: bool,
        temperature: float,
        request_timeout_s: float,
    ) -> None:
        if not targets:
            raise ValueError("at least one LLM target is required")
        self.targets = targets
        self.endpoints = [target.endpoint for target in targets]
        # KVP rows keep the canonical model for downstream comparisons; target
        # model IDs are recorded in the manifest because local and external
        # Super endpoints expose exact-equivalent models under different names.
        self.model = canonical_model
        self.no_think = no_think
        self.temperature = temperature
        self.retry_attempts = retry_attempts
        self.retry_base_delay_s = retry_base_delay_s
        self._semaphore = threading.Semaphore(max_workers)
        self._last_call = [0.0]
        self._last_call_lock = threading.Lock()
        self._min_interval = min_interval_s
        self._counter = itertools.count()
        self._clients = [
            OpenAI(base_url=target.endpoint, api_key=target.api_key, timeout=request_timeout_s)
            for target in targets
        ]

    @staticmethod
    def _estimated_prompt_tokens(system: str, user: str) -> int:
        text = system + "\n" + user
        return max(len(text.split()), len(text) // 4)

    @staticmethod
    def _target_fits(target: LLMTarget, prompt_tokens: int, max_tokens: int) -> bool:
        if target.max_model_len is None:
            return True
        return prompt_tokens + max_tokens <= target.max_model_len - 512

    def _next_client(self, system: str, user: str, max_tokens: int) -> tuple[OpenAI, LLMTarget]:
        prompt_tokens = self._estimated_prompt_tokens(system, user)
        start = next(self._counter)
        fallback_index = start % len(self._clients)
        for offset in range(len(self._clients)):
            index = (start + offset) % len(self._clients)
            target = self.targets[index]
            if self._target_fits(target, prompt_tokens, max_tokens):
                return self._clients[index], target
        return self._clients[fallback_index], self.targets[fallback_index]

    @staticmethod
    def _no_think_extra_body(endpoint: str) -> dict:
        if "inference-api.nvidia.com" in endpoint or "integrate.api.nvidia.com" in endpoint:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {"reasoning_effort": "none"}

    def call(self, system: str, user: str, max_tokens: int = 1024) -> Optional[str]:
        with self._semaphore:
            for attempt in range(self.retry_attempts):
                with self._last_call_lock:
                    elapsed = time.time() - self._last_call[0]
                    if elapsed < self._min_interval:
                        time.sleep(self._min_interval - elapsed)
                    client, target = self._next_client(system, user, max_tokens)
                    self._last_call[0] = time.time()
                try:
                    kwargs = {
                        "model": target.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": max_tokens,
                    }
                    if self.no_think:
                        kwargs["extra_body"] = self._no_think_extra_body(target.endpoint)
                    response = client.chat.completions.create(**kwargs)
                    return strip_think_blocks(response.choices[0].message.content or "")
                except Exception as exc:  # noqa: BLE001 - durable generation records failures per passage.
                    log.warning("LLM call attempt %d/%d failed: %s", attempt + 1, self.retry_attempts, exc)
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_base_delay_s * (attempt + 1))
            return None


def read_passages(path: Path) -> list[Passage]:
    return [Passage.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            # le_no_response is usually a transient endpoint/rate-limit failure,
            # so it is intentionally retried on resume.
            if row.get("status") in {"complete", "partial", "no_entailments", "le_parse_failed"}:
                completed.add(row["passage_id"])
    return completed


def read_rows(path: Path) -> list[KVPRow]:
    if not path.exists():
        return []
    return [KVPRow.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def process_passage(
    passage: Passage,
    llm: TimeoutLLMClient,
    *,
    max_premises_per_batch: int,
    batch_parse_attempts: int,
    le_max_tokens: int,
    batched_kvp_max_tokens: int,
) -> tuple[list[KVPRow], dict]:
    started_at = utc_now()
    le_raw = llm.call(LE_SYSTEM, LE_USER.format(text=passage.text), max_tokens=le_max_tokens)
    if not le_raw:
        return [], {
            "passage_id": passage.passage_id,
            "status": "le_no_response",
            "started_at": started_at,
            "finished_at": utc_now(),
            "row_count": 0,
            "entailment_count": 0,
            "premise_count": 0,
            "batched_row_count": 0,
            "fallback_row_count": 0,
            "missing_premise_count": 0,
        }
    le_list = parse_le_response(le_raw)
    if not le_list:
        return [], {
            "passage_id": passage.passage_id,
            "status": "le_parse_failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "row_count": 0,
            "entailment_count": 0,
            "premise_count": 0,
            "batched_row_count": 0,
            "fallback_row_count": 0,
            "missing_premise_count": 0,
            "le_response_chars": len(le_raw),
        }

    items = _entailment_items(le_list.entailments)
    if not items:
        return [], {
            "passage_id": passage.passage_id,
            "status": "no_entailments",
            "started_at": started_at,
            "finished_at": utc_now(),
            "row_count": 0,
            "entailment_count": len(le_list.entailments),
            "premise_count": 0,
            "batched_row_count": 0,
            "fallback_row_count": 0,
            "missing_premise_count": 0,
        }

    rows: list[KVPRow] = []
    missing_premises = 0
    for batch in _chunks(items, max_premises_per_batch):
        response = _call_batched_kvp(
            passage,
            batch,
            llm,
            parse_attempts=batch_parse_attempts,
            max_tokens=batched_kvp_max_tokens,
        )
        produced: set[tuple[int, int]] = set()
        if response is not None:
            batch_rows, produced = _rows_from_batched_response(
                response=response,
                passage=passage,
                items=batch,
                llm=llm,
            )
            rows.extend(batch_rows)

        for item in batch:
            if item.key in produced:
                continue
            fallback_row = _fallback_one_item(
                passage,
                item,
                llm,
                max_tokens=batched_kvp_max_tokens,
            )
            if fallback_row is not None:
                rows.append(fallback_row)
            else:
                missing_premises += 1

    batched_count = sum(1 for row in rows if row.extractor_prompt_hash == BATCHED_PROMPT_HASH)
    fallback_count = sum(1 for row in rows if row.extractor_prompt_hash == FALLBACK_PROMPT_HASH)
    status = "complete" if missing_premises == 0 else "partial"
    return rows, {
        "passage_id": passage.passage_id,
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "row_count": len(rows),
        "entailment_count": len(le_list.entailments),
        "premise_count": len(items),
        "batched_row_count": batched_count,
        "fallback_row_count": fallback_count,
        "missing_premise_count": missing_premises,
    }


def append_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for payload in payloads:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()


def append_rows(path: Path, rows: list[KVPRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(row.model_dump_json() + "\n")
            stream.flush()


def endpoint_api_key(endpoint: str, explicit_api_key: str | None = None) -> str:
    if explicit_api_key is not None:
        return explicit_api_key
    if "inference-api.nvidia.com" in endpoint or "integrate.api.nvidia.com" in endpoint:
        return get_external_judge_api_key()
    return "local"


def parse_targets(args: argparse.Namespace) -> list[LLMTarget]:
    if args.target:
        targets: list[LLMTarget] = []
        for raw in args.target:
            if "=" not in raw:
                raise ValueError("--target must be ENDPOINT=MODEL or ENDPOINT=MODEL@MAX_CONTEXT_TOKENS")
            endpoint, model = raw.split("=", 1)
            endpoint = endpoint.strip()
            model = model.strip()
            max_model_len: int | None = None
            if "@" in model:
                model, raw_max_model_len = model.rsplit("@", 1)
                max_model_len = int(raw_max_model_len)
            elif "inference-api.nvidia.com" not in endpoint and "integrate.api.nvidia.com" not in endpoint:
                max_model_len = 32768
            if not endpoint or not model:
                raise ValueError("--target must include non-empty endpoint and model")
            targets.append(
                LLMTarget(
                    endpoint=endpoint,
                    model=model,
                    api_key=endpoint_api_key(endpoint, args.api_key),
                    max_model_len=max_model_len,
                )
            )
        return targets
    if not args.model:
        raise ValueError("--model is required unless --target is supplied")
    max_model_len = None
    if "inference-api.nvidia.com" not in args.endpoint and "integrate.api.nvidia.com" not in args.endpoint:
        max_model_len = 32768
    return [
        LLMTarget(
            endpoint=args.endpoint,
            model=args.model,
            api_key=endpoint_api_key(args.endpoint, args.api_key),
            max_model_len=max_model_len,
        )
    ]


def manifest_targets(args: argparse.Namespace) -> list[dict[str, str]]:
    return [
        {"endpoint": target.endpoint, "model": target.model, "max_model_len": target.max_model_len}
        for target in parse_targets(args)
    ]


def write_run_manifest(path: Path, args: argparse.Namespace, selected_count: int, completed_count: int) -> None:
    targets = manifest_targets(args)
    payload = {
        "schema_version": "le_uncapped_batch.stage1a_durable.v1",
        "updated_at": utc_now(),
        "input_passages": str(args.input_passages),
        "output_dir": str(args.output_dir),
        "selected_passages": selected_count,
        "completed_before_start": completed_count,
        "endpoint": args.endpoint,
        "model": args.model,
        "targets": targets,
        "temperature": args.temperature,
        "max_workers": args.max_workers,
        "min_request_interval_s": args.min_request_interval_s,
        "retry_attempts": args.retry_attempts,
        "retry_base_delay_s": args.retry_base_delay_s,
        "request_timeout_s": args.request_timeout_s,
        "max_premises_per_batch": args.max_premises_per_batch,
        "batch_parse_attempts": args.batch_parse_attempts,
        "le_max_tokens": args.le_max_tokens,
        "batched_kvp_max_tokens": args.batched_kvp_max_tokens,
        "limits": {
            "max_passages": None,
            "max_entailments": None,
            "max_premises": None,
            "max_premises_per_batch_is_chunk_size_not_drop_cap": True,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-passages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://inference-api.nvidia.com/v1")
    parser.add_argument("--model", help="Canonical extractor model recorded on KVP rows; required unless --target is supplied.")
    parser.add_argument(
        "--target",
        action="append",
        help="Endpoint-specific target as ENDPOINT=MODEL or ENDPOINT=MODEL@MAX_CONTEXT_TOKENS. Repeat to round-robin across exact-equivalent endpoints.",
    )
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--min-request-interval-s", type=float, default=0.5)
    parser.add_argument("--retry-attempts", type=int, default=2)
    parser.add_argument("--retry-base-delay-s", type=float, default=5.0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--max-premises-per-batch", type=int, default=12)
    parser.add_argument("--batch-parse-attempts", type=int, default=2)
    parser.add_argument("--le-max-tokens", type=int, default=16384)
    parser.add_argument("--batched-kvp-max-tokens", type=int, default=16384)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.max_premises_per_batch < 1:
        raise ValueError("--max-premises-per-batch must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "stage1a_le.jsonl"
    results_path = args.output_dir / "stage1a_passage_results.jsonl"
    manifest_path = args.output_dir / "stage1a_batched_durable_manifest.json"
    entailments_path = args.output_dir / "provenance" / "entailments.jsonl"

    passages = read_passages(args.input_passages)
    completed = read_completed(results_path)
    selected = [passage for passage in passages if passage.passage_id not in completed]
    write_run_manifest(manifest_path, args, len(selected), len(completed))
    log.info("Stage 1A durable: %d passages selected, %d already completed", len(selected), len(completed))

    llm = TimeoutLLMClient(
        targets=parse_targets(args),
        canonical_model=args.model or parse_targets(args)[0].model,
        max_workers=args.max_workers,
        min_interval_s=args.min_request_interval_s,
        retry_attempts=args.retry_attempts,
        retry_base_delay_s=args.retry_base_delay_s,
        no_think=True,
        temperature=args.temperature,
        request_timeout_s=args.request_timeout_s,
    )

    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                process_passage,
                passage,
                llm,
                max_premises_per_batch=args.max_premises_per_batch,
                batch_parse_attempts=args.batch_parse_attempts,
                le_max_tokens=args.le_max_tokens,
                batched_kvp_max_tokens=args.batched_kvp_max_tokens,
            ): passage
            for passage in selected
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Stage 1A durable"):
            passage = futures[future]
            try:
                rows, result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep batch durable.
                rows = []
                result = {
                    "passage_id": passage.passage_id,
                    "status": "exception",
                    "finished_at": utc_now(),
                    "row_count": 0,
                    "entailment_count": 0,
                    "premise_count": 0,
                    "batched_row_count": 0,
                    "fallback_row_count": 0,
                    "missing_premise_count": 0,
                    "error": repr(exc),
                }
            with write_lock:
                if rows:
                    append_rows(rows_path, rows)
                append_jsonl(results_path, [result])
                log.info(
                    "passage=%s status=%s rows=%s entailments=%s premises=%s missing=%s",
                    result.get("passage_id"),
                    result.get("status"),
                    result.get("row_count"),
                    result.get("entailment_count"),
                    result.get("premise_count"),
                    result.get("missing_premise_count"),
                )

    all_rows = read_rows(rows_path)
    write_jsonl(entailments_path, entailments_from_kvp_rows(all_rows))
    write_run_manifest(manifest_path, args, 0, len(read_completed(results_path)))
    log.info("Stage 1A durable complete: %d rows -> %s", len(all_rows), rows_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
