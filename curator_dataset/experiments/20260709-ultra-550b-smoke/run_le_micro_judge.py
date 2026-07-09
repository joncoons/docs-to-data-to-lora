#!/usr/bin/env python3
"""Score LE micro QA rows with Claude via NVIDIA endpoint."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENDPOINT = "https://inference-api.nvidia.com/v1"
DEFAULT_MODEL = "azure/anthropic/claude-sonnet-4-6"
DEFAULT_SECRET_NAME = "nvidia-inference-key"
DEFAULT_SECRET_NAMESPACE = "runai-rag"
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
CORPORA = ("nim_curated", "nemo_usvcs_curated")
CELLS = {"super": "super-v3", "ultra": "ultra-550b"}

JUDGE_SYSTEM = (
    "You are an independent quality judge for NVIDIA technical QA SFT data. "
    "Score the candidate example only against the supplied source context. Return JSON only."
)

JUDGE_USER = """\
Score this candidate QA training example for a technical assistant.

Source context:
{context}

Entailment claim:
{claim}

Entailment premises:
{premises}

Question:
{question}

Answer:
{answer}

Score 1-5 for each metric:
- groundedness: answer is supported by the source context
- correctness: answer accurately resolves the question
- specificity: answer includes useful technical specifics when available
- usefulness: question and answer would be useful SFT data for NVIDIA technical support
- leakage: 5 means no hidden metadata/artifact dependence; 1 means serious leakage

Return JSON only with this shape:
{{
  "overall": 1-5,
  "groundedness": 1-5,
  "correctness": 1-5,
  "specificity": 1-5,
  "usefulness": 1-5,
  "leakage": 1-5,
  "decision": "pass" | "borderline" | "fail",
  "reason": "one short sentence"
}}
"""


@dataclass(frozen=True)
class JudgeConfig:
    endpoint: str
    model: str
    api_key: str
    max_tokens: int
    temperature: float
    timeout_s: float
    max_attempts: int
    retry_backoff_s: float
    retry_backoff_max_s: float
    retry_jitter_s: float


class JsonResponse:
    def __init__(self, status_code: int, headers: dict[str, str], body: str) -> None:
        self.status_code = status_code
        self.headers = headers
        self._body = body

    def json(self) -> dict[str, Any]:
        parsed = json.loads(self._body)
        if not isinstance(parsed, dict):
            raise ValueError("endpoint response was not a JSON object")
        return parsed

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            excerpt = self._body[:500].replace("\n", " ")
            raise RuntimeError(f"HTTP {self.status_code}: {excerpt}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def chat_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: float) -> JsonResponse:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return JsonResponse(response.status, dict(response.headers), response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return JsonResponse(exc.code, dict(exc.headers), exc.read().decode("utf-8", errors="replace"))


def read_api_key(secret_name: str, namespace: str, key: str) -> str:
    result = subprocess.run(
        ["kubectl", "get", "secret", secret_name, "-n", namespace, "-o", f"jsonpath={{.data.{key}}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return base64.b64decode(result.stdout.strip()).decode().strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line_number"] = line_number
            rows.append(row)
    return rows


def stable_key(seed: str, corpus: str, label: str, row: dict[str, Any]) -> str:
    material = "\n".join([
        seed,
        corpus,
        label,
        str(row.get("passage_id") or ""),
        str(row.get("entailment_index") or 0),
        str(row.get("premise_index") or 0),
        str(row.get("question") or ""),
        str(row.get("answer") or ""),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def select_rows(seed: str, rows_per_cell: int) -> list[dict[str, Any]]:
    selected = []
    for corpus in CORPORA:
        for label, directory in CELLS.items():
            path = EXPERIMENT_ROOT / "runs" / "le_micro" / corpus / directory / "stage1a_le.jsonl"
            rows = read_jsonl(path)
            rows.sort(key=lambda row: stable_key(seed, corpus, label, row))
            for ordinal, row in enumerate(rows[:rows_per_cell], start=1):
                selected.append({"corpus": corpus, "label": label, "cell_ordinal": ordinal, "row": row})
    return selected


def strip_code_fence(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", cleaned).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(parsed, dict):
        raise ValueError("judge response was not a JSON object")
    return parsed


def coerce_score(value: Any) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def coerce_payload(payload: dict[str, Any]) -> dict[str, Any]:
    decision = str(payload.get("decision") or "borderline").strip().lower()
    if decision not in {"pass", "borderline", "fail"}:
        decision = "borderline"
    return {
        "overall": coerce_score(payload.get("overall")),
        "groundedness": coerce_score(payload.get("groundedness")),
        "correctness": coerce_score(payload.get("correctness")),
        "specificity": coerce_score(payload.get("specificity")),
        "usefulness": coerce_score(payload.get("usefulness")),
        "leakage": coerce_score(payload.get("leakage")),
        "decision": decision,
        "reason": str(payload.get("reason") or "").strip(),
    }


def normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    def to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
    prompt = to_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion = to_int(usage.get("completion_tokens") or usage.get("output_tokens"))
    total = to_int(usage.get("total_tokens")) or prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def retry_delay(config: JudgeConfig, attempt: int, response: JsonResponse | None = None) -> float:
    retry_after = None
    if response is not None:
        try:
            raw = response.headers.get("retry-after")
            retry_after = float(raw) if raw else None
        except (TypeError, ValueError):
            retry_after = None
    delay = min(config.retry_backoff_s * (2 ** attempt), config.retry_backoff_max_s)
    if retry_after is not None:
        delay = max(delay, min(retry_after, config.retry_backoff_max_s))
    return delay + (random.uniform(0, config.retry_jitter_s) if config.retry_jitter_s else 0)


def judge_row(config: JudgeConfig, item: dict[str, Any]) -> dict[str, Any]:
    row = item["row"]
    user = JUDGE_USER.format(
        context=str(row.get("context") or ""),
        claim=str(row.get("entailment_claim") or ""),
        premises=json.dumps(row.get("entailment_premises") or [], ensure_ascii=False),
        question=str(row.get("question") or ""),
        answer=str(row.get("answer") or ""),
    )
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(config.max_attempts):
        started = time.perf_counter()
        try:
            response = post_json(chat_url(config.endpoint), headers, payload, config.timeout_s)
            latency = time.perf_counter() - started
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < config.max_attempts - 1:
                time.sleep(retry_delay(config, attempt, response))
                continue
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            raw = choice.get("message", {}).get("content") or choice.get("text") or ""
            parsed = coerce_payload(parse_json_object(raw))
            return {
                "schema_version": "le_micro_judge.row.v1",
                "created_at": utc_now(),
                "corpus": item["corpus"],
                "label": item["label"],
                "cell_ordinal": item["cell_ordinal"],
                "source_line_number": row.get("_source_line_number"),
                "passage_id": row.get("passage_id"),
                "entailment_index": row.get("entailment_index"),
                "premise_index": row.get("premise_index"),
                "extractor_model": row.get("extractor_model"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "scores": parsed,
                "judge": {
                    "endpoint": config.endpoint,
                    "model": config.model,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": normalize_usage(body.get("usage") or {}),
                    "latency_s": round(latency, 3),
                    "attempts": attempt + 1,
                    "raw_response_chars": len(raw),
                },
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < config.max_attempts - 1:
                time.sleep(retry_delay(config, attempt))
                continue
            break
    assert last_error is not None
    raise last_error


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(rows: list[dict[str, Any]], config: JudgeConfig, args: argparse.Namespace) -> dict[str, Any]:
    metrics = ["overall", "groundedness", "correctness", "specificity", "usefulness", "leakage"]
    by_cell: dict[str, dict[str, Any]] = {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["corpus"], row["label"])].append(row)
        for key in usage:
            usage[key] += int(row.get("judge", {}).get("usage", {}).get(key) or 0)
    for (corpus, label), group in sorted(grouped.items()):
        key = f"{corpus}/{label}"
        by_cell[key] = {
            "rows_judged": len(group),
            "decisions": {decision: sum(1 for row in group if row["scores"].get("decision") == decision) for decision in ["pass", "borderline", "fail"]},
            "avg_scores": {},
        }
        for metric in metrics:
            values = [row["scores"].get(metric) for row in group if isinstance(row["scores"].get(metric), int)]
            by_cell[key]["avg_scores"][metric] = round(sum(values) / len(values), 3) if values else None
    deltas = {}
    for corpus in CORPORA:
        s = by_cell.get(f"{corpus}/super", {}).get("avg_scores", {})
        u = by_cell.get(f"{corpus}/ultra", {}).get("avg_scores", {})
        deltas[corpus] = {
            metric: round((u.get(metric) or 0) - (s.get(metric) or 0), 3)
            if u.get(metric) is not None and s.get(metric) is not None else None
            for metric in metrics
        }
    return {
        "schema_version": "le_micro_judge.summary.v1",
        "created_at": utc_now(),
        "judge": {
            "endpoint": config.endpoint,
            "model": config.model,
            "api_key_source": f"Kubernetes Secret {args.secret_namespace}/{args.secret_name}:{args.secret_key}; value not persisted",
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        },
        "selection": {"seed": args.seed, "rows_per_cell": args.rows_per_cell},
        "rows_judged": len(rows),
        "by_cell": by_cell,
        "ultra_minus_super_deltas": deltas,
        "token_usage": usage,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LE Micro Claude Judge",
        "",
        f"Created: {summary['created_at']}",
        "",
        f"Judge model: `{summary['judge']['model']}` via `{summary['judge']['endpoint']}`.",
        "",
        f"Rows judged: {summary['rows_judged']} independent LE QA examples.",
        "",
        "## Average Scores",
        "",
        "| Cell | Rows | Pass | Borderline | Fail | Overall | Groundedness | Correctness | Specificity | Usefulness | Leakage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell, data in sorted(summary["by_cell"].items()):
        avg = data["avg_scores"]
        dec = data["decisions"]
        lines.append(
            f"| {cell} | {data['rows_judged']} | {dec.get('pass', 0)} | {dec.get('borderline', 0)} | {dec.get('fail', 0)} | "
            f"{avg.get('overall')} | {avg.get('groundedness')} | {avg.get('correctness')} | {avg.get('specificity')} | {avg.get('usefulness')} | {avg.get('leakage')} |"
        )
    lines.extend(["", "## Ultra Minus Super Deltas", "", "| Corpus | Overall | Groundedness | Correctness | Specificity | Usefulness | Leakage |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for corpus, deltas in sorted(summary["ultra_minus_super_deltas"].items()):
        lines.append(
            f"| {corpus} | {deltas.get('overall')} | {deltas.get('groundedness')} | {deltas.get('correctness')} | {deltas.get('specificity')} | {deltas.get('usefulness')} | {deltas.get('leakage')} |"
        )
    lines.extend(["", "## Interpretation", ""])
    nim = summary["ultra_minus_super_deltas"].get("nim_curated", {})
    nemo = summary["ultra_minus_super_deltas"].get("nemo_usvcs_curated", {})
    if (nim.get("overall") or 0) >= 0 and (nemo.get("overall") or 0) >= 0:
        lines.append("Claude did not find an aggregate quality regression for Ultra on this sample, so the larger Ultra row yield is worth a small Customizer r16 proxy training edge.")
    else:
        lines.append("Claude found at least one aggregate quality regression for Ultra on this sample; review row-level judgments before running training.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--secret-namespace", default=DEFAULT_SECRET_NAMESPACE)
    parser.add_argument("--secret-key", default="api-key")
    parser.add_argument("--rows-per-cell", type=int, default=4)
    parser.add_argument("--seed", default="20260709-ultra-550b-le-micro-judge-v1")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-backoff-s", type=float, default=4.0)
    parser.add_argument("--retry-backoff-max-s", type=float, default=60.0)
    parser.add_argument("--retry-jitter-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or EXPERIMENT_ROOT / "judge_runs" / f"le_micro_claude-sonnet-4-6-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = output_dir.resolve()
    output_dir.relative_to(EXPERIMENT_ROOT.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)
    config = JudgeConfig(
        endpoint=args.endpoint,
        model=args.model,
        api_key=read_api_key(args.secret_name, args.secret_namespace, args.secret_key),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        max_attempts=max(1, args.max_attempts),
        retry_backoff_s=args.retry_backoff_s,
        retry_backoff_max_s=max(args.retry_backoff_s, args.retry_backoff_max_s),
        retry_jitter_s=max(0.0, args.retry_jitter_s),
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in select_rows(args.seed, args.rows_per_cell):
        try:
            judged = judge_row(config, item)
            rows.append(judged)
            write_jsonl(output_dir / "judgments.jsonl", rows)
            print(json.dumps({"corpus": judged["corpus"], "label": judged["label"], "overall": judged["scores"].get("overall"), "decision": judged["scores"].get("decision")}))
        except Exception as exc:  # noqa: BLE001
            error = {"corpus": item["corpus"], "label": item["label"], "cell_ordinal": item["cell_ordinal"], "error": repr(exc), "created_at": utc_now()}
            errors.append(error)
            write_jsonl(output_dir / "errors.jsonl", errors)
            print(json.dumps({"error": error}), flush=True)
    summary = summarize(rows, config, args)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir / "report.md", summary)
    if errors:
        (output_dir / "errors.summary.json").write_text(json.dumps({"errors": errors}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_dir)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
