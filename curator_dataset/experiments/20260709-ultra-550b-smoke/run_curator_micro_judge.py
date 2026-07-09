#!/usr/bin/env python3
"""Judge Curator Super-vs-Ultra micro QA rows with Claude via NVIDIA endpoint."""
from __future__ import annotations

import argparse
import base64
import json
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
DEFAULT_MODEL = "azure/anthropic/claude-sonnet-4-6"
DEFAULT_ENDPOINT = "https://inference-api.nvidia.com/v1"
DEFAULT_SECRET_NAME = "nvidia-inference-key"
DEFAULT_SECRET_NAMESPACE = "runai-rag"
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

JUDGE_SYSTEM = (
    "You are an independent data-quality judge for NVIDIA technical QA SFT data. "
    "Compare candidate training examples only against the supplied source context. "
    "Return JSON only."
)

JUDGE_USER = """\
Two candidate QA training examples were generated from the same source segment.
Choose the better training example for a technical assistant.

Source context:
{context}

Candidate A question:
{question_a}

Candidate A answer:
{answer_a}

Candidate B question:
{question_b}

Candidate B answer:
{answer_b}

Judge using this priority order:
- answer is grounded in the source context
- answer is factually correct and specific
- question is useful, non-trivial, and answerable from the context
- answer is concise without losing necessary technical detail
- example does not rely on hidden metadata, generation artifacts, or unsupported assumptions

Return JSON only with this shape:
{{
  "winner": "A" | "B" | "TIE",
  "score_a": 1-5,
  "score_b": 1-5,
  "groundedness_a": 1-5,
  "groundedness_b": 1-5,
  "specificity_a": 1-5,
  "specificity_b": 1-5,
  "usefulness_a": 1-5,
  "usefulness_b": 1-5,
  "reason": "one short sentence"
}}
"""


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


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: float) -> JsonResponse:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            return JsonResponse(response.status, dict(response.headers), body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return JsonResponse(exc.code, dict(exc.headers), body)


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def chat_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def require_experiment_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.relative_to(EXPERIMENT_ROOT)
    return resolved


def read_api_key(secret_name: str, namespace: str, key: str) -> str:
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "secret",
            secret_name,
            "-n",
            namespace,
            "-o",
            f"jsonpath={{.data.{key}}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return base64.b64decode(result.stdout.strip()).decode().strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line_number"] = line_number
            rows.append(row)
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    lineage = row.get("lineage") or {}
    return (
        str(lineage.get("collection") or ""),
        str(lineage.get("document_id") or ""),
        str(lineage.get("segment_id") or ""),
        int(lineage.get("pair_index") or 0),
    )


def selection_key(seed: str, key: tuple[str, str, str, int]) -> str:
    material = "\n".join([seed, *map(str, key)])
    import hashlib

    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_pairs(corpus: str, super_path: Path, ultra_path: Path, limit: int, seed: str) -> list[dict[str, Any]]:
    super_rows = {row_key(row): row for row in read_jsonl(super_path)}
    ultra_rows = {row_key(row): row for row in read_jsonl(ultra_path)}
    keys = [key for key in sorted(set(super_rows) & set(ultra_rows), key=lambda k: selection_key(seed, k)) if key[0] == corpus]
    pairs = []
    for ordinal, key in enumerate(keys[:limit], start=1):
        pairs.append(
            {
                "pair_ordinal": ordinal,
                "pair_key": {
                    "collection": key[0],
                    "document_id": key[1],
                    "segment_id": key[2],
                    "pair_index": key[3],
                },
                "super": super_rows[key],
                "ultra": ultra_rows[key],
            }
        )
    return pairs


def strip_code_fence(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


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


def coerce_score(value: Any) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= score <= 5:
        return score
    return None


def coerce_payload(payload: dict[str, Any]) -> dict[str, Any]:
    winner = str(payload.get("winner") or "TIE").strip().upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    return {
        "winner": winner,
        "score_a": coerce_score(payload.get("score_a")),
        "score_b": coerce_score(payload.get("score_b")),
        "groundedness_a": coerce_score(payload.get("groundedness_a")),
        "groundedness_b": coerce_score(payload.get("groundedness_b")),
        "specificity_a": coerce_score(payload.get("specificity_a")),
        "specificity_b": coerce_score(payload.get("specificity_b")),
        "usefulness_a": coerce_score(payload.get("usefulness_a")),
        "usefulness_b": coerce_score(payload.get("usefulness_b")),
        "reason": str(payload.get("reason") or "").strip(),
    }


def retry_delay_s(config: JudgeConfig, attempt: int, response: Any | None = None) -> float:
    retry_after = None
    if response is not None:
        try:
            value = response.headers.get("retry-after")
            retry_after = float(value) if value else None
        except (TypeError, ValueError):
            retry_after = None
    delay = min(config.retry_backoff_s * (2**attempt), config.retry_backoff_max_s)
    if retry_after is not None:
        delay = max(delay, min(retry_after, config.retry_backoff_max_s))
    return delay + (random.uniform(0, config.retry_jitter_s) if config.retry_jitter_s else 0)


def map_winner(winner: str, swapped: bool) -> str:
    if winner == "TIE":
        return "tie"
    if not swapped:
        return "super" if winner == "A" else "ultra"
    return "ultra" if winner == "A" else "super"


def score_for_side(parsed: dict[str, Any], side: str, swapped: bool, metric: str) -> int | None:
    side_is_a = (side == "super" and not swapped) or (side == "ultra" and swapped)
    suffix = "a" if side_is_a else "b"
    return parsed.get(f"{metric}_{suffix}")


def judge_once(config: JudgeConfig, pair: dict[str, Any], *, swapped: bool) -> dict[str, Any]:
    left = pair["ultra"] if swapped else pair["super"]
    right = pair["super"] if swapped else pair["ultra"]
    user = JUDGE_USER.format(
        context=str(pair["super"].get("context") or pair["ultra"].get("context") or ""),
        question_a=str(left.get("prompt") or ""),
        answer_a=str(left.get("completion") or ""),
        question_b=str(right.get("prompt") or ""),
        answer_b=str(right.get("completion") or ""),
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
                time.sleep(retry_delay_s(config, attempt, response))
                continue
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            raw = choice.get("message", {}).get("content") or choice.get("text") or ""
            parsed = coerce_payload(parse_json_object(raw))
            return {
                "position": "ultra_as_a" if swapped else "super_as_a",
                "mapped_winner": map_winner(parsed["winner"], swapped),
                "raw_winner": parsed["winner"],
                "scores": {
                    "super": {
                        "overall": score_for_side(parsed, "super", swapped, "score"),
                        "groundedness": score_for_side(parsed, "super", swapped, "groundedness"),
                        "specificity": score_for_side(parsed, "super", swapped, "specificity"),
                        "usefulness": score_for_side(parsed, "super", swapped, "usefulness"),
                    },
                    "ultra": {
                        "overall": score_for_side(parsed, "ultra", swapped, "score"),
                        "groundedness": score_for_side(parsed, "ultra", swapped, "groundedness"),
                        "specificity": score_for_side(parsed, "ultra", swapped, "specificity"),
                        "usefulness": score_for_side(parsed, "ultra", swapped, "usefulness"),
                    },
                },
                "reason": parsed["reason"],
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
                time.sleep(retry_delay_s(config, attempt))
                continue
            break
    assert last_error is not None
    raise last_error


def resolve_winner(winners: list[str]) -> tuple[str, str]:
    if not winners:
        return "tie", "empty"
    unique = set(winners)
    if len(unique) == 1:
        return winners[0], "agree"
    non_ties = [winner for winner in winners if winner != "tie"]
    if len(set(non_ties)) == 1:
        return non_ties[0], "weak"
    return "tie", "conflict"


def avg_score(outcomes: list[dict[str, Any]], side: str, metric: str) -> float | None:
    values = [outcome.get("scores", {}).get(side, {}).get(metric) for outcome in outcomes]
    values = [value for value in values if isinstance(value, int)]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def judge_pair(config: JudgeConfig, pair: dict[str, Any], position_swap: bool) -> dict[str, Any]:
    outcomes = [judge_once(config, pair, swapped=False)]
    if position_swap:
        outcomes.append(judge_once(config, pair, swapped=True))
    winner, agreement = resolve_winner([outcome["mapped_winner"] for outcome in outcomes])
    return {
        "schema_version": "curator_micro_pairwise_judge.row.v1",
        "pair_ordinal": pair["pair_ordinal"],
        "pair_key": pair["pair_key"],
        "winner": winner,
        "agreement": agreement,
        "super_sample_id": pair["super"].get("sample_id"),
        "ultra_sample_id": pair["ultra"].get("sample_id"),
        "super_question": pair["super"].get("prompt"),
        "ultra_question": pair["ultra"].get("prompt"),
        "super_answer": pair["super"].get("completion"),
        "ultra_answer": pair["ultra"].get("completion"),
        "avg_scores": {
            "super": {metric: avg_score(outcomes, "super", metric) for metric in ["overall", "groundedness", "specificity", "usefulness"]},
            "ultra": {metric: avg_score(outcomes, "ultra", metric) for metric in ["overall", "groundedness", "specificity", "usefulness"]},
        },
        "outcomes": outcomes,
    }


def summarize(rows: list[dict[str, Any]], config: JudgeConfig, args: argparse.Namespace) -> dict[str, Any]:
    wins = {"super": 0, "ultra": 0, "tie": 0}
    agreements: dict[str, int] = {}
    corpora: dict[str, dict[str, int]] = {}
    score_totals: dict[str, dict[str, list[float]]] = {
        "super": {metric: [] for metric in ["overall", "groundedness", "specificity", "usefulness"]},
        "ultra": {metric: [] for metric in ["overall", "groundedness", "specificity", "usefulness"]},
    }
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        wins[row["winner"]] = wins.get(row["winner"], 0) + 1
        agreements[row["agreement"]] = agreements.get(row["agreement"], 0) + 1
        corpus = row["pair_key"]["collection"]
        corpora.setdefault(corpus, {"super": 0, "ultra": 0, "tie": 0})
        corpora[corpus][row["winner"]] = corpora[corpus].get(row["winner"], 0) + 1
        for side in ["super", "ultra"]:
            for metric, value in row.get("avg_scores", {}).get(side, {}).items():
                if isinstance(value, (int, float)):
                    score_totals[side][metric].append(float(value))
        for outcome in row.get("outcomes", []):
            counts = outcome.get("judge", {}).get("usage", {})
            for key in usage:
                usage[key] += int(counts.get(key) or 0)
    avg_scores = {
        side: {
            metric: round(sum(values) / len(values), 3) if values else None
            for metric, values in metrics.items()
        }
        for side, metrics in score_totals.items()
    }
    return {
        "schema_version": "curator_micro_pairwise_judge.summary.v1",
        "created_at": utc_now(),
        "judge": {
            "endpoint": config.endpoint,
            "model": config.model,
            "api_key_source": f"Kubernetes Secret {args.secret_namespace}/{args.secret_name}:{args.secret_key}; value not persisted",
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "position_swap": args.position_swap,
        },
        "selection": {
            "seed": args.seed,
            "pairs_per_corpus": args.pairs_per_corpus,
            "corpora": args.corpus,
        },
        "rows_judged": len(rows),
        "wins": wins,
        "agreements": agreements,
        "wins_by_corpus": corpora,
        "avg_scores": avg_scores,
        "score_delta_ultra_minus_super": {
            metric: (
                round((avg_scores["ultra"].get(metric) or 0) - (avg_scores["super"].get(metric) or 0), 3)
                if avg_scores["ultra"].get(metric) is not None and avg_scores["super"].get(metric) is not None
                else None
            )
            for metric in ["overall", "groundedness", "specificity", "usefulness"]
        },
        "token_usage": usage,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Curator Micro Claude Judge",
        "",
        f"Created: {summary['created_at']}",
        "",
        f"Judge model: `{summary['judge']['model']}` via `{summary['judge']['endpoint']}`.",
        "",
        f"Rows judged: {summary['rows_judged']} paired Super-vs-Ultra examples with position swap={summary['judge']['position_swap']}.",
        "",
        "## Wins",
        "",
        "| Side | Wins |",
        "| --- | ---: |",
    ]
    for side in ["super", "ultra", "tie"]:
        lines.append(f"| {side} | {summary['wins'].get(side, 0)} |")
    lines.extend(["", "## Wins By Corpus", "", "| Corpus | Super | Ultra | Tie |", "| --- | ---: | ---: | ---: |"])
    for corpus, wins in sorted(summary["wins_by_corpus"].items()):
        lines.append(f"| {corpus} | {wins.get('super', 0)} | {wins.get('ultra', 0)} | {wins.get('tie', 0)} |")
    lines.extend(["", "## Average Scores", "", "| Metric | Super | Ultra | Delta |", "| --- | ---: | ---: | ---: |"])
    for metric in ["overall", "groundedness", "specificity", "usefulness"]:
        s = summary["avg_scores"]["super"].get(metric)
        u = summary["avg_scores"]["ultra"].get(metric)
        d = summary["score_delta_ultra_minus_super"].get(metric)
        lines.append(f"| {metric} | {s} | {u} | {d} |")
    lines.extend(["", "## Interpretation", ""])
    ultra = summary["wins"].get("ultra", 0)
    super_ = summary["wins"].get("super", 0)
    if ultra > super_:
        lines.append("Claude preferred Ultra on this small paired sample, so a proxy LoRA r16 training edge is reasonable before any full-corpus regeneration.")
    elif super_ > ultra:
        lines.append("Claude preferred Super on this small paired sample, so Ultra does not currently justify proxy LoRA training without a larger or more targeted sample.")
    else:
        lines.append("Claude did not separate Super and Ultra on this small paired sample; collect a larger judge sample or stop the Ultra path before training.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--secret-namespace", default=DEFAULT_SECRET_NAMESPACE)
    parser.add_argument("--secret-key", default="api-key")
    parser.add_argument("--pairs-per-corpus", type=int, default=8)
    parser.add_argument("--corpus", action="append", choices=["nim_curated", "nemo_usvcs_curated"], default=[])
    parser.add_argument("--seed", default="20260709-ultra-550b-curator-micro-judge-v1")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-backoff-s", type=float, default=4.0)
    parser.add_argument("--retry-backoff-max-s", type=float, default=60.0)
    parser.add_argument("--retry-jitter-s", type=float, default=1.0)
    parser.add_argument("--position-swap", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.corpus:
        args.corpus = ["nim_curated", "nemo_usvcs_curated"]
    output_dir = args.output_dir or EXPERIMENT_ROOT / "judge_runs" / f"curator_micro_claude-sonnet-4-6-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = require_experiment_path(output_dir)
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
    pairs: list[dict[str, Any]] = []
    for corpus in args.corpus:
        pairs.extend(
            build_pairs(
                corpus,
                EXPERIMENT_ROOT / "runs" / "curator_micro" / corpus / "super-v3" / "normalized" / "sft.jsonl",
                EXPERIMENT_ROOT / "runs" / "curator_micro" / corpus / "ultra-550b" / "normalized" / "sft.jsonl",
                args.pairs_per_corpus,
                args.seed,
            )
        )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for pair in pairs:
        try:
            row = judge_pair(config, pair, args.position_swap)
            rows.append(row)
            write_jsonl(output_dir / "judgments.jsonl", rows)
            print(json.dumps({"pair_ordinal": row["pair_ordinal"], "corpus": row["pair_key"]["collection"], "winner": row["winner"], "agreement": row["agreement"]}))
        except Exception as exc:  # noqa: BLE001
            error = {"pair": pair["pair_key"], "error": repr(exc), "created_at": utc_now()}
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
