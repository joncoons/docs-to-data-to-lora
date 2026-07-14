"""Run a stratified live Stage 1A batched-KVP experiment.

This script is intentionally isolated from the production pipeline. It samples
passages from existing Stage 0 artifacts, sends the full selected passage text to
Nemotron 3 Ultra through a configured OpenAI-compatible inference endpoint, writes batched Stage 1A
artifacts, and records timing/call metrics that support baseline latency
interpolation.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import logging
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scripts.pipeline.models import KVPRow, Passage  # noqa: E402
from scripts.pipeline.prompts import KVP_SYSTEM, KVP_USER, LE_SYSTEM  # noqa: E402
from scripts.pipeline.provenance import entailments_from_kvp_rows  # noqa: E402
from scripts.pipeline.provenance_io import write_jsonl  # noqa: E402

LOG = logging.getLogger(__name__)

DEFAULT_INPUTS = {
    "nim_curated": Path("<DATASET_ROOT>/nim_curated/passages.jsonl"),
    "nemo_usvcs_curated": Path("<DATASET_ROOT>/nemo_usvcs_curated/passages.jsonl"),
}
DEFAULT_MODEL = "nvidia/nvidia/nemotron-3-ultra"
DEFAULT_BASE_URL = "http://llm-frontier.default.svc.cluster.local:8000/v1"
DEFAULT_TOKEN_CAP = 16_384


@dataclass(frozen=True)
class SampleRecord:
    collection: str
    passage_id: str
    source_url: str
    token_count: int
    token_bin: str
    doc_kind: str
    product_family: str
    product_name: str


class InstrumentedLLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        retry_attempts: int,
        retry_base_delay_s: float,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.retry_attempts = retry_attempts
        self.retry_base_delay_s = retry_base_delay_s
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.events: list[dict[str, Any]] = []

    def _phase_for(self, system: str) -> str:
        if system == LE_SYSTEM:
            return "le"
        if system == KVP_SYSTEM:
            return "fallback_kvp"
        return "batched_kvp"

    def call(self, system: str, user: str, max_tokens: int = 1024) -> str | None:
        return self.call_with_phase(
            self._phase_for(system),
            system,
            user,
            max_tokens=max_tokens,
        )

    def call_with_phase(
        self,
        phase: str,
        system: str,
        user: str,
        *,
        max_tokens: int,
    ) -> str | None:
        last_error: str | None = None
        for attempt in range(self.retry_attempts):
            started = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                elapsed = time.perf_counter() - started
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                self.events.append({
                    "phase": phase,
                    "attempt": attempt + 1,
                    "ok": True,
                    "elapsed_s": round(elapsed, 4),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "content_chars": len(content),
                })
                return content.strip()
            except Exception as exc:  # noqa: BLE001 - experiment runner records provider errors.
                elapsed = time.perf_counter() - started
                last_error = f"{type(exc).__name__}: {exc}"
                self.events.append({
                    "phase": phase,
                    "attempt": attempt + 1,
                    "ok": False,
                    "elapsed_s": round(elapsed, 4),
                    "error": last_error[:500],
                })
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_base_delay_s * (attempt + 1))
        LOG.warning("LLM call failed after retries: %s", last_error)
        return None


def _load_batched_module() -> Any:
    module_path = Path(__file__).resolve().parent / "stage1a_batched_kvp.py"
    spec = importlib.util.spec_from_file_location("stage1a_batched_kvp_refactor", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_api_key(secret_name: str, namespace: str, key_name: str) -> str:
    raw = subprocess.check_output([
        "kubectl",
        "get",
        "secret",
        secret_name,
        "-n",
        namespace,
        "-o",
        f"jsonpath={{.data.{key_name}}}",
    ]).strip()
    value = base64.b64decode(raw).decode().strip()
    if not value:
        raise RuntimeError(f"K8s secret {namespace}/{secret_name}:{key_name} is empty")
    return value


def _token_bin(token_count: int) -> str:
    if token_count <= 512:
        return "short_00000_00512"
    if token_count <= 2048:
        return "medium_00513_02048"
    if token_count <= 8192:
        return "long_02049_08192"
    if token_count <= DEFAULT_TOKEN_CAP:
        return "xlong_08193_16384"
    return "over_cap"


def _stable_order(seed: str, passage: Passage) -> str:
    digest = hashlib.sha256(f"{seed}\0{passage.passage_id}".encode()).hexdigest()
    return digest


def _load_passages(path: Path) -> list[Passage]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [Passage.model_validate_json(line) for line in path.read_text().splitlines() if line]


def select_stratified_passages(
    inputs: dict[str, Path],
    *,
    per_collection: int,
    token_cap: int,
    seed: str,
) -> tuple[list[tuple[str, Passage]], dict[str, Any]]:
    selected: list[tuple[str, Passage]] = []
    summary: dict[str, Any] = {"collections": {}, "token_cap": token_cap}

    for collection, path in inputs.items():
        passages = _load_passages(path)
        eligible = [p for p in passages if p.token_count <= token_cap]
        over_cap = [p for p in passages if p.token_count > token_cap]
        groups: dict[tuple[str, str], list[Passage]] = defaultdict(list)
        for passage in eligible:
            groups[(passage.doc_kind, _token_bin(passage.token_count))].append(passage)
        for group in groups.values():
            group.sort(key=lambda p: _stable_order(seed, p))

        ordered_group_keys = sorted(groups)
        pointers = {key: 0 for key in ordered_group_keys}
        collection_selected: list[Passage] = []
        while len(collection_selected) < per_collection and ordered_group_keys:
            progressed = False
            for key in ordered_group_keys:
                pointer = pointers[key]
                group = groups[key]
                if pointer >= len(group):
                    continue
                collection_selected.append(group[pointer])
                pointers[key] = pointer + 1
                progressed = True
                if len(collection_selected) >= per_collection:
                    break
            if not progressed:
                break

        if len(collection_selected) < per_collection:
            raise RuntimeError(
                f"Only selected {len(collection_selected)} passages for {collection}; "
                f"wanted {per_collection}"
            )
        selected.extend((collection, passage) for passage in collection_selected)
        summary["collections"][collection] = {
            "path": str(path),
            "total_passages": len(passages),
            "eligible_passages": len(eligible),
            "over_cap_passages": len(over_cap),
            "over_cap_max_tokens": max((p.token_count for p in over_cap), default=0),
            "selected_passages": len(collection_selected),
            "selected_token_bins": dict(Counter(_token_bin(p.token_count) for p in collection_selected)),
            "selected_doc_kinds": dict(Counter(p.doc_kind for p in collection_selected)),
        }
    return selected, summary


def _sample_record(collection: str, passage: Passage) -> SampleRecord:
    return SampleRecord(
        collection=collection,
        passage_id=passage.passage_id,
        source_url=passage.url,
        token_count=passage.token_count,
        token_bin=_token_bin(passage.token_count),
        doc_kind=passage.doc_kind,
        product_family=passage.product_family,
        product_name=passage.product_name,
    )


def _baseline_calibration_items(rows: list[KVPRow], limit: int, seed: str) -> list[KVPRow]:
    unique: dict[tuple[str, int | None, int | None], KVPRow] = {}
    for row in rows:
        key = (row.passage_id, row.entailment_index, row.premise_index)
        unique.setdefault(key, row)
    items = list(unique.values())
    items.sort(key=lambda row: hashlib.sha256(
        f"{seed}\0{row.passage_id}\0{row.entailment_index}\0{row.premise_index}".encode()
    ).hexdigest())
    return items[:limit]


def _run_baseline_kvp_calibration(
    rows: list[KVPRow],
    *,
    llm: InstrumentedLLMClient,
    limit: int,
    seed: str,
) -> int:
    items = _baseline_calibration_items(rows, limit, seed)
    for row in tqdm(items, desc="Baseline KVP calibration"):
        premises = row.entailment_premises or []
        premise_index = row.premise_index or 0
        if premise_index >= len(premises):
            continue
        llm.call_with_phase(
            "baseline_calibration_kvp",
            KVP_SYSTEM,
            KVP_USER.format(
                premise=premises[premise_index],
                conclusion=row.entailment_claim or row.answer,
                text=row.context,
            ),
            max_tokens=4096,
        )
    return len(items)


def _usage_sum(events: list[dict[str, Any]], field: str) -> int | None:
    values = [event.get(field) for event in events if event.get(field) is not None]
    if not values:
        return None
    return int(sum(values))


def _metrics(
    *,
    run_id: str,
    model: str,
    base_url: str,
    temperature: float,
    token_cap: int,
    sample_summary: dict[str, Any],
    selected: list[tuple[str, Passage]],
    rows: list[KVPRow],
    events: list[dict[str, Any]],
    total_elapsed_s: float,
    calibration_limit: int,
) -> dict[str, Any]:
    phases = Counter(event["phase"] for event in events)
    ok_events = [event for event in events if event.get("ok")]
    ok_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ok_events:
        ok_by_phase[event["phase"]].append(event)

    premise_keys = {
        (row.passage_id, row.entailment_index, row.premise_index)
        for row in rows
    }
    entailment_ids = {row.entailment_id for row in rows if row.entailment_id}
    fallback_rows = [row for row in rows if row.extractor_prompt_hash != rows[0].extractor_prompt_hash] if rows else []

    le_elapsed = sum(event["elapsed_s"] for event in ok_by_phase.get("le", []))
    calibration_events = ok_by_phase.get("baseline_calibration_kvp", [])
    calibration_avg_s = (
        sum(event["elapsed_s"] for event in calibration_events) / len(calibration_events)
        if calibration_events else None
    )
    interpolated_baseline_s = (
        le_elapsed + len(premise_keys) * calibration_avg_s
        if calibration_avg_s is not None else None
    )
    latency_reduction_pct = (
        (1 - (total_elapsed_s / interpolated_baseline_s)) * 100
        if interpolated_baseline_s and interpolated_baseline_s > 0 else None
    )
    baseline_call_count = len(selected) + len(premise_keys)
    batched_call_count = phases.get("le", 0) + phases.get("batched_kvp", 0) + phases.get("fallback_kvp", 0)

    return {
        "run_id": run_id,
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "token_cap": token_cap,
        "selected_passages": len(selected),
        "selected_tokens_total": sum(p.token_count for _, p in selected),
        "selected_tokens_max": max((p.token_count for _, p in selected), default=0),
        "sample_summary": sample_summary,
        "rows": len(rows),
        "entailments": len(entailment_ids),
        "unique_premises": len(premise_keys),
        "fallback_rows": len(fallback_rows),
        "event_counts": dict(phases),
        "batched_call_count": batched_call_count,
        "baseline_call_count_interpolated": baseline_call_count,
        "request_count_reduction_pct": round(
            (1 - (batched_call_count / baseline_call_count)) * 100,
            2,
        ) if baseline_call_count else None,
        "total_elapsed_s": round(total_elapsed_s, 3),
        "le_elapsed_s": round(le_elapsed, 3),
        "baseline_kvp_calibration_limit": calibration_limit,
        "baseline_kvp_calibration_calls": len(calibration_events),
        "baseline_kvp_calibration_avg_s": round(calibration_avg_s, 3)
        if calibration_avg_s is not None else None,
        "baseline_wall_s_interpolated": round(interpolated_baseline_s, 3)
        if interpolated_baseline_s is not None else None,
        "latency_reduction_pct_interpolated": round(latency_reduction_pct, 2)
        if latency_reduction_pct is not None else None,
        "usage": {
            "prompt_tokens": _usage_sum(ok_events, "prompt_tokens"),
            "completion_tokens": _usage_sum(ok_events, "completion_tokens"),
            "total_tokens": _usage_sum(ok_events, "total_tokens"),
        },
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    ap.add_argument("--per-collection", type=int, default=15)
    ap.add_argument("--token-cap", type=int, default=DEFAULT_TOKEN_CAP)
    ap.add_argument("--seed", default="stage2-efficiency-stratified-v1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--max-premises-per-batch", type=int, default=12)
    ap.add_argument("--batch-parse-attempts", type=int, default=1)
    ap.add_argument("--retry-attempts", type=int, default=2)
    ap.add_argument("--retry-base-delay-s", type=float, default=2.0)
    ap.add_argument("--baseline-calibration-premises", type=int, default=12)
    ap.add_argument("--secret-name", default="llm-api-key")
    ap.add_argument("--secret-namespace", default="runai-rag")
    ap.add_argument("--secret-key", default="api-key")
    ap.add_argument("--nim-passages", type=Path, default=DEFAULT_INPUTS["nim_curated"])
    ap.add_argument("--nemo-passages", type=Path, default=DEFAULT_INPUTS["nemo_usvcs_curated"])
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_id = args.run_id or datetime.now(UTC).strftime("stratified-ultra-550b-%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    batch_dir = run_dir / "conservative_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "nim_curated": args.nim_passages,
        "nemo_usvcs_curated": args.nemo_passages,
    }
    selected, sample_summary = select_stratified_passages(
        inputs,
        per_collection=args.per_collection,
        token_cap=args.token_cap,
        seed=args.seed,
    )
    (run_dir / "sample_manifest.jsonl").write_text("".join(
        json.dumps(asdict(_sample_record(collection, passage)), sort_keys=True) + "\n"
        for collection, passage in selected
    ))
    (run_dir / "input_passage_ids.txt").write_text("".join(
        passage.passage_id + "\n" for _, passage in selected
    ))
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_id": run_id,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "token_cap": args.token_cap,
        "per_collection": args.per_collection,
        "seed": args.seed,
        "max_premises_per_batch": args.max_premises_per_batch,
        "batch_parse_attempts": args.batch_parse_attempts,
        "baseline_calibration_premises": args.baseline_calibration_premises,
        "inputs": {name: str(path) for name, path in inputs.items()},
    }, indent=2, sort_keys=True) + "\n")

    api_key = _read_api_key(args.secret_name, args.secret_namespace, args.secret_key)
    llm = InstrumentedLLMClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        retry_attempts=args.retry_attempts,
        retry_base_delay_s=args.retry_base_delay_s,
    )
    batched = _load_batched_module()

    rows: list[KVPRow] = []
    per_passage: list[dict[str, Any]] = []
    started = time.perf_counter()
    for collection, passage in tqdm(selected, desc="Batched Stage 1A stratified"):
        before_events = len(llm.events)
        before_rows = len(rows)
        passage_started = time.perf_counter()
        new_rows = batched.process_passage_1a_batched(
            passage,
            llm,
            max_premises_per_batch=args.max_premises_per_batch,
            batch_parse_attempts=args.batch_parse_attempts,
        )
        rows.extend(new_rows)
        passage_events = llm.events[before_events:]
        per_passage.append({
            "collection": collection,
            "passage_id": passage.passage_id,
            "token_count": passage.token_count,
            "token_bin": _token_bin(passage.token_count),
            "doc_kind": passage.doc_kind,
            "elapsed_s": round(time.perf_counter() - passage_started, 4),
            "rows": len(rows) - before_rows,
            "calls": len(passage_events),
            "phases": dict(Counter(event["phase"] for event in passage_events)),
        })

    calibration_count = _run_baseline_kvp_calibration(
        rows,
        llm=llm,
        limit=args.baseline_calibration_premises,
        seed=args.seed,
    )
    total_elapsed_s = time.perf_counter() - started

    out_file = batch_dir / "stage1a_le.batched_kvp.jsonl"
    with out_file.open("w") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    write_jsonl(batch_dir / "provenance" / "entailments.batched_kvp.jsonl", entailments_from_kvp_rows(rows))
    (run_dir / "call_events.jsonl").write_text("".join(
        json.dumps(event, sort_keys=True) + "\n" for event in llm.events
    ))
    (run_dir / "per_passage_metrics.jsonl").write_text("".join(
        json.dumps(item, sort_keys=True) + "\n" for item in per_passage
    ))
    metrics = _metrics(
        run_id=run_id,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        token_cap=args.token_cap,
        sample_summary=sample_summary,
        selected=selected,
        rows=rows,
        events=llm.events,
        total_elapsed_s=total_elapsed_s,
        calibration_limit=calibration_count,
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
