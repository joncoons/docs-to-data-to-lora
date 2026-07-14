"""Stage 1C: Instruction diversity pass on top-density passages."""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from typing import Any, Literal

from scripts.pipeline.llm_client import LLMClient
from scripts.pipeline.models import KVPRow, Passage
from scripts.pipeline.provenance import (
    passage_modalities,
    passage_source_chunk_ids,
    passage_source_kinds,
    passage_source_revision_id,
    passage_source_systems,
)
from scripts.pipeline.prompts import INSTRUCTION_SYSTEM, INSTRUCTION_USER

log = logging.getLogger(__name__)


def density_score(passage: Passage) -> float:
    """chunk_index_span × (unique_product_terms / token_count).

    Proxy chunk_index_span = len(chunk_ids).
    Proxy unique_product_terms = product_family + product_name distinct values
    (which is at most 2 here); since this is a flat score across passages, we
    use len(set([product_family, product_name])) for sane comparison.
    """
    span = max(1, len(passage.chunk_ids))
    terms = len({passage.product_family, passage.product_name})
    if passage.token_count <= 0:
        return 0.0
    return span * (terms / passage.token_count) * 1000.0  # scale for readable numbers


SelectionMode = Literal["stratified", "top_density", "all"]


def _target_selection_count(
    passages: list[Passage],
    top_percent: float,
    min_passages: int,
) -> int:
    if not passages:
        return 0
    if len(passages) <= min_passages:
        return len(passages)
    return min(len(passages), max(min_passages, int(len(passages) * top_percent)))


def select_top_density_passages(
    passages: list[Passage], top_percent: float = 0.25, min_passages: int = 100
) -> list[Passage]:
    scored = sorted(passages, key=density_score, reverse=True)
    return scored[:_target_selection_count(passages, top_percent, min_passages)]


def _round_robin_bins(bins: list[list[Passage]], target_count: int) -> list[Passage]:
    selected: list[Passage] = []
    seen: set[str] = set()
    while len(selected) < target_count:
        advanced = False
        for bucket in bins:
            if not bucket:
                continue
            passage = bucket.pop(0)
            if passage.passage_id in seen:
                continue
            selected.append(passage)
            seen.add(passage.passage_id)
            advanced = True
            if len(selected) >= target_count:
                break
        if not advanced:
            break
    return selected


def select_stratified_passages(
    passages: list[Passage],
    top_percent: float = 0.25,
    min_passages: int = 100,
    density_bins: int = 4,
) -> list[Passage]:
    target_count = _target_selection_count(passages, top_percent, min_passages)
    if target_count == len(passages):
        return list(passages)
    ordered = sorted(
        passages,
        key=lambda passage: (density_score(passage), passage.token_count, passage.passage_id),
        reverse=True,
    )
    density_bins = max(1, density_bins)
    buckets: list[list[Passage]] = [[] for _ in range(density_bins)]
    for index, passage in enumerate(ordered):
        buckets[index * density_bins // len(ordered)].append(passage)
    for bucket in buckets:
        bucket.sort(key=lambda passage: (passage.token_count, passage.passage_id), reverse=True)
    return _round_robin_bins(buckets, target_count)


def select_stage1c_passages(
    passages: list[Passage],
    *,
    selection_mode: SelectionMode = "stratified",
    top_percent: float = 0.25,
    min_passages: int = 100,
) -> list[Passage]:
    if selection_mode == "all":
        return list(passages)
    if selection_mode == "top_density":
        return select_top_density_passages(passages, top_percent=top_percent, min_passages=min_passages)
    if selection_mode == "stratified":
        return select_stratified_passages(passages, top_percent=top_percent, min_passages=min_passages)
    raise ValueError(f"unknown Stage 1C selection mode: {selection_mode}")


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()


def parse_instruction_response(raw: str) -> list[dict]:
    s = _strip_fences(raw)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    return obj.get("pairs", [])


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def process_passage_1c_result(passage: Passage, domain: str, llm: LLMClient) -> tuple[str, list[KVPRow], str | None]:
    raw = llm.call(
        INSTRUCTION_SYSTEM,
        INSTRUCTION_USER.format(domain=domain, passage=passage.text),
        max_tokens=4096,
    )
    if not raw:
        return "llm_empty", [], None
    pairs = parse_instruction_response(raw)
    if not pairs:
        return "no_pairs", [], None
    rows: list[KVPRow] = []
    source_revision_id = passage_source_revision_id(passage)
    source_chunk_ids = passage_source_chunk_ids(passage)
    source_systems = passage_source_systems(passage)
    source_kinds = passage_source_kinds(passage)
    modalities = passage_modalities(passage)
    for p in pairs:
        if not p.get("question") or not p.get("answer"):
            continue
        itype = p.get("type", "")
        if itype not in ("summary", "listicle", "procedural"):
            continue
        rows.append(KVPRow(
            passage_id=passage.passage_id,
            source_url=passage.url,
            product_family=passage.product_family,
            stage="1c",
            instr_type=itype,
            question=p["question"].strip(),
            answer=p["answer"].strip(),
            context=passage.text,
            source_revision_ids=[source_revision_id],
            source_chunk_ids=source_chunk_ids,
            source_systems=source_systems or None,
            source_kinds=source_kinds or None,
            modalities=modalities or None,
            refined=False,
        ))
    if not rows:
        return "no_valid_pairs", [], None
    return "complete", rows, None


def process_passage_1c(passage: Passage, domain: str, llm: LLMClient) -> list[KVPRow]:
    _, rows, _ = process_passage_1c_result(passage, domain, llm)
    return rows


def read_stage1c_rows(path: Path) -> list[KVPRow]:
    if not path.exists():
        return []
    return [KVPRow.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def latest_stage1c_statuses(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[str(row["passage_id"])] = row
    return latest


def append_stage1c_rows(stream: Any, rows: list[KVPRow]) -> None:
    for row in rows:
        stream.write(row.model_dump_json() + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def append_stage1c_status(
    stream: Any,
    passage: Passage,
    *,
    status: str,
    row_count: int,
    error: str | None = None,
) -> None:
    payload = {
        "finished_at": utc_now(),
        "passage_id": passage.passage_id,
        "source_url": passage.url,
        "status": status,
        "row_count": row_count,
        "error": error,
    }
    stream.write(json.dumps(payload, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run_stage1c(passages: list[Passage], domain: str, llm: LLMClient,
                output_dir: Path, top_percent: float = 0.25,
                min_passages: int = 100, max_workers: int = 5,
                selection_mode: SelectionMode = "stratified",
                resume: bool = False) -> list[KVPRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_stage1c_passages(
        passages,
        selection_mode=selection_mode,
        top_percent=top_percent,
        min_passages=min_passages,
    )
    log.info(
        "Stage 1C: %d passages selected (mode=%s top %.0f%% floor %d)",
        len(selected),
        selection_mode,
        top_percent * 100,
        min_passages,
    )

    out_file = output_dir / "stage1c_instruction.jsonl"
    status_file = output_dir / "stage1c_passage_results.jsonl"
    if not resume:
        out_file.unlink(missing_ok=True)
        status_file.unlink(missing_ok=True)

    existing_rows = read_stage1c_rows(out_file)
    rows_by_passage = {row.passage_id for row in existing_rows}
    latest_status = latest_stage1c_statuses(status_file)
    terminal_no_row_statuses = {"no_pairs", "no_valid_pairs"}
    pending: list[Passage] = []
    skipped = 0
    for passage in selected:
        status = str(latest_status.get(passage.passage_id, {}).get("status") or "")
        if passage.passage_id in rows_by_passage or status in terminal_no_row_statuses:
            skipped += 1
            continue
        pending.append(passage)

    if skipped:
        log.info("Stage 1C: resuming with %d skipped passages and %d pending", skipped, len(pending))
    all_rows: list[KVPRow] = list(existing_rows)
    if not pending:
        log.info("Stage 1C: %d KVPs already available at %s", len(all_rows), out_file)
        return all_rows

    with out_file.open("a", encoding="utf-8") as row_stream, status_file.open("a", encoding="utf-8") as status_stream:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_passage_1c_result, p, domain, llm): p for p in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage 1C"):
                passage = futures[fut]
                try:
                    status, rows, error = fut.result()
                except Exception as exc:  # noqa: BLE001 - recorded for durable retry.
                    status, rows, error = "exception", [], str(exc)
                    log.warning("Stage 1C passage failed: %s: %s", passage.passage_id, exc)
                if rows:
                    append_stage1c_rows(row_stream, rows)
                    all_rows.extend(rows)
                append_stage1c_status(
                    status_stream,
                    passage,
                    status=status,
                    row_count=len(rows),
                    error=error,
                )

    log.info("Stage 1C: %d KVPs -> %s", len(all_rows), out_file)
    return all_rows
