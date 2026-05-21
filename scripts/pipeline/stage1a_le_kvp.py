"""Stage 1A: Logical Entailment → KVP (one Q+A per premise)."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from pydantic import ValidationError
from tqdm import tqdm

from scripts.pipeline.llm_client import LLMClient
from scripts.pipeline.models import KVPRow, LogEntailment, Passage, QAKeyValuePair
from scripts.pipeline.prompts import KVP_SYSTEM, KVP_USER, LE_SYSTEM, LE_USER

log = logging.getLogger(__name__)


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()


def _extract_json_object(raw: str) -> Optional[dict]:
    s = _strip_fences(raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def parse_le_response(raw: str) -> Optional[LogEntailment]:
    obj = _extract_json_object(raw)
    if not obj:
        return None
    try:
        return LogEntailment(**obj)
    except ValidationError:
        return None


def parse_kvp_response(raw: str) -> Optional[QAKeyValuePair]:
    obj = _extract_json_object(raw)
    if not obj:
        return None
    try:
        return QAKeyValuePair(**obj)
    except ValidationError:
        return None


def process_passage_1a(passage: Passage, llm: LLMClient) -> list[KVPRow]:
    """Run LE → KVP on one passage. Returns up to 3 KVP rows."""
    rows: list[KVPRow] = []

    le_raw = llm.call(LE_SYSTEM, LE_USER.format(text=passage.text), max_tokens=512)
    if not le_raw:
        return rows
    le = parse_le_response(le_raw)
    if not le:
        log.debug("Stage 1A: no valid entailment for %s", passage.passage_id)
        return rows

    for i, premise in enumerate(le.premises[:3]):
        kvp_raw = llm.call(
            KVP_SYSTEM,
            KVP_USER.format(premise=premise, conclusion=le.conclusion, text=passage.text),
            max_tokens=256,
        )
        if not kvp_raw:
            continue
        kvp = parse_kvp_response(kvp_raw)
        if not kvp:
            continue
        rows.append(KVPRow(
            passage_id=passage.passage_id,
            source_url=passage.url,
            product_family=passage.product_family,
            stage="1a",
            premise_index=i,
            question=kvp.question.strip(),
            answer=kvp.answer.strip(),
            context=passage.text,
            refined=False,
        ))
    return rows


def run_stage1a(passages: list[Passage], llm: LLMClient, output_dir: Path,
                max_workers: int = 5) -> list[KVPRow]:
    out_file = output_dir / "stage1a_le.jsonl"
    all_rows: list[KVPRow] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_passage_1a, p, llm) for p in passages]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage 1A"):
            all_rows.extend(fut.result())

    with out_file.open("w") as f:
        for row in all_rows:
            f.write(row.model_dump_json() + "\n")
    log.info("Stage 1A: %d KVPs → %s", len(all_rows), out_file)
    return all_rows
