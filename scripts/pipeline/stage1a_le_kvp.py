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
from scripts.pipeline.models import KVPRow, LogEntailment, LogEntailmentList, Passage, QAKeyValuePair
from scripts.pipeline.provenance import (
    entailment_id_for_passage,
    entailments_from_kvp_rows,
    passage_source_chunk_ids,
    passage_source_revision_id,
    sha256_text,
)
from scripts.pipeline.provenance_io import write_jsonl
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


def parse_le_response(raw: str) -> Optional[LogEntailmentList]:
    """Parse LLM response as a LogEntailmentList (multi-entailment)."""
    obj = _extract_json_object(raw)
    if not obj:
        return None
    try:
        return LogEntailmentList(**obj)
    except ValidationError:
        # Fallback: if the model returned a single-entailment shape
        # (legacy/training data), wrap it as a list of one.
        try:
            single = LogEntailment(**obj)
            return LogEntailmentList(entailments=[single])
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
    """Run LE → KVP on one passage. Multiple entailments per passage; up to 3 KVPs per entailment."""
    rows: list[KVPRow] = []

    le_raw = llm.call(LE_SYSTEM, LE_USER.format(text=passage.text), max_tokens=8192)
    if not le_raw:
        return rows
    le_list = parse_le_response(le_raw)
    if not le_list:
        log.debug("Stage 1A: no valid entailments for %s", passage.passage_id)
        return rows

    extractor_model = getattr(llm, "model", None)
    if not isinstance(extractor_model, str):
        extractor_model = None
    extractor_temperature = getattr(llm, "temperature", None)
    if not isinstance(extractor_temperature, (int, float)):
        extractor_temperature = None

    for ent_idx, ent in enumerate(le_list.entailments):
        ent_id = entailment_id_for_passage(
            passage, ent_idx, ent.conclusion, ent.premises
        )
        source_revision_id = passage_source_revision_id(passage)
        source_chunk_ids = passage_source_chunk_ids(passage)
        for prem_idx, premise in enumerate(ent.premises[:3]):
            kvp_raw = llm.call(
                KVP_SYSTEM,
                KVP_USER.format(premise=premise, conclusion=ent.conclusion, text=passage.text),
                max_tokens=4096,
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
                entailment_index=ent_idx,
                premise_index=prem_idx,
                sample_id=None,
                entailment_id=ent_id,
                entailment_claim=ent.conclusion,
                entailment_premises=ent.premises,
                source_revision_ids=[source_revision_id],
                source_chunk_ids=source_chunk_ids,
                extractor_model=extractor_model,
                extractor_prompt_hash=sha256_text(LE_SYSTEM + "\n" + LE_USER),
                extractor_temperature=extractor_temperature,
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
    write_jsonl(
        output_dir / "provenance" / "entailments.jsonl",
        entailments_from_kvp_rows(all_rows),
    )
    log.info("Stage 1A: %d KVPs → %s", len(all_rows), out_file)
    return all_rows
