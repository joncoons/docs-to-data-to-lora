"""Stage 1C: Instruction diversity pass on top-density passages."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from scripts.pipeline.llm_client import LLMClient
from scripts.pipeline.models import KVPRow, Passage
from scripts.pipeline.provenance import passage_source_chunk_ids, passage_source_revision_id
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


def select_top_density_passages(
    passages: list[Passage], top_percent: float = 0.25, min_passages: int = 100
) -> list[Passage]:
    scored = sorted(passages, key=density_score, reverse=True)
    if len(passages) < min_passages:
        # Corpus smaller than the floor — return everything
        return scored
    cap = int(len(passages) * top_percent)
    return scored[:cap]


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


def process_passage_1c(passage: Passage, domain: str, llm: LLMClient) -> list[KVPRow]:
    raw = llm.call(
        INSTRUCTION_SYSTEM,
        INSTRUCTION_USER.format(domain=domain, passage=passage.text),
        max_tokens=4096,
    )
    if not raw:
        return []
    pairs = parse_instruction_response(raw)
    rows: list[KVPRow] = []
    source_revision_id = passage_source_revision_id(passage)
    source_chunk_ids = passage_source_chunk_ids(passage)
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
            refined=False,
        ))
    return rows


def run_stage1c(passages: list[Passage], domain: str, llm: LLMClient,
                output_dir: Path, top_percent: float = 0.25,
                min_passages: int = 100, max_workers: int = 5) -> list[KVPRow]:
    selected = select_top_density_passages(passages, top_percent=top_percent,
                                            min_passages=min_passages)
    log.info("Stage 1C: %d passages selected (top %.0f%% with floor %d)",
             len(selected), top_percent * 100, min_passages)

    out_file = output_dir / "stage1c_instruction.jsonl"
    all_rows: list[KVPRow] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_passage_1c, p, domain, llm) for p in selected]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage 1C"):
            all_rows.extend(fut.result())

    with out_file.open("w") as f:
        for r in all_rows:
            f.write(r.model_dump_json() + "\n")
    log.info("Stage 1C: %d KVPs → %s", len(all_rows), out_file)
    return all_rows
