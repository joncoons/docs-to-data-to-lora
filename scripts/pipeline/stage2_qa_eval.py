"""Stage 2: QA Eval refinement — super-120b self-eval, refine or drop."""
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
from scripts.pipeline.models import KVPRow, QAEvaluation
from scripts.pipeline.prompts import QA_EVAL_SYSTEM, QA_EVAL_USER

log = logging.getLogger(__name__)


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()


def parse_eval_response(raw: str) -> Optional[QAEvaluation]:
    s = _strip_fences(raw)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    try:
        return QAEvaluation(**obj)
    except ValidationError:
        return None


def refine_row(row: KVPRow, llm: LLMClient) -> Optional[KVPRow]:
    raw = llm.call(QA_EVAL_SYSTEM,
                   QA_EVAL_USER.format(question=row.question, answer=row.answer,
                                       context=row.context),
                   max_tokens=2048)
    if not raw:
        return None
    ev = parse_eval_response(raw)
    if not ev:
        return None
    changed = (ev.prompt.strip() != row.question.strip() or
               ev.completion.strip() != row.answer.strip())
    return row.model_copy(update={
        "question": ev.prompt.strip(),
        "answer": ev.completion.strip(),
        "refined": changed,
    })


def run_stage2(rows: list[KVPRow], llm: LLMClient, output_dir: Path,
               max_workers: int = 5) -> tuple[list[KVPRow], list[KVPRow]]:
    """Returns (kept_rows, dropped_rows)."""
    kept: list[KVPRow] = []
    dropped: list[KVPRow] = []

    out_file = output_dir / "stage2_eval.jsonl"
    drop_file = output_dir / "stage2_dropped.jsonl"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(refine_row, r, llm): r for r in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage 2"):
            orig = futures[fut]
            result = fut.result()
            if result is None:
                dropped.append(orig)
            else:
                kept.append(result)

    with out_file.open("w") as f:
        for r in kept:
            f.write(r.model_dump_json() + "\n")
    with drop_file.open("w") as f:
        for r in dropped:
            f.write(r.model_dump_json() + "\n")
    log.info("Stage 2: %d kept, %d dropped → %s", len(kept), len(dropped), out_file)
    return kept, dropped
