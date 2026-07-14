"""Optional batched Stage 1A logical-entailment to KVP expansion.

This keeps the Stage 1A audit boundary:

    passage -> logical entailments -> QA rows

Compared with ``stage1a_le_kvp``, only the KVP expansion step is batched. The
LE call still runs first, and any missing/invalid batched KVP rows fall back to
the original per-premise KVP prompt.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from scripts.pipeline.models import KVPRow, LogEntailment, Passage, QAKeyValuePair
from scripts.pipeline.prompts import KVP_SYSTEM, KVP_USER, LE_SYSTEM, LE_USER
from scripts.pipeline.provenance import (
    entailment_id_for_passage,
    entailments_from_kvp_rows,
    passage_modalities,
    passage_source_chunk_ids,
    passage_source_kinds,
    passage_source_revision_id,
    passage_source_systems,
    sha256_text,
)
from scripts.pipeline.provenance_io import write_jsonl
from scripts.pipeline.stage1a_le_kvp import parse_kvp_response, parse_le_response

log = logging.getLogger(__name__)


class Stage1ALLM(Protocol):
    model: object
    temperature: object

    def call(self, system: str, user: str, max_tokens: int = 1024) -> Optional[str]:
        ...


BATCHED_KVP_SYSTEM = (
    "You are a technical documentation analyst. Generate QA pairs as JSON. "
    "Return ONLY valid JSON - no explanation, no markdown fences."
)

BATCHED_KVP_USER = """\
Given logical entailments extracted from one documentation passage, generate one
substantive question-answer pair for each listed premise.

Rules for each QUESTION:
- Use the supplied entailment_index and premise_index exactly as provided.
- Derive the question from that specific PREMISE.
- Require specific, verifiable knowledge such as an env var name, command,
  parameter, version, configuration value, or product-specific identifier.
- Avoid yes/no questions and avoid generic "what is X" questions when X is a
  well-known term.

Rules for each ANSWER:
- State the core claim conveyed by the corresponding CONCLUSION.
- Cite specific identifier(s), value(s), command(s), or version-specific details
  from the source text that justify the claim.
- Use complete declarative sentences.
- Keep the answer to 2-4 sentences and 50-120 tokens when the source supports it.

Both questions and answers must be derivable ONLY from the source text. Do not
introduce facts not present in the source.

Return JSON:
{{
  "pairs": [
    {{
      "entailment_index": 0,
      "premise_index": 0,
      "question": "...",
      "answer": "..."
    }}
  ]
}}

Entailment premises to convert:
{entailment_items}

Source text:
{text}"""

BATCHED_PROMPT_HASH = sha256_text(
    LE_SYSTEM + "\n" + LE_USER + "\n" + BATCHED_KVP_SYSTEM + "\n" + BATCHED_KVP_USER
)
FALLBACK_PROMPT_HASH = sha256_text(
    LE_SYSTEM + "\n" + LE_USER + "\n" + KVP_SYSTEM + "\n" + KVP_USER
)


class BatchedKVPPair(BaseModel):
    entailment_index: int = Field(ge=0)
    premise_index: int = Field(ge=0)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class BatchedKVPResponse(BaseModel):
    pairs: list[BatchedKVPPair] = Field(default_factory=list)


@dataclass(frozen=True)
class EntailmentPremiseItem:
    entailment_index: int
    premise_index: int
    premise: str
    conclusion: str
    premises: list[str]

    @property
    def key(self) -> tuple[int, int]:
        return self.entailment_index, self.premise_index


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_json_object(raw: str) -> Optional[dict]:
    text = _strip_fences(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def parse_batched_kvp_response(raw: str) -> Optional[BatchedKVPResponse]:
    obj = _extract_json_object(raw)
    if not obj:
        return None
    try:
        return BatchedKVPResponse(**obj)
    except ValidationError:
        return None


def _entailment_items(entailments: list[LogEntailment]) -> list[EntailmentPremiseItem]:
    items: list[EntailmentPremiseItem] = []
    for entailment_index, entailment in enumerate(entailments):
        for premise_index, premise in enumerate(entailment.premises):
            items.append(
                EntailmentPremiseItem(
                    entailment_index=entailment_index,
                    premise_index=premise_index,
                    premise=premise,
                    conclusion=entailment.conclusion,
                    premises=entailment.premises,
                )
            )
    return items


def _chunks(
    items: list[EntailmentPremiseItem],
    size: int,
) -> Iterable[list[EntailmentPremiseItem]]:
    if size < 1:
        raise ValueError("batch size must be >= 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _render_entailment_items(items: list[EntailmentPremiseItem]) -> str:
    payload = [
        {
            "entailment_index": item.entailment_index,
            "premise_index": item.premise_index,
            "premise": item.premise,
            "conclusion": item.conclusion,
        }
        for item in items
    ]
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _normalized_question(question: str) -> str:
    return re.sub(r"\W+", "", question.casefold())


def _extractor_model(llm: Stage1ALLM) -> str | None:
    extractor_model = getattr(llm, "model", None)
    return extractor_model if isinstance(extractor_model, str) else None


def _extractor_temperature(llm: Stage1ALLM) -> float | None:
    extractor_temperature = getattr(llm, "temperature", None)
    if isinstance(extractor_temperature, (int, float)):
        return float(extractor_temperature)
    return None


def _row_from_pair(
    *,
    passage: Passage,
    item: EntailmentPremiseItem,
    question: str,
    answer: str,
    llm: Stage1ALLM,
    prompt_hash: str,
) -> KVPRow:
    source_revision_id = passage_source_revision_id(passage)
    entailment_id = entailment_id_for_passage(
        passage,
        item.entailment_index,
        item.conclusion,
        item.premises,
    )
    source_systems = passage_source_systems(passage)
    source_kinds = passage_source_kinds(passage)
    modalities = passage_modalities(passage)

    return KVPRow(
        passage_id=passage.passage_id,
        source_url=passage.url,
        product_family=passage.product_family,
        stage="1a",
        entailment_index=item.entailment_index,
        premise_index=item.premise_index,
        sample_id=None,
        entailment_id=entailment_id,
        entailment_claim=item.conclusion,
        entailment_premises=item.premises,
        source_revision_ids=[source_revision_id],
        source_chunk_ids=passage_source_chunk_ids(passage),
        source_systems=source_systems or None,
        source_kinds=source_kinds or None,
        modalities=modalities or None,
        extractor_model=_extractor_model(llm),
        extractor_prompt_hash=prompt_hash,
        extractor_temperature=_extractor_temperature(llm),
        question=question.strip(),
        answer=answer.strip(),
        context=passage.text,
        refined=False,
    )


def _fallback_one_item(
    passage: Passage,
    item: EntailmentPremiseItem,
    llm: Stage1ALLM,
    *,
    max_tokens: int,
) -> KVPRow | None:
    raw = llm.call(
        KVP_SYSTEM,
        KVP_USER.format(
            premise=item.premise,
            conclusion=item.conclusion,
            text=passage.text,
        ),
        max_tokens=max_tokens,
    )
    if not raw:
        return None
    kvp: QAKeyValuePair | None = parse_kvp_response(raw)
    if not kvp:
        return None
    return _row_from_pair(
        passage=passage,
        item=item,
        question=kvp.question,
        answer=kvp.answer,
        llm=llm,
        prompt_hash=FALLBACK_PROMPT_HASH,
    )


def _call_batched_kvp(
    passage: Passage,
    items: list[EntailmentPremiseItem],
    llm: Stage1ALLM,
    *,
    parse_attempts: int,
    max_tokens: int,
) -> BatchedKVPResponse | None:
    if parse_attempts < 1:
        raise ValueError("parse_attempts must be >= 1")
    user = BATCHED_KVP_USER.format(
        entailment_items=_render_entailment_items(items),
        text=passage.text,
    )
    for _ in range(parse_attempts):
        raw = llm.call(BATCHED_KVP_SYSTEM, user, max_tokens=max_tokens)
        if not raw:
            continue
        parsed = parse_batched_kvp_response(raw)
        if parsed is not None and parsed.pairs:
            return parsed
    return None


def _rows_from_batched_response(
    *,
    response: BatchedKVPResponse,
    passage: Passage,
    items: list[EntailmentPremiseItem],
    llm: Stage1ALLM,
) -> tuple[list[KVPRow], set[tuple[int, int]]]:
    by_key = {item.key: item for item in items}
    rows: list[KVPRow] = []
    produced: set[tuple[int, int]] = set()
    normalized_questions: set[str] = set()

    for pair in response.pairs:
        item = by_key.get((pair.entailment_index, pair.premise_index))
        if item is None or item.key in produced:
            continue
        normalized = _normalized_question(pair.question)
        if normalized and normalized in normalized_questions:
            continue
        normalized_questions.add(normalized)
        rows.append(
            _row_from_pair(
                passage=passage,
                item=item,
                question=pair.question,
                answer=pair.answer,
                llm=llm,
                prompt_hash=BATCHED_PROMPT_HASH,
            )
        )
        produced.add(item.key)

    return rows, produced


def process_passage_1a_batched(
    passage: Passage,
    llm: Stage1ALLM,
    *,
    max_premises_per_batch: int = 12,
    batch_parse_attempts: int = 2,
    le_max_tokens: int = 16384,
    batched_kvp_max_tokens: int = 16384,
) -> list[KVPRow]:
    le_raw = llm.call(LE_SYSTEM, LE_USER.format(text=passage.text), max_tokens=le_max_tokens)
    if not le_raw:
        return []
    le_list = parse_le_response(le_raw)
    if not le_list:
        log.debug("Batched Stage 1A: no valid entailments for %s", passage.passage_id)
        return []

    rows: list[KVPRow] = []
    items = _entailment_items(le_list.entailments)
    if not items:
        return rows

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

    return rows


def run_stage1a_batched(
    passages: list[Passage],
    llm: Stage1ALLM,
    output_dir: Path,
    max_workers: int = 5,
    *,
    max_premises_per_batch: int = 12,
    batch_parse_attempts: int = 2,
    le_max_tokens: int = 16384,
    batched_kvp_max_tokens: int = 16384,
    output_filename: str = "stage1a_le.jsonl",
    entailments_filename: str = "entailments.jsonl",
) -> list[KVPRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / output_filename
    entailments_file = output_dir / "provenance" / entailments_filename
    rows: list[KVPRow] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                process_passage_1a_batched,
                passage,
                llm,
                max_premises_per_batch=max_premises_per_batch,
                batch_parse_attempts=batch_parse_attempts,
                le_max_tokens=le_max_tokens,
                batched_kvp_max_tokens=batched_kvp_max_tokens,
            )
            for passage in passages
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Stage 1A batched KVP",
        ):
            rows.extend(future.result())

    with out_file.open("w") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    write_jsonl(entailments_file, entailments_from_kvp_rows(rows))
    log.info("Batched Stage 1A: %d KVPs -> %s", len(rows), out_file)
    return rows
