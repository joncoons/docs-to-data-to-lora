"""Conservative Stage 1A efficiency prototype.

This module keeps the current two-step audit boundary:

    passage -> logical entailments -> QA rows

The difference from `scripts.pipeline.stage1a_le_kvp` is that the second step
generates QA rows for all parsed premises in a capped batch instead of making
one KVP call per premise. Invalid or missing batched rows fall back to the
current per-premise KVP prompt.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.models import (  # noqa: E402
    KVPRow,
    LogEntailment,
    LogEntailmentList,
    Passage,
    QAKeyValuePair,
)
from scripts.pipeline.prompts import KVP_SYSTEM, KVP_USER, LE_SYSTEM, LE_USER  # noqa: E402
from scripts.pipeline.provenance import (  # noqa: E402
    entailment_id_for_passage,
    entailments_from_kvp_rows,
    passage_modalities,
    passage_source_chunk_ids,
    passage_source_kinds,
    passage_source_revision_id,
    passage_source_systems,
    sha256_text,
)
from scripts.pipeline.provenance_io import write_jsonl  # noqa: E402
log = logging.getLogger(__name__)


class LLMClient(Protocol):
    model: object
    temperature: object

    def call(self, system: str, user: str, max_tokens: int = 1024) -> Optional[str]:
        ...

DEFAULT_NIM_ENDPOINTS = os.getenv(
    "PIPELINE_NIM_ENDPOINTS",
    "http://nim-llm-super-120b-bw.runai-rag:8000/v1",
)
DEFAULT_LLM_MODEL = os.getenv("PIPELINE_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
DEFAULT_LLM_API_KEY = os.getenv("PIPELINE_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY") or "local"

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



def parse_le_response(raw: str) -> Optional[LogEntailmentList]:
    obj = _extract_json_object(raw)
    if not obj:
        return None
    try:
        return LogEntailmentList(**obj)
    except ValidationError:
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
        for premise_index, premise in enumerate(entailment.premises[:3]):
            items.append(EntailmentPremiseItem(
                entailment_index=entailment_index,
                premise_index=premise_index,
                premise=premise,
                conclusion=entailment.conclusion,
                premises=entailment.premises,
            ))
    return items


def _chunks(items: list[EntailmentPremiseItem], size: int) -> Iterable[list[EntailmentPremiseItem]]:
    if size < 1:
        raise ValueError("batch size must be >= 1")
    for start in range(0, len(items), size):
        yield items[start:start + size]


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


def _row_from_pair(
    *,
    passage: Passage,
    item: EntailmentPremiseItem,
    question: str,
    answer: str,
    llm: LLMClient,
    prompt_hash: str,
) -> KVPRow:
    source_revision_id = passage_source_revision_id(passage)
    source_chunk_ids = passage_source_chunk_ids(passage)
    source_systems = passage_source_systems(passage)
    source_kinds = passage_source_kinds(passage)
    modalities = passage_modalities(passage)
    entailment_id = entailment_id_for_passage(
        passage,
        item.entailment_index,
        item.conclusion,
        item.premises,
    )
    extractor_model = getattr(llm, "model", None)
    if not isinstance(extractor_model, str):
        extractor_model = None
    extractor_temperature = getattr(llm, "temperature", None)
    if not isinstance(extractor_temperature, (int, float)):
        extractor_temperature = None

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
        source_chunk_ids=source_chunk_ids,
        source_systems=source_systems or None,
        source_kinds=source_kinds or None,
        modalities=modalities or None,
        extractor_model=extractor_model,
        extractor_prompt_hash=prompt_hash,
        extractor_temperature=extractor_temperature,
        question=question.strip(),
        answer=answer.strip(),
        context=passage.text,
        refined=False,
    )


def _fallback_one_item(
    passage: Passage,
    item: EntailmentPremiseItem,
    llm: LLMClient,
) -> KVPRow | None:
    raw = llm.call(
        KVP_SYSTEM,
        KVP_USER.format(
            premise=item.premise,
            conclusion=item.conclusion,
            text=passage.text,
        ),
        max_tokens=4096,
    )
    if not raw:
        return None
    kvp = parse_kvp_response(raw)
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
    llm: LLMClient,
    *,
    parse_attempts: int,
) -> BatchedKVPResponse | None:
    if parse_attempts < 1:
        raise ValueError("parse_attempts must be >= 1")
    user = BATCHED_KVP_USER.format(
        entailment_items=_render_entailment_items(items),
        text=passage.text,
    )
    for _ in range(parse_attempts):
        raw = llm.call(BATCHED_KVP_SYSTEM, user, max_tokens=8192)
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
    llm: LLMClient,
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
        rows.append(_row_from_pair(
            passage=passage,
            item=item,
            question=pair.question,
            answer=pair.answer,
            llm=llm,
            prompt_hash=BATCHED_PROMPT_HASH,
        ))
        produced.add(item.key)

    return rows, produced


def process_passage_1a_batched(
    passage: Passage,
    llm: LLMClient,
    *,
    max_premises_per_batch: int = 12,
    batch_parse_attempts: int = 2,
) -> list[KVPRow]:
    le_raw = llm.call(LE_SYSTEM, LE_USER.format(text=passage.text), max_tokens=8192)
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
            fallback_row = _fallback_one_item(passage, item, llm)
            if fallback_row is not None:
                rows.append(fallback_row)

    return rows


def read_passages_jsonl(path: Path) -> list[Passage]:
    if not path.exists():
        raise FileNotFoundError(f"Passage input not found: {path}")
    return [Passage.model_validate_json(line) for line in path.read_text().splitlines() if line]


def run_stage1a_batched(
    passages: list[Passage],
    llm: LLMClient,
    output_dir: Path,
    *,
    max_premises_per_batch: int = 12,
    batch_parse_attempts: int = 2,
    output_filename: str = "stage1a_le.batched_kvp.jsonl",
    entailments_filename: str = "entailments.batched_kvp.jsonl",
) -> list[KVPRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / output_filename
    entailments_file = output_dir / "provenance" / entailments_filename
    rows: list[KVPRow] = []

    for passage in tqdm(passages, desc="Stage 1A batched KVP"):
        rows.extend(process_passage_1a_batched(
            passage,
            llm,
            max_premises_per_batch=max_premises_per_batch,
            batch_parse_attempts=batch_parse_attempts,
        ))

    with out_file.open("w") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    write_jsonl(entailments_file, entailments_from_kvp_rows(rows))
    log.info("Batched Stage 1A: %d KVPs -> %s", len(rows), out_file)
    return rows


def _parse_endpoints(value: str) -> list[str]:
    endpoints = [endpoint.strip() for endpoint in value.split(",") if endpoint.strip()]
    if not endpoints:
        raise ValueError("At least one LLM endpoint is required")
    return endpoints


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run isolated Stage 1A batched KVP prototype.")
    ap.add_argument("--input-passages", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--nim-endpoints", default=DEFAULT_NIM_ENDPOINTS)
    ap.add_argument("--model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--api-key", default=DEFAULT_LLM_API_KEY)
    ap.add_argument("--max-workers", type=int, default=5)
    ap.add_argument("--min-request-interval-s", type=float, default=0.5)
    ap.add_argument("--retry-attempts", type=int, default=3)
    ap.add_argument("--retry-base-delay-s", type=float, default=5.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-premises-per-batch", type=int, default=12)
    ap.add_argument("--batch-parse-attempts", type=int, default=2)
    ap.add_argument("--no-think", action="store_true", default=True)
    ap.add_argument("--allow-think", dest="no_think", action="store_false")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args(argv)
    from scripts.pipeline.llm_client import LLMClient as RuntimeLLMClient

    llm = RuntimeLLMClient(
        endpoints=_parse_endpoints(args.nim_endpoints),
        model=args.model,
        api_key=args.api_key,
        max_workers=args.max_workers,
        min_interval_s=args.min_request_interval_s,
        retry_attempts=args.retry_attempts,
        retry_base_delay_s=args.retry_base_delay_s,
        no_think=args.no_think,
        temperature=args.temperature,
    )
    passages = read_passages_jsonl(args.input_passages)
    run_stage1a_batched(
        passages,
        llm,
        args.output_dir,
        max_premises_per_batch=args.max_premises_per_batch,
        batch_parse_attempts=args.batch_parse_attempts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
