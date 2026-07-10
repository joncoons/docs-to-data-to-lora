"""Pydantic schemas for the Stage 2 dataset pipeline.

Direct descendants of pydantic_models.py from the Jan 2025 logical entailment
experiment, upgraded to Pydantic 2.x and split for the multi-stage pipeline.
"""
from typing import Literal
from pydantic import BaseModel, Field


class LogEntailment(BaseModel):
    """Output of Stage 1A entailment extraction."""
    conclusion: str = Field(description="Identified theme, concept or topic")
    premises: list[str] = Field(min_length=1,
                                description="Logical premises supporting the conclusion")
    context: str = Field(default="", description="Ancillary details supporting the main theme")
    entities: str = Field(default="", description="Identified noteworthy entities")
    recommendations: str = Field(default="", description="Additional suggested actions")


class LogEntailmentList(BaseModel):
    """Multiple logical entailments extracted from a single passage.

    A passage may cover multiple topics, each warranting its own
    (conclusion, premises) entailment. The original Jan 2025 LE
    notebook handled this via semantic sub-chunking; we collapse the
    same intent into a single LLM call returning a list.
    """
    entailments: list[LogEntailment] = Field(min_length=1)


class QAKeyValuePair(BaseModel):
    """Output of Stage 1A KVP generation: premise→question, conclusion→answer."""
    question: str
    answer: str


class QAEvaluation(BaseModel):
    """Output of Stage 2 QA Eval refinement."""
    prompt: str
    completion: str


class SynthesisPair(BaseModel):
    type: Literal["bridging", "contrastive"]
    question: str
    answer: str


class SynthesisPairs(BaseModel):
    """Output of Stage 1B kNN synthesis (one bridging + one contrastive per call)."""
    pairs: list[dict]  # dicts for backwards-compatible raw judge JSON


class InstructionPair(BaseModel):
    type: Literal["summary", "listicle", "procedural"]
    question: str
    answer: str


class InstructionPairs(BaseModel):
    """Output of Stage 1C instruction diversity (summary/listicle/procedural)."""
    pairs: list[dict]


class Passage(BaseModel):
    """A reconstructed passage emitted by Stage 0."""
    passage_id: str
    url: str
    text: str
    token_count: int
    chunk_ids: list[str]
    product_family: str
    product_name: str
    doc_kind: Literal["html", "pdf"]
    source_revision_id: str | None = None
    source_chunk_ids: list[str] | None = None
    source_systems: list[str] | None = None
    source_kinds: list[str] | None = None
    modalities: list[str] | None = None


class KVPRow(BaseModel):
    """A single row in any per-stage JSONL output."""
    passage_id: str
    source_url: str
    product_family: str
    stage: Literal["1a", "1b", "1c", "1.5", "2"]
    question: str
    answer: str
    context: str
    refined: bool = False
    # Optional per-stage extras (any of these may be None):
    premise_index: int | None = None
    entailment_index: int | None = None   # which entailment within a passage
    qa_type: str | None = None          # for 1b/1c
    instr_type: str | None = None       # for 1c
    target_product_family: str | None = None  # for 1.5
    retrieved_urls: list[str] | None = None   # for 1.5
    neighbor_urls: list[str] | None = None    # for 1b
    # Provenance sidecar fields. These are optional so older JSONL outputs still
    # load and the Customizer-facing shape remains unchanged downstream.
    sample_id: str | None = None
    entailment_id: str | None = None
    entailment_claim: str | None = None
    entailment_premises: list[str] | None = None
    source_revision_ids: list[str] | None = None
    source_chunk_ids: list[str] | None = None
    source_systems: list[str] | None = None
    source_kinds: list[str] | None = None
    modalities: list[str] | None = None
    extractor_model: str | None = None
    extractor_prompt_hash: str | None = None
    extractor_temperature: float | None = None
