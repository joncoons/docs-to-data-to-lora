"""Register NeMo Evaluator targets and configs for Stage 3 eval.

Pure builder functions (target/config payloads + build_dataset_payload) are
unit-tested; the CLI wires the Evaluator builders to the live EvaluatorClient
and POSTs. Idempotent on re-run (HTTP 409 "already exists" warnings are
logged and skipped).

Note: datasets are NOT Evaluator entities — they live in entity-store and are
registered by operational Task 9. build_dataset_payload is exported here so
the operational task can re-use the canonical payload shape.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.evaluator_client import EvaluatorClient  # noqa: E402

log = logging.getLogger(__name__)


# --- adapter metadata --------------------------------------------------

_BASE_FROM_SIZE_SLUG = {
    "3.2-1b": "meta/llama-3.2-1b-instruct",
    "3.2-3b": "meta/llama-3.2-3b-instruct",
    "3.1-8b": "meta/llama-3.1-8b-instruct",
    # MoE base (Stage 3 Nemotron Nano r=16, see training_session.log)
    "nemotron-nano-30b": "nvidia/nemotron-3-nano-30b-a3b",
}


@dataclass(frozen=True)
class AdapterRow:
    name: str
    base_model: str
    job_id: str
    collection: str

    @classmethod
    def from_log_line(cls, line: str) -> "AdapterRow":
        """Parse one row of evals/training_session.log into AdapterRow.

        Collection is derived from the adapter name prefix (lora-nim-* vs
        lora-nemo-usvcs-*), not from the surrounding section header — section
        headers are ambiguous when the MoE inventory table mixes both corpora
        under one heading.

        Dense-Llama row shape: `| <name> | <job_id> | <train> | <val> | <wall> |`
        MoE merged-adapter row shape: `| <name> | <path> | <size> | <source> |`
        Shard rows (`-shard-a` / `-shard-b` suffix) are skipped — only the
        merged adapter is a usable eval target.
        """
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 2:
            raise ValueError(f"Cannot parse row: {line!r}")
        name, second = cells[0], cells[1]
        if re.search(r"-shard-[ab]$", name):
            raise ValueError(f"shard row not registrable: {name!r}")
        # Derive collection from name prefix.
        if name.startswith("lora-nim-"):
            collection = "nim_curated"
        elif name.startswith("lora-nemo-usvcs-"):
            collection = "nemo_usvcs_curated"
        else:
            raise ValueError(f"Cannot derive collection from name: {name!r}")
        # MoE merged adapter pattern, e.g. "lora-nim-nemotron-nano-30b-r16".
        # second cell is a filesystem path; use "ties-merged" sentinel for job_id
        # because Round 1 source customizer ids are not recoverable.
        m = re.search(r"-(nemotron-nano-30b)-r\d+$", name)
        if m:
            return cls(name=name, base_model=_BASE_FROM_SIZE_SLUG[m.group(1)],
                       job_id="ties-merged", collection=collection)
        # Dense-Llama pattern, e.g. "lora-nim-llama-3.2-1b-r16"
        m = re.search(r"-(\d\.\d-\d+b)-r\d+$", name)
        if not m:
            raise ValueError(f"Cannot extract base from name: {name!r}")
        size_slug = m.group(1)
        if size_slug not in _BASE_FROM_SIZE_SLUG:
            raise ValueError(f"Unknown base size in name: {name!r}")
        return cls(name=name, base_model=_BASE_FROM_SIZE_SLUG[size_slug],
                   job_id=second, collection=collection)


# --- payload builders --------------------------------------------------

# NeMo Evaluator target schema (validated against current /openapi.json):
# top-level type must be one of model|cached_outputs|retriever|rag|rows|dataset.
# We use "model" for all three target kinds (adapter, base, RAG-as-model) with
# the nested ModelInput.api_endpoint (APIEndpointData) carrying the URL and
# model_id (=OAI model_name routing). A single LoRA-enabled NIM serves both
# base and adapter via the model_id routing.

def build_adapter_target(row: AdapterRow, nim_url: str) -> dict:
    return {
        "name": row.name,
        "namespace": "default",
        "type": "model",
        "model": {
            "api_endpoint": {
                "url": f"{nim_url.rstrip('/')}/v1/chat/completions",
                "model_id": row.name,
                "format": "nim",
            },
        },
    }


def build_base_target(base_model: str, nim_url: str) -> dict:
    # "meta/llama-3.2-3b-instruct" → "base-llama-3.2-3b-instruct"
    suffix = base_model.split("/", 1)[1]
    return {
        "name": f"base-{suffix}",
        "namespace": "default",
        "type": "model",
        "model": {
            "api_endpoint": {
                "url": f"{nim_url.rstrip('/')}/v1/chat/completions",
                "model_id": base_model,
                "format": "nim",
            },
        },
    }


def build_rag_target(collection: str, rag_url: str) -> dict:
    """RAG target — registered as type=model pointing at rag-server /generate.

    The Evaluator's native RAGTargetInput requires a full pipeline definition
    (retriever + generator + cached_outputs) that's structurally heavier than
    we need. Simpler: treat the RAG path as a single model endpoint and let
    rag-server own the retrieval. Per-request collection scoping requires
    either:
      (a) a proxy that translates Evaluator's OAI-style request body to
          rag-server's Prompt schema and injects collection_names=[<coll>], or
      (b) two rag-server deployments with different default collections.
    Decide empirically when the first RAG smoke job runs.
    """
    short = collection.replace("_", "-")
    return {
        "name": f"rag-49b-{short}",
        "namespace": "default",
        "type": "model",
        "model": {
            "api_endpoint": {
                "url": f"{rag_url.rstrip('/')}/generate",
                "model_id": f"rag-49b-{short}",  # sentinel; rag-server ignores
                "format": "nim",
            },
        },
    }


def build_dataset_payload(collection: str, files_url: str) -> dict:
    """Build the JSON payload for an entity-store dataset registration.

    Note: datasets live in NeMo Entity Store (not Evaluator). This payload is
    POSTed to http://nemo-entity-store:8000/v1/datasets by operational Task 9
    (after uploading the test_set.jsonl files to NeMo Data Store via the HF
    Hub API). This script's CLI does NOT register datasets — it only handles
    Evaluator-owned targets and configs. The builder is exported so the
    operational task can re-use it.
    """
    return {
        "name": f"stage3-{collection.replace('_','-')}-test",
        "namespace": "default",
        "description": (
            f"Stage 3 held-out test set for {collection} "
            f"(10% KVP holdout, seed 42)"
        ),
        "format": "hf",
        "files_url": files_url,
        "hf_endpoint": "http://nemo-data-store:3000/v1/hf",
    }


_RUBRIC_PROMPT = """Grade this question + answer + ground-truth + source-context tuple on four 1-5 axes.

Question:        {question}
Adapter answer:  {answer}
Ground truth:    {ground_truth}
Source context:  {context}

Score each axis 1-5:
- Accuracy: does the adapter answer match the ground-truth in facts?
- Completeness: does it cover everything the ground-truth says?
- Faithfulness: are claims grounded in the source context?
- Clarity: is it readable and well-formed?

Return JSON: {{"accuracy": int, "completeness": int, "faithfulness": int, "clarity": int, "reason": str}}.
"""

_INFERENCE_PROMPT = "{question}"


_PAIRWISE_PROMPT = """Compare two answers (A and B) to the same question.

Question:       {question}
Answer A:       {answer_a}
Answer B:       {answer_b}
Source context: {context}

Which answer is better grounded, more complete, more faithful to the context?
Return JSON: {{"winner": "A"|"B"|"TIE", "reason": str}}.
"""


def build_singleaxis_config() -> dict:
    return {
        "name": "stage3-singleaxis-rubric",
        "namespace": "default",
        "description": "Stage 3 single-axis 4-criteria rubric, Claude Sonnet judge",
        "type": "custom",
        "params": {
            "parallelism": 4,
            "temperature": 0.0001,  # Evaluator schema requires temperature > 0; greedy-equivalent
            "max_tokens": 600,
            "extra": {
                "judge_model": "aws/anthropic/bedrock-claude-sonnet-4-6",
                "judge_endpoint": "https://inference-api.nvidia.com/v1/chat/completions",
                "inference_prompt": _INFERENCE_PROMPT,
                "rubric_prompt": _RUBRIC_PROMPT,
            },
        },
    }


def build_pairwise_config() -> dict:
    return {
        "name": "stage3-pairwise-tournament",
        "namespace": "default",
        "description": "Stage 3 pairwise A/B/Tie with position swap, Claude judge",
        "type": "custom",
        "params": {
            "parallelism": 4,
            "temperature": 0.0001,  # Evaluator schema requires temperature > 0; greedy-equivalent
            "max_tokens": 300,
            "extra": {
                "judge_model": "aws/anthropic/bedrock-claude-sonnet-4-6",
                "judge_endpoint": "https://inference-api.nvidia.com/v1/chat/completions",
                "pairwise_prompt": _PAIRWISE_PROMPT,
                "position_swap": True,
            },
        },
    }


# --- adapter inventory from log ---------------------------------------

def load_adapters_from_log(log_path: Path) -> list[AdapterRow]:
    """Parse evals/training_session.log into AdapterRow list.

    Walks all table rows starting with `| lora-`, derives collection from the
    name itself, dedupes by name (the merged-adapter row may appear in both a
    corpus section and the final inventory table).
    """
    seen: dict[str, AdapterRow] = {}
    for line in log_path.read_text().splitlines():
        if not line.startswith("| lora-"):
            continue
        try:
            row = AdapterRow.from_log_line(line)
        except ValueError as e:
            log.debug("skipping row %r: %s", line, e)
            continue
        seen.setdefault(row.name, row)
    return list(seen.values())


# --- idempotent create helpers ----------------------------------------

def _create_target_idempotent(client: EvaluatorClient, payload: dict,
                               label: str) -> None:
    """POST a target; tolerate 409 (already exists), re-raise everything else."""
    import httpx
    try:
        tid = client.create_target(payload)
        log.info("%s target id=%s name=%s", label, tid, payload["name"])
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409:
            log.warning("%s target already exists, skipping: %s",
                        label, payload["name"])
        else:
            raise


def _create_config_idempotent(client: EvaluatorClient, payload: dict) -> None:
    """POST a config; tolerate 409 (already exists), re-raise everything else."""
    import httpx
    try:
        cid = client.create_config(payload)
        log.info("config id=%s name=%s", cid, payload["name"])
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409:
            log.warning("config already exists, skipping: %s",
                        payload["name"])
        else:
            raise


# --- CLI --------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator-url",
                    default="http://192.168.1.187:30913")
    ap.add_argument("--log-path", type=Path,
                    default=_REPO_ROOT / "evals" / "training_session.log")
    ap.add_argument("--adapter-targets", action="store_true",
                    help="Register 12 adapter targets")
    ap.add_argument("--base-targets", action="store_true",
                    help="Register 3 base reference targets")
    ap.add_argument("--rag-targets", action="store_true",
                    help="Register 2 RAG targets")
    ap.add_argument("--configs", action="store_true",
                    help="Register both eval configs (singleaxis + pairwise)")
    # A single LoRA-enabled NIM per base model serves BOTH base-only inference
    # (request body model=<base_model>) and LoRA-applied inference (model=<adapter>)
    # via OpenAI-API model routing. No separate base NIM is needed.
    ap.add_argument("--nim-url-1b", default="http://nim-llama-3.2-1b:8000")
    ap.add_argument("--nim-url-3b", default="http://nim-llama-3.2-3b:8000")
    ap.add_argument("--nim-url-8b", default="http://nim-llama-3.1-8b:8000")
    ap.add_argument("--nim-url-nano", default="http://nim-nemotron-nano:8000",
                    help="Nano-30B-A3B MoE adapter-serving NIM (r=16 LoRA)")
    ap.add_argument("--rag-url", default="http://rag-server.runai-rag:8081",
                    help="rag-server /generate base; was rag-agent-toolkit but "
                         "switched to bypass the agent for per-request collection "
                         "scoping (see build_rag_target docstring)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    adapters = load_adapters_from_log(args.log_path)
    log.info("loaded %d adapters from %s", len(adapters), args.log_path)

    nim_url_for = {
        "meta/llama-3.2-1b-instruct": args.nim_url_1b,
        "meta/llama-3.2-3b-instruct": args.nim_url_3b,
        "meta/llama-3.1-8b-instruct": args.nim_url_8b,
        "nvidia/nemotron-3-nano-30b-a3b": args.nim_url_nano,
    }
    # Base targets reuse the same LoRA-enabled NIMs (model-id routing handles
    # which inference path runs). Nano is intentionally excluded from the
    # base-target sweep — only its LoRA variant is in scope per Stage 3.
    base_url_for = {
        "meta/llama-3.2-1b-instruct": args.nim_url_1b,
        "meta/llama-3.2-3b-instruct": args.nim_url_3b,
        "meta/llama-3.1-8b-instruct": args.nim_url_8b,
    }

    with EvaluatorClient(args.evaluator_url) as client:
        if args.adapter_targets:
            for a in adapters:
                p = build_adapter_target(a, nim_url=nim_url_for[a.base_model])
                _create_target_idempotent(client, p, label="adapter")
        if args.base_targets:
            for base, url in base_url_for.items():
                p = build_base_target(base, nim_url=url)
                _create_target_idempotent(client, p, label="base")
        if args.rag_targets:
            for coll in ("nim_curated", "nemo_usvcs_curated"):
                p = build_rag_target(coll, rag_url=args.rag_url)
                _create_target_idempotent(client, p, label="rag")
        if args.configs:
            for builder in (build_singleaxis_config, build_pairwise_config):
                p = builder()
                _create_config_idempotent(client, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
