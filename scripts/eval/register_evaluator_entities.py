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
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.evaluator_client import EvaluatorClient  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_EVALUATOR_URL = os.getenv("EVALUATOR_URL", "http://nemo-evaluator:8000")
DEFAULT_NIM_PROXY_URL = os.getenv("NIM_PROXY_URL", "http://nemo-nim-proxy:8000")
DEFAULT_TRAINING_SESSION_LOG = Path(
    os.getenv(
        "TRAINING_SESSION_LOG",
        str(_REPO_ROOT / "evals" / "training_session.log"),
    )
)


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

# NeMo Evaluator target schema (validated against /openapi.json): top-level
# type=model with nested ModelInput.api_endpoint (url + model_id + format).
#
# Every Stage 3 target should point at native NeMo NIM Proxy. Payload or
# response adaptation should be handled by Evaluator target/configuration or
# Evaluator interceptors before introducing a custom proxy.
#
# Target naming convention (post-2026-05-27 redesign):
#   - LoRA adapters:  lora-{corpus}-{base_short}-r{rank}   (unchanged)
#   - Base models:    bare base-model identifier, org/ stripped, e.g.
#                       "llama-3.2-1b-instruct"
#                       "llama-3.3-nemotron-super-49b-v1.5"
#   - 49B is just another base target — corpus disambiguation lives in the
#     *dataset* (-with-context variants per corpus), not in the target name.

_NEMOTRON_SUPER_49B_BASE = "nvidia/llama-3.3-nemotron-super-49b-v1.5"


def _base_target_name(base_model: str) -> str:
    """Strip org/ prefix — slashes aren't allowed in Evaluator entity names."""
    return base_model.split("/", 1)[1]


def build_adapter_target(row: AdapterRow, proxy_url: str) -> dict:
    return {
        "name": row.name,
        "namespace": "default",
        "type": "model",
        "model": {
            "api_endpoint": {
                "url": f"{proxy_url.rstrip('/')}/v1/chat/completions",
                "model_id": row.name,
                "format": "nim",
            },
        },
    }


def build_base_target(base_model: str, proxy_url: str) -> dict:
    target_name = _base_target_name(base_model)
    return {
        "name": target_name,
        "namespace": "default",
        "type": "model",
        "model": {
            "api_endpoint": {
                "url": f"{proxy_url.rstrip('/')}/v1/chat/completions",
                "model_id": target_name,
                "format": "nim",
            },
        },
    }


def build_49b_target(proxy_url: str) -> dict:
    """The Nemotron-Super-49B comparator. Registered as a model target;
    corpus pairing is handled by which -with-context dataset it's evaluated on,
    not by duplicating the target."""
    return build_base_target(_NEMOTRON_SUPER_49B_BASE, proxy_url)


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


# RAGAS input_template: maps our test-row + model-sample fields to the schema
# RAGAS's EvaluationDataset.from_list expects. Renders to a JSON object using
# Jinja's `tojson` filter to safely escape strings (newlines, quotes).
#
# Field mapping rationale:
#   user_input         ← `prompt` (full baked prompt; question is at the end)
#   retrieved_contexts ← `[prompt]` (single-element list containing the full
#                        baked prompt — chunks are inside it. Question text is
#                        harmless noise for grounding judging.)
#   response           ← `response` (model output from sample)
#   reference          ← `completion` (ground-truth answer)
_RAGAS_INPUT_TEMPLATE = (
    '{\n'
    '  "user_input":         {{ prompt | tojson }},\n'
    '  "retrieved_contexts": [{{ prompt | tojson }}],\n'
    '  "response":           {{ response | tojson }},\n'
    '  "reference":          {{ completion | tojson }}\n'
    '}'
)

_JUDGE_MODEL_REF = "default/claude-sonnet-4-6-judge"


def _ragas_metric(metric_type: str) -> dict:
    """Build one RAGAS metric config entry for the rubric tasks block."""
    return {
        "type": metric_type,
        "params": {
            "judge": {"model": _JUDGE_MODEL_REF},
            "input_template": _RAGAS_INPUT_TEMPLATE,
        },
    }


def build_singleaxis_config() -> dict:
    """Stage 3 single-axis RAGAS rubric: Faithfulness + ResponseRelevancy + AnswerAccuracy.

    These three are the canonical RAG-eval subset (the prior nim-sft-final
    experiment used a near-identical set: faithfulness, answer_relevancy,
    context_precision). All scored by Claude Sonnet via NVIDIA Inference API
    (model entity: default/claude-sonnet-4-6-judge).
    """
    return {
        "name": "stage3-singleaxis-rubric",
        "namespace": "default",
        "description": (
            "Stage 3 single-axis RAGAS — Faithfulness + ResponseRelevancy + "
            "AnswerAccuracy, Claude Sonnet 4.6 judge"
        ),
        "type": "custom",
        "params": {
            "parallelism":  4,
            "temperature":  0.0001,
            "max_tokens":   8192,
        },
        "tasks": {
            "ragas_rubric": {
                "type": "chat-completion",
                "params": {"template": "{{prompt}}"},
                "metrics": {
                    "faithfulness":       _ragas_metric("faithfulness"),
                    "response_relevancy": _ragas_metric("response_relevancy"),
                    "answer_accuracy":    _ragas_metric("answer_accuracy"),
                },
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
            "max_tokens": 8192,     # target inference budget; reasoning models (49B) need room for <think> + answer
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
    ap.add_argument("--evaluator-url", default=DEFAULT_EVALUATOR_URL,
                    help="NeMo Evaluator base URL. Defaults to EVALUATOR_URL or "
                         "http://nemo-evaluator:8000.")
    ap.add_argument("--evaluator-api-key", default=os.getenv("EVALUATOR_API_KEY"),
                    help="Optional Evaluator bearer token. Defaults to EVALUATOR_API_KEY.")
    ap.add_argument("--log-path", type=Path, default=DEFAULT_TRAINING_SESSION_LOG,
                    help="Training-session inventory log. Defaults to TRAINING_SESSION_LOG "
                         "or evals/training_session.log.")
    ap.add_argument("--adapter-targets", action="store_true",
                    help="Register the 14 LoRA adapter targets (12 Llama + 2 Nano r=16)")
    ap.add_argument("--base-targets", action="store_true",
                    help="Register the 3 dense Llama base reference targets")
    ap.add_argument("--49b-target", "--rag-target", dest="target_49b", action="store_true",
                    help="Register the single Nemotron-Super-49B-v1.5 comparator target")
    ap.add_argument("--configs", action="store_true",
                    help="Register both eval configs (singleaxis + pairwise)")
    ap.add_argument("--all", action="store_true",
                    help="Register adapter targets, base targets, 49B comparator, and configs")
    ap.add_argument("--proxy-url", default=DEFAULT_NIM_PROXY_URL,
                    help="NeMo NIM Proxy base URL. Defaults to NIM_PROXY_URL or "
                         "http://nemo-nim-proxy:8000. All Stage 3 Evaluator "
                         "model targets point at this proxy's /v1/chat/completions endpoint.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    register_adapter_targets = args.all or args.adapter_targets
    register_base_targets = args.all or args.base_targets
    register_49b_target = args.all or args.target_49b
    register_configs = args.all or args.configs
    if not any((register_adapter_targets, register_base_targets,
                register_49b_target, register_configs)):
        log.warning("no registration flags selected; use --all or an individual flag")
        return 0

    adapters = load_adapters_from_log(args.log_path)
    log.info("loaded %d adapters from %s", len(adapters), args.log_path)

    # Bases that get a no-LoRA reference target in the matrix. Nano is excluded
    # (only its LoRA variants are in Stage 3 scope); 49B is registered via the
    # separate build_49b_target helper to keep its naming explicit.
    _BASE_TARGETS = [
        "meta/llama-3.2-1b-instruct",
        "meta/llama-3.2-3b-instruct",
        "meta/llama-3.1-8b-instruct",
    ]

    with EvaluatorClient(args.evaluator_url, api_key=args.evaluator_api_key) as client:
        if register_adapter_targets:
            for a in adapters:
                p = build_adapter_target(a, proxy_url=args.proxy_url)
                _create_target_idempotent(client, p, label="adapter")
        if register_base_targets:
            for base in _BASE_TARGETS:
                p = build_base_target(base, proxy_url=args.proxy_url)
                _create_target_idempotent(client, p, label="base")
        if register_49b_target:
            p = build_49b_target(proxy_url=args.proxy_url)
            _create_target_idempotent(client, p, label="49b")
        if register_configs:
            for builder in (build_singleaxis_config, build_pairwise_config):
                p = builder()
                _create_config_idempotent(client, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
