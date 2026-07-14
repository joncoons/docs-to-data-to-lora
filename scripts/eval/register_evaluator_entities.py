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

DEFAULT_EVALUATOR_URL = os.getenv("EVALUATOR_URL", "http://nemo-evaluator:7331")
DEFAULT_NIM_PROXY_URL = os.getenv(
    "NIM_PROXY_URL",
    "http://rag-oai-proxy.runai-rag:8080",
)
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
}


@dataclass(frozen=True)
class AdapterRow:
    name: str
    base_model: str
    job_id: str
    collection: str

    @classmethod
    def from_log_line(cls, line: str) -> "AdapterRow":
        """Parse one dense-Llama adapter row into AdapterRow.

        Expected row shape: `| <name> | <job_id> | <train> | <val> | <wall> |`.
        Collection is derived from the adapter name prefix (`lora-nim-*` or
        `lora-nemo-usvcs-*`) rather than from surrounding markdown headings.
        """
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 2:
            raise ValueError(f"Cannot parse row: {line!r}")
        name, second = cells[0], cells[1]
        # Derive collection from name prefix.
        if name.startswith("lora-nim-"):
            collection = "nim_curated"
        elif name.startswith("lora-nemo-usvcs-"):
            collection = "nemo_usvcs_curated"
        else:
            raise ValueError(f"Cannot derive collection from name: {name!r}")
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
#                       "llama-3.3-nemotron-super-70b-v1.5"
#   - 70B is a dense reference target, not the judge. Corpus disambiguation
#     lives in the dataset, not in duplicate target names.

_REFERENCE_BASE_MODEL = os.getenv(
    "EVALUATOR_REFERENCE_MODEL",
    "meta/llama-3.3-70b-instruct",
)


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


def build_reference_target(proxy_url: str) -> dict:
    """Dense reference comparator target.

    Corpus pairing is handled by the evaluation dataset, not by duplicating
    the model target per corpus.
    """
    return build_base_target(_REFERENCE_BASE_MODEL, proxy_url)


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

_JUDGE_MODEL_REF = os.getenv(
    "EVALUATOR_JUDGE_MODEL_REF",
    "default/frontier-judge",
)


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
    context_precision). All are scored by the configured independent judge
    model entity.
    """
    return {
        "name": "stage3-singleaxis-rubric",
        "namespace": "default",
        "description": (
            "Stage 3 single-axis RAGAS — Faithfulness + ResponseRelevancy + "
            "AnswerAccuracy, independent judge"
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
        "description": "Stage 3 pairwise A/B/Tie with position swap, independent judge",
        "type": "custom",
        "params": {
            "parallelism": 4,
            "temperature": 0.0001,  # Evaluator schema requires temperature > 0; greedy-equivalent
            "max_tokens": 8192,     # target inference budget; reasoning models may need room for <think> + answer
            "extra": {
                "judge_model": os.getenv(
                    "EVALUATOR_PAIRWISE_JUDGE_MODEL",
                    "frontier-judge",
                ),
                "judge_endpoint": "http://llm-judge.default.svc.cluster.local:8000/v1/chat/completions",
                "pairwise_prompt": _PAIRWISE_PROMPT,
                "position_swap": True,
            },
        },
    }


# --- adapter inventory from log ---------------------------------------

def load_adapters_from_log(log_path: Path) -> list[AdapterRow]:
    """Parse a generated training-session inventory into AdapterRow entries.

    Walks all table rows starting with `| lora-`, derives collection from the
    name itself, and dedupes by adapter name.
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

def _create_target_idempotent(
    client: EvaluatorClient,
    payload: dict,
    label: str,
    update_existing: bool = False,
) -> None:
    """POST a target; optionally PATCH on 409 when payloads need refresh."""
    import httpx
    try:
        tid = client.create_target(payload)
        log.info("%s target id=%s name=%s", label, tid, payload["name"])
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409:
            if update_existing:
                namespace = payload.get("namespace", "default")
                try:
                    tid = client.update_target(namespace, payload["name"], payload)
                    log.info("%s target updated id=%s name=%s", label, tid, payload["name"])
                except httpx.HTTPStatusError as update_error:
                    if update_error.response.status_code != 501:
                        raise
                    log.warning(
                        "%s target PATCH unsupported; deleting and recreating: %s",
                        label,
                        payload["name"],
                    )
                    client.delete_target(namespace, payload["name"])
                    tid = client.create_target(payload)
                    log.info("%s target recreated id=%s name=%s", label, tid, payload["name"])
            else:
                log.warning("%s target already exists, skipping: %s",
                            label, payload["name"])
        else:
            raise


def _create_config_idempotent(
    client: EvaluatorClient,
    payload: dict,
    update_existing: bool = False,
) -> None:
    """POST a config; optionally PATCH on 409 when payloads need refresh."""
    import httpx
    try:
        cid = client.create_config(payload)
        log.info("config id=%s name=%s", cid, payload["name"])
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409:
            if update_existing:
                namespace = payload.get("namespace", "default")
                try:
                    cid = client.update_config(namespace, payload["name"], payload)
                    log.info("config updated id=%s name=%s", cid, payload["name"])
                except httpx.HTTPStatusError as update_error:
                    if update_error.response.status_code != 501:
                        raise
                    log.warning(
                        "config PATCH unsupported; deleting and recreating: %s",
                        payload["name"],
                    )
                    client.delete_config(namespace, payload["name"])
                    cid = client.create_config(payload)
                    log.info("config recreated id=%s name=%s", cid, payload["name"])
            else:
                log.warning("config already exists, skipping: %s",
                            payload["name"])
        else:
            raise


# --- CLI --------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator-url", default=DEFAULT_EVALUATOR_URL,
                    help="NeMo Evaluator base URL. Defaults to EVALUATOR_URL or "
                         "http://nemo-evaluator:7331.")
    ap.add_argument("--evaluator-api-key", default=os.getenv("EVALUATOR_API_KEY"),
                    help="Optional Evaluator bearer token. Defaults to EVALUATOR_API_KEY.")
    ap.add_argument("--log-path", type=Path, default=DEFAULT_TRAINING_SESSION_LOG,
                    help="Training-session inventory log. Defaults to TRAINING_SESSION_LOG "
                         "or evals/training_session.log.")
    ap.add_argument("--update-existing", action="store_true",
                    help="Refresh existing targets/configs on HTTP 409. Uses PATCH when supported; falls back to delete/recreate on 501.")
    ap.add_argument("--adapter-targets", action="store_true",
                    help="Register the 14 LoRA adapter targets (12 Llama + 2 Nano r=16)")
    ap.add_argument("--base-targets", action="store_true",
                    help="Register dense Llama plus Nano base reference targets")
    ap.add_argument("--reference-target", "--70b-target", dest="reference_target", action="store_true",
                    help="Register the dense reference comparator target")
    ap.add_argument("--configs", action="store_true",
                    help="Register both eval configs (singleaxis + pairwise)")
    ap.add_argument("--all", action="store_true",
                    help="Register adapter targets, dense base targets, the reference comparator, and configs")
    ap.add_argument("--proxy-url", default=DEFAULT_NIM_PROXY_URL,
                    help="OpenAI-compatible model proxy base URL. Defaults to "
                         "NIM_PROXY_URL or http://rag-oai-proxy.runai-rag:8080 "
                         "for the current eval test cluster. Native NIM Proxy can "
                         "be substituted when available.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    register_adapter_targets = args.all or args.adapter_targets
    register_base_targets = args.all or args.base_targets
    register_reference_target = args.all or args.reference_target
    register_configs = args.all or args.configs
    if not any((register_adapter_targets, register_base_targets,
                register_reference_target, register_configs)):
        log.warning("no registration flags selected; use --all or an individual flag")
        return 0

    adapters = load_adapters_from_log(args.log_path)
    log.info("loaded %d adapters from %s", len(adapters), args.log_path)

    # Dense no-LoRA base targets used by the matrix. The 70B-class reference is
    # registered separately to keep comparator naming explicit.
    _BASE_TARGETS = [
        "meta/llama-3.2-1b-instruct",
        "meta/llama-3.2-3b-instruct",
        "meta/llama-3.1-8b-instruct",
    ]

    with EvaluatorClient(args.evaluator_url, api_key=args.evaluator_api_key) as client:
        if register_adapter_targets:
            for a in adapters:
                p = build_adapter_target(a, proxy_url=args.proxy_url)
                _create_target_idempotent(
                    client, p, label="adapter", update_existing=args.update_existing
                )
        if register_base_targets:
            for base in _BASE_TARGETS:
                p = build_base_target(base, proxy_url=args.proxy_url)
                _create_target_idempotent(
                    client, p, label="base", update_existing=args.update_existing
                )
        if register_reference_target:
            p = build_reference_target(proxy_url=args.proxy_url)
            _create_target_idempotent(
                client, p, label="reference", update_existing=args.update_existing
            )
        if register_configs:
            for builder in (build_singleaxis_config, build_pairwise_config):
                p = builder()
                _create_config_idempotent(client, p, update_existing=args.update_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
