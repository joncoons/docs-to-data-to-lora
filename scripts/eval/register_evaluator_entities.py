"""Register NeMo Evaluator datasets, targets, and configs for Stage 3 eval.

Pure builder functions are unit-tested; the CLI wires them to the real
EvaluatorClient and POSTs. Idempotent (re-running checks for existing
entities).
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
}


@dataclass(frozen=True)
class AdapterRow:
    name: str
    base_model: str
    job_id: str
    collection: str

    @classmethod
    def from_log_line(cls, line: str, collection: str) -> "AdapterRow":
        """Parse one row of evals/training_session.log into AdapterRow.

        Row shape: `| <name> | <job_id> | <train> | <val> | <wallclock> |`
        """
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 2:
            raise ValueError(f"Cannot parse row: {line!r}")
        name, job_id = cells[0], cells[1]
        # Identify base from adapter name suffix: "...-3.2-1b-r16" → 3.2-1b
        m = re.search(r"-(\d\.\d-\d+b)-r\d+$", name)
        if not m:
            raise ValueError(f"Cannot extract base from name: {name!r}")
        size_slug = m.group(1)
        if size_slug not in _BASE_FROM_SIZE_SLUG:
            raise ValueError(f"Unknown base size in name: {name!r}")
        base = _BASE_FROM_SIZE_SLUG[size_slug]
        return cls(name=name, base_model=base, job_id=job_id,
                   collection=collection)


# --- payload builders --------------------------------------------------

def build_adapter_target(row: AdapterRow, nim_url: str) -> dict:
    rank = row.name.rsplit("-r", 1)[1]
    return {
        "name": row.name,
        "namespace": "default",
        "description": (
            f"Stage 3 adapter — {row.collection} × {row.base_model} r{rank}"
        ),
        "type": "model",
        "model": {
            "api_endpoint": {
                "url": f"{nim_url.rstrip('/')}/v1/chat/completions",
                "model_id": row.name,
            },
        },
    }


def build_base_target(base_model: str, nim_url: str) -> dict:
    # "meta/llama-3.2-3b-instruct" → "base-llama-3.2-3b-instruct"
    suffix = base_model.split("/", 1)[1]
    return {
        "name": f"base-{suffix}",
        "namespace": "default",
        "description": f"Unmodified base reference — {base_model}",
        "type": "model",
        "model": {
            "api_endpoint": {
                "url": f"{nim_url.rstrip('/')}/v1/chat/completions",
                "model_id": base_model,
            },
        },
    }


def build_rag_target(collection: str, rag_url: str) -> dict:
    # "nim_curated" → "nim-curated"  →  "rag-49b-nim-curated"
    short = collection.replace("_", "-")
    return {
        "name": f"rag-49b-{short}",
        "namespace": "default",
        "description": (
            f"Nemotron-Super-49B with corpus-scoped RAG against "
            f"{collection} collection"
        ),
        "type": "rag",
        "rag": {
            "api_endpoint": {
                "url": f"{rag_url.rstrip('/')}/api/agent/generate/stream",
            },
            "collection_name": collection,
        },
    }


def build_dataset_payload(collection: str, files_url: str) -> dict:
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
            "temperature": 0.0,
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
            "temperature": 0.0,
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
    """Parse evals/training_session.log into AdapterRow list."""
    rows: list[AdapterRow] = []
    current_corpus: str | None = None
    for line in log_path.read_text().splitlines():
        if "## NIM corpus" in line:
            current_corpus = "nim_curated"
        elif "## NeMo USvcs corpus" in line:
            current_corpus = "nemo_usvcs_curated"
        elif current_corpus and line.startswith("| lora-"):
            try:
                rows.append(AdapterRow.from_log_line(line, current_corpus))
            except ValueError as e:
                log.debug("skipping row %r: %s", line, e)
    return rows


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
    ap.add_argument("--nim-url-1b", default="http://nim-llama-3.2-1b:8000")
    ap.add_argument("--nim-url-3b", default="http://nim-llama-3.2-3b:8000")
    ap.add_argument("--nim-url-8b", default="http://nim-llama-3.1-8b:8000")
    ap.add_argument("--base-url-1b", default="http://nim-llama-3.2-1b-base:8000")
    ap.add_argument("--base-url-3b", default="http://nim-llama-3.2-3b-base:8000")
    ap.add_argument("--base-url-8b", default="http://nim-llama-3.1-8b-base:8000")
    ap.add_argument("--rag-url", default="http://rag-agent-toolkit.runai-rag:8000")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    adapters = load_adapters_from_log(args.log_path)
    log.info("loaded %d adapters from %s", len(adapters), args.log_path)

    nim_url_for = {
        "meta/llama-3.2-1b-instruct": args.nim_url_1b,
        "meta/llama-3.2-3b-instruct": args.nim_url_3b,
        "meta/llama-3.1-8b-instruct": args.nim_url_8b,
    }
    base_url_for = {
        "meta/llama-3.2-1b-instruct": args.base_url_1b,
        "meta/llama-3.2-3b-instruct": args.base_url_3b,
        "meta/llama-3.1-8b-instruct": args.base_url_8b,
    }

    with EvaluatorClient(args.evaluator_url) as client:
        if args.adapter_targets:
            for a in adapters:
                p = build_adapter_target(a, nim_url=nim_url_for[a.base_model])
                try:
                    tid = client.create_target(p)
                    log.info("adapter target id=%s name=%s", tid, p["name"])
                except Exception as e:
                    log.warning("adapter target create failed name=%s: %s",
                                p["name"], e)
        if args.base_targets:
            for base, url in base_url_for.items():
                p = build_base_target(base, nim_url=url)
                try:
                    tid = client.create_target(p)
                    log.info("base target id=%s name=%s", tid, p["name"])
                except Exception as e:
                    log.warning("base target create failed name=%s: %s",
                                p["name"], e)
        if args.rag_targets:
            for coll in ("nim_curated", "nemo_usvcs_curated"):
                p = build_rag_target(coll, rag_url=args.rag_url)
                try:
                    tid = client.create_target(p)
                    log.info("rag target id=%s name=%s", tid, p["name"])
                except Exception as e:
                    log.warning("rag target create failed name=%s: %s",
                                p["name"], e)
        if args.configs:
            for builder in (build_singleaxis_config, build_pairwise_config):
                p = builder()
                try:
                    cid = client.create_config(p)
                    log.info("config id=%s name=%s", cid, p["name"])
                except Exception as e:
                    log.warning("config create failed name=%s: %s",
                                p["name"], e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
