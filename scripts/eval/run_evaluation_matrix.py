"""Orchestrate Wave A (single-axis), Wave B (LoRA-vs-LoRA pairwise), and
Wave C (49B-RAG vs LoRA pairwise) Evaluator jobs."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.evaluator_client import EvalJobStatus, EvaluatorClient  # noqa: E402
from scripts.eval.register_evaluator_entities import (  # noqa: E402
    AdapterRow,
    load_adapters_from_log,
)

log = logging.getLogger(__name__)


_DATASET_FOR_COLLECTION = {
    "nim_curated":          "default/stage3-nim-curated-test",
    "nemo_usvcs_curated":   "default/stage3-nemo-usvcs-curated-test",
}

# Bases that are NOT in the no-LoRA base-target sweep (Stage 3 spec: Nano is
# LoRA-only because no non-LoRA Nano baseline was requested).
_BASES_WITHOUT_BASE_TARGET = {"nvidia/nemotron-3-nano-30b-a3b"}


def _base_target_name(base_model: str) -> str:
    """meta/llama-3.2-3b-instruct → default/base-llama-3.2-3b-instruct"""
    return f"default/base-{base_model.split('/',1)[1]}"


def _rag_target_name(collection: str) -> str:
    short = collection.replace("_", "-")
    return f"default/rag-49b-{short}"


def build_singleaxis_jobs(adapters: list[AdapterRow],
                          config_name: str) -> list[dict]:
    """Return list of single-axis job payloads.

    For the canonical Stage 3 inventory (14 adapters across 4 bases, 2 corpora):
      - 14 adapter jobs (one per adapter, evaluated on its corpus's test set)
      -  6 base jobs   (3 bases that have base targets × 2 corpora)
      -  2 RAG jobs    (49B-RAG × 2 corpora)
      = 22 jobs total
    """
    jobs: list[dict] = []
    for a in adapters:
        jobs.append({
            "config": config_name,
            "target": f"default/{a.name}",
            "dataset": _DATASET_FOR_COLLECTION[a.collection],
        })
    seen_bases: set[str] = set()
    for a in adapters:
        if a.base_model in seen_bases or a.base_model in _BASES_WITHOUT_BASE_TARGET:
            continue
        seen_bases.add(a.base_model)
        for coll in ("nim_curated", "nemo_usvcs_curated"):
            jobs.append({
                "config": config_name,
                "target": _base_target_name(a.base_model),
                "dataset": _DATASET_FOR_COLLECTION[coll],
            })
    for coll in ("nim_curated", "nemo_usvcs_curated"):
        jobs.append({
            "config": config_name,
            "target": _rag_target_name(coll),
            "dataset": _DATASET_FOR_COLLECTION[coll],
        })
    return jobs


def build_pairwise_jobs(adapters: list[AdapterRow],
                        config_name: str) -> list[dict]:
    """Wave B — LoRA-vs-LoRA pairwise within each corpus.

    With 7 LoRAs per corpus (3 Llama r=16 + 3 Llama r=32 + 1 Nano r=16),
    that's 7C2=21 pairs per corpus × 2 corpora = 42 jobs.
    """
    by_corpus: dict[str, list[AdapterRow]] = {}
    for a in adapters:
        by_corpus.setdefault(a.collection, []).append(a)
    jobs: list[dict] = []
    for coll, rows in by_corpus.items():
        for a, b in combinations(rows, 2):
            jobs.append({
                "config": config_name,
                # Evaluator job submission requires a `target` field; for
                # pairwise we conventionally set it to target_a. The actual
                # pair lives in `extra.target_a` and `extra.target_b` — the
                # judge config handles the pairwise logic, not this `target`.
                "target": f"default/{a.name}",
                "dataset": _DATASET_FOR_COLLECTION[coll],
                "extra": {
                    "target_a": f"default/{a.name}",
                    "target_b": f"default/{b.name}",
                },
            })
    return jobs


def build_49b_pairwise_jobs(adapters: list[AdapterRow],
                            config_name: str) -> list[dict]:
    """Wave C — 49B-RAG vs every LoRA adapter on the same corpus.

    Per Stage 3 spec, only LoRA-enabled targets are in scope for the 49B
    comparison (non-LoRA bases excluded). With 7 LoRAs per corpus × 2
    corpora = 14 jobs.
    """
    jobs: list[dict] = []
    for a in adapters:
        rag_target = _rag_target_name(a.collection)
        adapter_target = f"default/{a.name}"
        jobs.append({
            "config": config_name,
            "target": rag_target,  # nominal anchor (see pairwise note above)
            "dataset": _DATASET_FOR_COLLECTION[a.collection],
            "extra": {
                "target_a": rag_target,
                "target_b": adapter_target,
            },
        })
    return jobs


def submit_wave(client: EvaluatorClient, jobs: list[dict]) -> list[str]:
    job_ids: list[str] = []
    for j in jobs:
        body = dict(j)  # shallow copy
        job_ids.append(client.submit_job(body))
    return job_ids


def wait_all(client: EvaluatorClient, job_ids: list[str],
             poll_interval: float = 30.0,
             max_wait_s: float = 6 * 3600,
             max_consecutive_errors: int = 10) -> dict[str, EvalJobStatus]:
    """Poll until every job_id is terminal or the deadline expires.

    Per-job consecutive-error counter: if any single job_id fails to poll
    `max_consecutive_errors` times in a row, abort with RuntimeError so a
    misconfigured URL / auth doesn't silently burn the entire deadline.
    A successful poll resets the counter for that job_id.
    """
    deadline = time.monotonic() + max_wait_s
    pending = set(job_ids)
    final: dict[str, EvalJobStatus] = {}
    consecutive_errors: dict[str, int] = {jid: 0 for jid in job_ids}
    while pending and time.monotonic() < deadline:
        for jid in list(pending):
            try:
                s = client.get_status(jid)
                consecutive_errors[jid] = 0
            except Exception as e:
                consecutive_errors[jid] += 1
                log.warning("poll error on %s (%d/%d consecutive): %s",
                            jid, consecutive_errors[jid],
                            max_consecutive_errors, e)
                if consecutive_errors[jid] >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Job {jid} failed {max_consecutive_errors} "
                        f"consecutive polls; aborting wait_all"
                    ) from e
                continue
            if s in {EvalJobStatus.COMPLETED, EvalJobStatus.FAILED,
                     EvalJobStatus.CANCELLED}:
                final[jid] = s
                pending.discard(jid)
        if pending:
            time.sleep(poll_interval)
    if pending:
        raise TimeoutError(f"{len(pending)} jobs not terminal: {sorted(pending)}")
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator-url", default="http://192.168.1.187:30913")
    ap.add_argument("--log-path", type=Path,
                    default=_REPO_ROOT / "evals" / "training_session.log")
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "evals" / "evaluator_job_ids.json")
    ap.add_argument("--wave", choices=["A", "B", "C", "all"], default="all")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    adapters = load_adapters_from_log(args.log_path)
    log.info("loaded %d adapters", len(adapters))

    out_map: dict = {}
    if args.out.exists():
        out_map = json.loads(args.out.read_text())

    with EvaluatorClient(args.evaluator_url) as client:
        if args.wave in ("A", "all"):
            sa_jobs = build_singleaxis_jobs(
                adapters, "default/stage3-singleaxis-rubric"
            )
            log.info("submitting Wave A: %d jobs", len(sa_jobs))
            sa_ids = submit_wave(client, sa_jobs)
            out_map["wave_a"] = list(zip(sa_ids, sa_jobs))
            args.out.write_text(json.dumps(out_map, indent=2))
            log.info("Wave A submitted; polling for terminal state...")
            wait_all(client, sa_ids)
            log.info("Wave A complete")

        if args.wave in ("B", "all"):
            pw_jobs = build_pairwise_jobs(
                adapters, "default/stage3-pairwise-tournament"
            )
            log.info("submitting Wave B: %d jobs", len(pw_jobs))
            pw_ids = submit_wave(client, pw_jobs)
            out_map["wave_b"] = list(zip(pw_ids, pw_jobs))
            args.out.write_text(json.dumps(out_map, indent=2))
            log.info("Wave B submitted; polling for terminal state...")
            wait_all(client, pw_ids)
            log.info("Wave B complete")

        if args.wave in ("C", "all"):
            c_jobs = build_49b_pairwise_jobs(
                adapters, "default/stage3-pairwise-tournament"
            )
            log.info("submitting Wave C (49B-RAG vs LoRA): %d jobs", len(c_jobs))
            c_ids = submit_wave(client, c_jobs)
            out_map["wave_c"] = list(zip(c_ids, c_jobs))
            args.out.write_text(json.dumps(out_map, indent=2))
            log.info("Wave C submitted; polling for terminal state...")
            wait_all(client, c_ids)
            log.info("Wave C complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
