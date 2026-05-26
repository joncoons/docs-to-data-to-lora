"""Orchestrate Wave A (20 single-axis) + Wave B (30 pairwise) Evaluator jobs."""
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


def _base_target_name(base_model: str) -> str:
    """meta/llama-3.2-3b-instruct → default/base-llama-3.2-3b-instruct"""
    return f"default/base-{base_model.split('/',1)[1]}"


def _rag_target_name(collection: str) -> str:
    short = collection.replace("_", "-")
    return f"default/rag-49b-{short}"


def build_singleaxis_jobs(adapters: list[AdapterRow],
                          config_name: str) -> list[dict]:
    """Return list of 20 single-axis job payloads."""
    jobs: list[dict] = []
    # 12 adapter jobs (each on its corpus dataset)
    for a in adapters:
        jobs.append({
            "config": config_name,
            "target": f"default/{a.name}",
            "dataset": _DATASET_FOR_COLLECTION[a.collection],
        })
    # 3 base targets × 2 corpora = 6 jobs
    seen_bases: set[str] = set()
    for a in adapters:
        if a.base_model in seen_bases:
            continue
        seen_bases.add(a.base_model)
        for coll in ("nim_curated", "nemo_usvcs_curated"):
            jobs.append({
                "config": config_name,
                "target": _base_target_name(a.base_model),
                "dataset": _DATASET_FOR_COLLECTION[coll],
            })
    # 2 RAG jobs
    for coll in ("nim_curated", "nemo_usvcs_curated"):
        jobs.append({
            "config": config_name,
            "target": _rag_target_name(coll),
            "dataset": _DATASET_FOR_COLLECTION[coll],
        })
    return jobs


def build_pairwise_jobs(adapters: list[AdapterRow],
                        config_name: str) -> list[dict]:
    """Return list of 30 pairwise job payloads (15 per corpus)."""
    by_corpus: dict[str, list[AdapterRow]] = {}
    for a in adapters:
        by_corpus.setdefault(a.collection, []).append(a)
    jobs: list[dict] = []
    for coll, rows in by_corpus.items():
        for a, b in combinations(rows, 2):
            jobs.append({
                "config": config_name,
                "target": f"default/{a.name}",  # primary; pair is in extra
                "dataset": _DATASET_FOR_COLLECTION[coll],
                "extra": {
                    "target_a": f"default/{a.name}",
                    "target_b": f"default/{b.name}",
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
             max_wait_s: float = 6 * 3600) -> dict[str, EvalJobStatus]:
    deadline = time.monotonic() + max_wait_s
    pending = set(job_ids)
    final: dict[str, EvalJobStatus] = {}
    while pending and time.monotonic() < deadline:
        for jid in list(pending):
            try:
                s = client.get_status(jid)
            except Exception as e:
                log.warning("poll error on %s: %s", jid, e)
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
    ap.add_argument("--wave", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    adapters = load_adapters_from_log(args.log_path)
    log.info("loaded %d adapters", len(adapters))

    out_map: dict = {}
    if args.out.exists():
        out_map = json.loads(args.out.read_text())

    with EvaluatorClient(args.evaluator_url) as client:
        if args.wave in ("A", "both"):
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

        if args.wave in ("B", "both"):
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
