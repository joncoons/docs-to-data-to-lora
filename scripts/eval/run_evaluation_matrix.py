"""Orchestrate Wave A (single-axis), Wave B (base/LoRA pairwise), and
Wave C (49B vs LoRA pairwise) Evaluator jobs."""
from __future__ import annotations

import argparse
import json
import logging
import os
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

DEFAULT_EVALUATOR_URL = os.getenv("EVALUATOR_URL", "http://nemo-evaluator:7331")
DEFAULT_TRAINING_SESSION_LOG = Path(
    os.getenv(
        "TRAINING_SESSION_LOG",
        str(_REPO_ROOT / "evals" / "training_session.log"),
    )
)
DEFAULT_EVALUATOR_JOB_IDS_OUT = Path(
    os.getenv(
        "EVALUATOR_JOB_IDS_OUT",
        str(_REPO_ROOT / "evals" / "evaluator_job_ids.json"),
    )
)


# Test datasets are the context-baked variants (per bake_context_into_testset.py).
# Each row's prompt has retrieved chunks pre-prepended, so every Evaluator call
# sees identical (context + question) input regardless of which target answers.
_DATASET_FOR_COLLECTION = {
    "nim_curated":          "default/stage3-nim-curated-test-with-context",
    "nemo_usvcs_curated":   "default/stage3-nemo-usvcs-curated-test-with-context",
}

# Bases that are NOT in the Wave A no-LoRA base-target sweep. 49B remains a
# separate comparator target in Wave C; Nano now has its own base-only NIM
# deployment and should be evaluated as a base reference target.
_BASES_NOT_IN_WAVE_A_BASE_SWEEP = {
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
}

_NEMOTRON_SUPER_49B_TARGET = "default/llama-3.3-nemotron-super-49b-v1.5"


def _base_target_name(base_model: str) -> str:
    """meta/llama-3.2-3b-instruct → default/llama-3.2-3b-instruct"""
    return f"default/{base_model.split('/', 1)[1]}"


def build_singleaxis_jobs(adapters: list[AdapterRow],
                          config_name: str) -> list[dict]:
    """Wave A single-axis: every LoRA + every dense Llama base, scored on the
    matching-corpus -with-context test set. 49B is intentionally excluded from
    single-axis per the 2026-05-27 redesign (it only appears in Wave C pairwise).

    For the Stage 3 inventory (14 adapters across 4 bases, 2 corpora):
      - adapter jobs (one per adapter, paired with its corpus's test set)
      - base jobs (dense Llama plus Nano base targets × 2 corpora)
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
        if a.base_model in seen_bases or a.base_model in _BASES_NOT_IN_WAVE_A_BASE_SWEEP:
            continue
        seen_bases.add(a.base_model)
        for coll in ("nim_curated", "nemo_usvcs_curated"):
            jobs.append({
                "config": config_name,
                "target": _base_target_name(a.base_model),
                "dataset": _DATASET_FOR_COLLECTION[coll],
            })
    return jobs


def build_pairwise_jobs(adapters: list[AdapterRow],
                        config_name: str) -> list[dict]:
    """Wave B — within-family pairwise checks for each corpus.

    Phase 2 needs two gates:
      - base-vs-adapter to prove the LoRA improved the base model
      - adapter-vs-adapter to compare ranks/variants within the same corpus

    For the full Stage 3 inventory:
      7 adapters per corpus -> 7 base-vs-adapter jobs plus 7C2 LoRA-vs-LoRA
      jobs = 28 jobs per corpus, 56 total.
    """
    by_corpus: dict[str, list[AdapterRow]] = {}
    for a in adapters:
        by_corpus.setdefault(a.collection, []).append(a)
    jobs: list[dict] = []
    for coll, rows in by_corpus.items():
        for a in rows:
            base_target = _base_target_name(a.base_model)
            adapter_target = f"default/{a.name}"
            jobs.append({
                "config": config_name,
                "target": base_target,
                "dataset": _DATASET_FOR_COLLECTION[coll],
                "extra": {
                    "target_a": base_target,
                    "target_b": adapter_target,
                },
            })
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
    """Wave C — single 49B target vs every LoRA adapter, paired with the
    LoRA's matching-corpus -with-context test set.

    Corpus disambiguation lives in the *dataset*, not the target: both Wave C
    rows use the same Nemotron-Super-49B-v1.5 target, but each is paired with
    the test set corresponding to the LoRA's training corpus.

    14 LoRAs × 1 target each (matching dataset by corpus) = 14 jobs.
    """
    jobs: list[dict] = []
    for a in adapters:
        adapter_target = f"default/{a.name}"
        jobs.append({
            "config": config_name,
            "target": _NEMOTRON_SUPER_49B_TARGET,  # nominal anchor (see pairwise note above)
            "dataset": _DATASET_FOR_COLLECTION[a.collection],
            "extra": {
                "target_a": _NEMOTRON_SUPER_49B_TARGET,
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


def write_job_map(path: Path, out_map: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out_map, indent=2) + "\n")


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
    ap.add_argument("--evaluator-url", default=DEFAULT_EVALUATOR_URL,
                    help="NeMo Evaluator base URL. Defaults to EVALUATOR_URL or "
                         "http://nemo-evaluator:7331.")
    ap.add_argument("--evaluator-api-key", default=os.getenv("EVALUATOR_API_KEY"),
                    help="Optional Evaluator bearer token. Defaults to EVALUATOR_API_KEY.")
    ap.add_argument("--log-path", type=Path, default=DEFAULT_TRAINING_SESSION_LOG,
                    help="Training-session inventory log. Defaults to TRAINING_SESSION_LOG "
                         "or evals/training_session.log.")
    ap.add_argument("--out", type=Path, default=DEFAULT_EVALUATOR_JOB_IDS_OUT,
                    help="Path for submitted Evaluator job IDs. Defaults to "
                         "EVALUATOR_JOB_IDS_OUT or evals/evaluator_job_ids.json.")
    ap.add_argument("--wave", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--submit-only", action="store_true",
                    help="Submit jobs and write IDs without polling for terminal status.")
    ap.add_argument("--limit-jobs", type=int, default=0,
                    help="Limit jobs submitted per selected wave. Use 1 for smoke tests.")
    ap.add_argument("--poll-interval", type=float, default=30.0,
                    help="Seconds between Evaluator status polls when not using --submit-only.")
    ap.add_argument("--max-wait-s", type=float, default=6 * 3600,
                    help="Maximum seconds to wait per wave when polling.")
    ap.add_argument("--max-consecutive-errors", type=int, default=10,
                    help="Abort after this many consecutive poll errors for one job.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    adapters = load_adapters_from_log(args.log_path)
    log.info("loaded %d adapters", len(adapters))

    out_map: dict = {}
    if args.out.exists():
        out_map = json.loads(args.out.read_text())

    def submit_and_maybe_wait(wave_key: str, label: str, jobs: list[dict]) -> None:
        if args.limit_jobs > 0:
            jobs = jobs[:args.limit_jobs]
        log.info("submitting %s: %d jobs", label, len(jobs))
        job_ids = submit_wave(client, jobs)
        out_map[wave_key] = list(zip(job_ids, jobs))
        write_job_map(args.out, out_map)
        if args.submit_only:
            log.info("%s submitted; skipping poll due to --submit-only", label)
            return
        log.info("%s submitted; polling for terminal state...", label)
        wait_all(
            client,
            job_ids,
            poll_interval=args.poll_interval,
            max_wait_s=args.max_wait_s,
            max_consecutive_errors=args.max_consecutive_errors,
        )
        log.info("%s complete", label)

    with EvaluatorClient(args.evaluator_url, api_key=args.evaluator_api_key) as client:
        if args.wave in ("A", "all"):
            submit_and_maybe_wait(
                "wave_a",
                "Wave A",
                build_singleaxis_jobs(adapters, "default/stage3-singleaxis-rubric"),
            )

        if args.wave in ("B", "all"):
            submit_and_maybe_wait(
                "wave_b",
                "Wave B",
                build_pairwise_jobs(adapters, "default/stage3-pairwise-tournament"),
            )

        if args.wave in ("C", "all"):
            submit_and_maybe_wait(
                "wave_c",
                "Wave C (49B vs LoRA)",
                build_49b_pairwise_jobs(adapters, "default/stage3-pairwise-tournament"),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
