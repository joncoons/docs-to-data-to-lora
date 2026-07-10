#!/usr/bin/env python3
"""Build Stage 2 SFT dataset from an ES collection.

Usage:
  python scripts/build_v2_dataset.py --collection nim_curated \\
      --output /mnt/nvme2/peft/datasets/v2/nim_curated
  python scripts/build_v2_dataset.py --collection nemo_usvcs_curated \\
      --output /mnt/nvme2/peft/datasets/v2/nemo_usvcs_curated --stage all
  python scripts/build_v2_dataset.py --collection nim_curated --resume
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `scripts.pipeline.*` is importable
# when this script is run directly (e.g. `python scripts/build_v2_dataset.py`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.pipeline.external_judge_client import ExternalJudge  # noqa: E402
from scripts.pipeline.config import (  # noqa: E402
    Config,
    get_es_password,
    get_external_judge_api_key,
)
from scripts.pipeline.dataset_admission import (  # noqa: E402
    admitted_dataset_samples_from_kvp_rows,
)
from scripts.pipeline.es_client import make_es_client  # noqa: E402
from scripts.pipeline.finalize_dataset import finalize_dataset  # noqa: E402
from scripts.pipeline.llm_client import LLMClient  # noqa: E402
from scripts.pipeline.models import KVPRow, Passage  # noqa: E402
from scripts.pipeline.progress import Progress  # noqa: E402
from scripts.pipeline.provenance_io import write_jsonl  # noqa: E402
from scripts.pipeline.stage0_corpus_prep import run_stage0  # noqa: E402
from scripts.pipeline.stage1a_batched_kvp import run_stage1a_batched  # noqa: E402
from scripts.pipeline.stage1a_le_kvp import run_stage1a  # noqa: E402
from scripts.pipeline.stage1b_synthesis import run_stage1b  # noqa: E402
from scripts.pipeline.stage1c_instruction import run_stage1c  # noqa: E402
from scripts.pipeline.stage1_5_gapfill import run_stage1_5  # noqa: E402
from scripts.pipeline.stage2_qa_eval import run_stage2  # noqa: E402
from scripts.pipeline.stage3_curator import run_stage3  # noqa: E402
from scripts.pipeline.stage4_validation import run_stage4  # noqa: E402


DEFAULT_STAGE1A_BATCHED_TEMPERATURE = 0.95


def _read_jsonl_rows(path: Path) -> list[KVPRow]:
    if not path.exists():
        return []
    return [KVPRow.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _read_passages(path: Path) -> list[Passage]:
    if not path.exists():
        return []
    return [Passage.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _parse_endpoints(value: str, label: str = "inference") -> list[str]:
    endpoints = [endpoint.strip() for endpoint in value.split(",") if endpoint.strip()]
    if not endpoints:
        raise ValueError(f"At least one {label} endpoint is required")
    return endpoints


def _endpoint_requires_api_key(endpoints: list[str]) -> bool:
    return any(
        "inference-api.nvidia.com" in endpoint
        or "integrate.api.nvidia.com" in endpoint
        for endpoint in endpoints
    )


def _build_stage1a_llm(args: argparse.Namespace, cfg: Config) -> LLMClient:
    if args.stage1a_nim_endpoints:
        endpoints = _parse_endpoints(args.stage1a_nim_endpoints, "Stage 1A")
    else:
        endpoints = cfg.nim_endpoints

    if args.stage1a_model:
        model = args.stage1a_model
    else:
        model = cfg.super120b_model

    if args.stage1a_temperature is not None:
        temperature = args.stage1a_temperature
    elif args.stage1a_mode == "batched":
        temperature = DEFAULT_STAGE1A_BATCHED_TEMPERATURE
    else:
        temperature = 0.2

    api_key = args.stage1a_api_key
    if api_key is None and _endpoint_requires_api_key(endpoints):
        api_key = get_external_judge_api_key()
    return LLMClient(
        endpoints=endpoints,
        model=model,
        api_key=api_key or "local",
        max_workers=cfg.max_workers,
        min_interval_s=cfg.min_request_interval_s,
        retry_attempts=cfg.retry_attempts,
        retry_base_delay_s=cfg.retry_base_delay_s,
        no_think=True,
        temperature=temperature,
    )


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None:
        return None
    return float(value)


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    return int(value)


def _build_stage2_llm(args: argparse.Namespace, cfg: Config) -> LLMClient:
    endpoints = (
        _parse_endpoints(args.stage2_qa_endpoints, "Stage 2 QA")
        if args.stage2_qa_endpoints
        else cfg.stage2_qa_endpoints
    )
    model = args.stage2_qa_model or cfg.stage2_qa_model
    temperature = (
        args.stage2_qa_temperature
        if args.stage2_qa_temperature is not None
        else cfg.stage2_qa_temperature
    )
    api_key = args.stage2_qa_api_key
    if api_key is None and _endpoint_requires_api_key(endpoints):
        api_key = get_external_judge_api_key()
    return LLMClient(
        endpoints=endpoints,
        model=model,
        api_key=api_key or "local",
        max_workers=cfg.max_workers,
        min_interval_s=cfg.min_request_interval_s,
        retry_attempts=cfg.retry_attempts,
        retry_base_delay_s=cfg.retry_base_delay_s,
        no_think=True,
        temperature=temperature,
    )


def _needs_stage1a_llm_override(args: argparse.Namespace) -> bool:
    if args.stage1a_mode == "batched":
        return True
    return any(
        value is not None
        for value in (
            args.stage1a_nim_endpoints,
            args.stage1a_model,
            args.stage1a_api_key,
            args.stage1a_temperature,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", required=True,
                    choices=["nim_curated", "nemo_usvcs_curated"])
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--stage", default="all",
                    choices=["0", "1a", "1b", "1c", "1.5", "2", "3", "4", "all"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + estimated yields, no LLM calls")
    ap.add_argument("--stage1-5-mode", choices=["manifest", "legacy-direct"],
                    default="manifest",
                    help="Stage 1.5 default writes gap_manifest/Data Designer inputs; "
                         "legacy-direct preserves the older direct LLM generator")
    ap.add_argument("--stage1a-mode", choices=["legacy", "batched"],
                    default=os.getenv("PIPELINE_STAGE1A_MODE", "legacy"),
                    help="Stage 1A implementation. legacy preserves one KVP call per "
                         "premise; batched uses conservative batched KVP expansion "
                         "with fallback.")
    ap.add_argument("--stage1a-nim-endpoints",
                    default=os.getenv("PIPELINE_STAGE1A_NIM_ENDPOINTS"),
                    help="Optional comma-separated endpoint override for Stage 1A only.")
    ap.add_argument("--stage1a-model", default=os.getenv("PIPELINE_STAGE1A_MODEL"),
                    help="Optional model override for Stage 1A only, e.g. "
                         "nvidia/nvidia/nemotron-3-super-v3.")
    ap.add_argument("--stage1a-api-key", default=os.getenv("PIPELINE_STAGE1A_API_KEY"),
                    help="Optional API key override for Stage 1A only.")
    ap.add_argument("--stage1a-temperature", type=float,
                    default=_env_optional_float("PIPELINE_STAGE1A_TEMPERATURE"),
                    help="Optional temperature override for Stage 1A only.")
    ap.add_argument("--stage1a-max-premises-per-batch", type=int,
                    default=int(os.getenv("PIPELINE_STAGE1A_MAX_PREMISES_PER_BATCH", "12")),
                    help="Maximum premise/conclusion items per batched Stage 1A KVP call.")
    ap.add_argument("--stage1a-batch-parse-attempts", type=int,
                    default=int(os.getenv("PIPELINE_STAGE1A_BATCH_PARSE_ATTEMPTS", "2")),
                    help="Batched KVP parse attempts before per-premise fallback.")
    ap.add_argument("--stage1a-le-max-tokens", type=int,
                    default=int(os.getenv("PIPELINE_STAGE1A_LE_MAX_TOKENS", "16384")),
                    help="Stage 1A logical-entailment completion budget.")
    ap.add_argument("--stage1a-batched-kvp-max-tokens", type=int,
                    default=int(os.getenv("PIPELINE_STAGE1A_BATCHED_KVP_MAX_TOKENS", "16384")),
                    help="Stage 1A batched KVP completion budget.")
    ap.add_argument("--stage1c-selection-mode",
                    choices=["stratified", "top_density", "all"],
                    default=os.getenv("PIPELINE_STAGE1C_SELECTION_MODE"),
                    help="Stage 1C passage selection mode. Defaults to config/env stratified.")
    ap.add_argument("--stage2-qa-endpoints",
                    default=os.getenv("PIPELINE_STAGE2_QA_ENDPOINTS"),
                    help="Optional comma-separated endpoint override for Stage 2 QA admission.")
    ap.add_argument("--stage2-qa-model", default=os.getenv("PIPELINE_STAGE2_QA_MODEL"),
                    help="Optional model override for Stage 2 QA admission, e.g. "
                         "nvidia/nvidia/nemotron-3-ultra.")
    ap.add_argument("--stage2-qa-api-key", default=os.getenv("PIPELINE_STAGE2_QA_API_KEY"),
                    help="Optional API key override for Stage 2 QA admission.")
    ap.add_argument("--stage2-qa-temperature", type=float,
                    default=_env_optional_float("PIPELINE_STAGE2_QA_TEMPERATURE"),
                    help="Optional temperature override for Stage 2 QA admission.")
    ap.add_argument("--stage2-qa-max-tokens", type=int,
                    default=_env_optional_int("PIPELINE_STAGE2_QA_MAX_TOKENS"),
                    help="Stage 2 QA admission completion budget.")
    ap.add_argument("--stage2-execution-surface",
                    default=os.getenv("PIPELINE_STAGE2_EXECUTION_SURFACE"),
                    help="Audit label for Stage 2 QA execution surface, e.g. "
                         "curator_llm_quality or direct_qa_eval.")
    ap.add_argument("--max-passages", type=int, default=None,
                    help="Subsample to N passages after Stage 0 (for smoke testing)")
    args = ap.parse_args()
    if args.stage1a_max_premises_per_batch < 1:
        ap.error("--stage1a-max-premises-per-batch must be >= 1")
    if args.stage1a_batch_parse_attempts < 1:
        ap.error("--stage1a-batch-parse-attempts must be >= 1")
    if args.stage1a_le_max_tokens < 1:
        ap.error("--stage1a-le-max-tokens must be >= 1")
    if args.stage1a_batched_kvp_max_tokens < 1:
        ap.error("--stage1a-batched-kvp-max-tokens must be >= 1")
    if args.stage2_qa_max_tokens is not None and args.stage2_qa_max_tokens < 1:
        ap.error("--stage2-qa-max-tokens must be >= 1")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("build_v2")

    cfg = Config()
    args.output.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.output / "progress.json")
    domain = "NVIDIA NIM" if args.collection == "nim_curated" else "NVIDIA NeMo Microservices"
    system_prompt = (
        f"You are a precise {domain} technical assistant. "
        "Answer based on official documentation."
    )

    if args.dry_run:
        log.info("DRY RUN — collection=%s output=%s stage=%s",
                 args.collection, args.output, args.stage)
        return 0

    es = make_es_client(cfg.es_host, get_es_password())
    llm = LLMClient(
        endpoints=cfg.nim_endpoints, model=cfg.super120b_model,
        max_workers=cfg.max_workers, min_interval_s=cfg.min_request_interval_s,
        retry_attempts=cfg.retry_attempts, retry_base_delay_s=cfg.retry_base_delay_s,
        no_think=True,
    )

    passages: list[Passage] = []
    seed_vectors: dict[str, list[float]] = {}
    stage1a_rows: list[KVPRow] = []
    stage1b_rows: list[KVPRow] = []
    stage1c_rows: list[KVPRow] = []
    stage1_5_rows: list[KVPRow] = []
    stage2_rows: list[KVPRow] = []

    # --- Stage 0 ---
    if args.stage in ("0", "all"):
        if args.resume and progress.is_done("0"):
            log.info("Stage 0: skipping (already done)")
            passages = _read_passages(args.output / "passages.jsonl")
        else:
            passages, seed_vectors = run_stage0(es, args.collection, args.output,
                                                 min_passage_tokens=cfg.min_passage_tokens)
            progress.mark_done("0")
    else:
        passages = _read_passages(args.output / "passages.jsonl")

    if args.max_passages and args.max_passages < len(passages):
        import random
        random.seed(42)
        passages = random.sample(passages, args.max_passages)
        log.info("Subsampled to %d passages (--max-passages)", len(passages))
        # Rewrite passages.jsonl to reflect the subsample
        with (args.output / "passages.jsonl").open("w") as f:
            for p in passages:
                f.write(p.model_dump_json() + "\n")

    # --- Stage 1A ---
    if args.stage in ("1a", "all"):
        if args.resume and progress.is_done("1a"):
            stage1a_rows = _read_jsonl_rows(args.output / "stage1a_le.jsonl")
        else:
            stage1a_llm = (
                _build_stage1a_llm(args, cfg)
                if _needs_stage1a_llm_override(args)
                else llm
            )
            log.info("Stage 1A: mode=%s model=%s",
                     args.stage1a_mode, stage1a_llm.model)
            if args.stage1a_mode == "batched":
                stage1a_rows = run_stage1a_batched(
                    passages,
                    stage1a_llm,
                    args.output,
                    max_workers=cfg.max_workers,
                    max_premises_per_batch=args.stage1a_max_premises_per_batch,
                    batch_parse_attempts=args.stage1a_batch_parse_attempts,
                    le_max_tokens=args.stage1a_le_max_tokens,
                    batched_kvp_max_tokens=args.stage1a_batched_kvp_max_tokens,
                )
            else:
                stage1a_rows = run_stage1a(passages, stage1a_llm, args.output,
                                           max_workers=cfg.max_workers)
            progress.mark_done("1a")
    else:
        stage1a_rows = _read_jsonl_rows(args.output / "stage1a_le.jsonl")

    # --- Stage 1B ---
    if args.stage in ("1b", "all"):
        if args.resume and progress.is_done("1b"):
            stage1b_rows = _read_jsonl_rows(args.output / "stage1b_synthesis.jsonl")
        else:
            if not seed_vectors:
                log.warning("Stage 1B: re-scrolling ES for seed vectors")
                _, seed_vectors = run_stage0(es, args.collection, args.output,
                                              min_passage_tokens=cfg.min_passage_tokens)
            stage1b_rows = run_stage1b(passages, seed_vectors, es, args.collection,
                                        domain, llm, args.output,
                                        max_workers=cfg.max_workers,
                                        knn_k=cfg.knn_k,
                                        num_candidates=cfg.knn_num_candidates,
                                        top_neighbors=cfg.knn_top_neighbors,
                                        max_context_tokens=cfg.knn_max_context_tokens,
                                        resume=args.resume)
            progress.mark_done("1b")
    else:
        stage1b_rows = _read_jsonl_rows(args.output / "stage1b_synthesis.jsonl")

    # --- Stage 1C ---
    if args.stage in ("1c", "all"):
        if args.resume and progress.is_done("1c"):
            stage1c_rows = _read_jsonl_rows(args.output / "stage1c_instruction.jsonl")
        else:
            stage1c_rows = run_stage1c(passages, domain, llm, args.output,
                                        top_percent=cfg.stage1c_top_percent,
                                        min_passages=cfg.stage1c_min_passages,
                                        max_workers=cfg.max_workers,
                                        selection_mode=args.stage1c_selection_mode or cfg.stage1c_selection_mode,
                                        resume=args.resume)
            progress.mark_done("1c")
    else:
        stage1c_rows = _read_jsonl_rows(args.output / "stage1c_instruction.jsonl")

    # --- Stage 1.5 ---
    if args.stage in ("1.5", "all"):
        if args.resume and progress.is_done("1.5"):
            stage1_5_rows = _read_jsonl_rows(args.output / "stage1_5_gapfill.jsonl")
        else:
            existing = stage1a_rows + stage1b_rows + stage1c_rows
            stage1_5_rows = run_stage1_5(passages, existing, es, args.collection,
                                          llm, args.output,
                                          threshold_factor=cfg.bias_threshold_factor,
                                          target_factor=cfg.bias_gapfill_target_factor,
                                          top_n_chunks=cfg.gapfill_top_n_chunks,
                                          pairs_per_call=cfg.gapfill_pairs_per_call,
                                          max_attempt_factor=cfg.gapfill_max_attempt_factor,
                                          legacy_direct=(
                                              args.stage1_5_mode == "legacy-direct"
                                          ))
            progress.mark_done("1.5")
    else:
        stage1_5_rows = _read_jsonl_rows(args.output / "stage1_5_gapfill.jsonl")

    # --- Stage 2 ---
    all_pre_eval = stage1a_rows + stage1b_rows + stage1c_rows + stage1_5_rows
    if args.stage in ("2", "all"):
        if args.resume and progress.is_done("2"):
            stage2_rows = _read_jsonl_rows(args.output / "stage2_eval.jsonl")
        else:
            stage2_llm = _build_stage2_llm(args, cfg)
            stage2_max_tokens = args.stage2_qa_max_tokens or cfg.stage2_qa_max_tokens
            stage2_execution_surface = args.stage2_execution_surface or cfg.stage2_execution_surface
            log.info(
                "Stage 2: QA admission model=%s endpoints=%s execution_surface=%s",
                stage2_llm.model,
                stage2_llm.endpoints,
                stage2_execution_surface,
            )
            stage2_rows, _ = run_stage2(
                all_pre_eval,
                stage2_llm,
                args.output,
                max_workers=cfg.max_workers,
                resume=args.resume,
                execution_surface=stage2_execution_surface,
                max_tokens=stage2_max_tokens,
            )
            progress.mark_done("2")
    else:
        stage2_rows = _read_jsonl_rows(args.output / "stage2_eval.jsonl")

    # --- Stage 3 ---
    if args.stage in ("3", "all"):
        if args.resume and progress.is_done("3"):
            log.info("Stage 3: skipping (already done)")
        else:
            run_stage3(stage2_rows, args.output, system_prompt,
                       train_ratio=cfg.train_val_split,
                       minhash_threshold=cfg.minhash_threshold,
                       min_q_tokens=cfg.min_question_tokens,
                       min_a_tokens=cfg.min_answer_tokens)
            progress.mark_done("3")

    # --- Stage 4 ---
    if args.stage in ("4", "all"):
        if args.resume and progress.is_done("4"):
            log.info("Stage 4: skipping (already done)")
        else:
            judge = ExternalJudge(
                base_url=cfg.external_judge_base_url,
                api_key=get_external_judge_api_key(),
                model=cfg.external_judge_model,
            )
            run_stage4(stage2_rows, judge, args.output, args.collection,
                       sample_size=cfg.judge_sample_size,
                       threshold=cfg.judge_pass_threshold)
            progress.mark_done("4")

    sample_rows = stage2_rows or all_pre_eval
    if sample_rows:
        write_jsonl(
            args.output / "provenance" / "dataset_samples.jsonl",
            admitted_dataset_samples_from_kvp_rows(
                sample_rows,
                dataset_dir=args.output,
                system_prompt=system_prompt,
            ),
        )
        log.info(
            "Provenance: %d dataset samples → %s",
            len(sample_rows),
            args.output / "provenance" / "dataset_samples.jsonl",
        )
        manifest = finalize_dataset(
            args.output,
            dataset_name=args.collection,
            system_prompt=system_prompt,
            observability_dir=None,
        )
        log.info("Provenance: dataset version %s", manifest["dataset_version_id"])

    log.info("Done. Output dir: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
