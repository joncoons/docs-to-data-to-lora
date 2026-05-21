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
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `scripts.pipeline.*` is importable
# when this script is run directly (e.g. `python scripts/build_v2_dataset.py`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.pipeline.claude_client import ClaudeJudge
from scripts.pipeline.config import Config, get_claude_api_key, get_es_password
from scripts.pipeline.es_client import make_es_client
from scripts.pipeline.llm_client import LLMClient
from scripts.pipeline.models import KVPRow, Passage
from scripts.pipeline.progress import Progress
from scripts.pipeline.stage0_corpus_prep import run_stage0
from scripts.pipeline.stage1a_le_kvp import run_stage1a
from scripts.pipeline.stage1b_synthesis import run_stage1b
from scripts.pipeline.stage1c_instruction import run_stage1c
from scripts.pipeline.stage1_5_gapfill import run_stage1_5
from scripts.pipeline.stage2_qa_eval import run_stage2
from scripts.pipeline.stage3_curator import run_stage3
from scripts.pipeline.stage4_validation import run_stage4


def _read_jsonl_rows(path: Path) -> list[KVPRow]:
    if not path.exists():
        return []
    return [KVPRow.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _read_passages(path: Path) -> list[Passage]:
    if not path.exists():
        return []
    return [Passage.model_validate_json(line) for line in path.read_text().splitlines() if line]


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
    args = ap.parse_args()

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
    )
    llm_nothink = LLMClient(
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

    # --- Stage 1A ---
    if args.stage in ("1a", "all"):
        if args.resume and progress.is_done("1a"):
            stage1a_rows = _read_jsonl_rows(args.output / "stage1a_le.jsonl")
        else:
            stage1a_rows = run_stage1a(passages, llm, args.output,
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
                                        max_context_tokens=cfg.knn_max_context_tokens)
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
                                        max_workers=cfg.max_workers)
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
                                          llm_nothink, args.output,
                                          threshold_factor=cfg.bias_threshold_factor,
                                          target_factor=cfg.bias_gapfill_target_factor,
                                          top_n_chunks=cfg.gapfill_top_n_chunks,
                                          pairs_per_call=cfg.gapfill_pairs_per_call,
                                          max_attempt_factor=cfg.gapfill_max_attempt_factor)
            progress.mark_done("1.5")
    else:
        stage1_5_rows = _read_jsonl_rows(args.output / "stage1_5_gapfill.jsonl")

    # --- Stage 2 ---
    all_pre_eval = stage1a_rows + stage1b_rows + stage1c_rows + stage1_5_rows
    if args.stage in ("2", "all"):
        if args.resume and progress.is_done("2"):
            stage2_rows = _read_jsonl_rows(args.output / "stage2_eval.jsonl")
        else:
            stage2_rows, _ = run_stage2(all_pre_eval, llm, args.output,
                                         max_workers=cfg.max_workers)
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
            judge = ClaudeJudge(
                base_url=cfg.claude_base_url,
                api_key=get_claude_api_key(),
                model=cfg.claude_model,
            )
            run_stage4(stage2_rows, judge, args.output, args.collection,
                       sample_size=cfg.judge_sample_size,
                       threshold=cfg.judge_pass_threshold)
            progress.mark_done("4")

    log.info("Done. Output dir: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
