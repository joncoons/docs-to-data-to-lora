"""Stage 3: Curator dedup/quality/split — pure-Python."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from scripts.pipeline.models import KVPRow
from scripts.pipeline.stage3_tokenizer import default_stage3_tokenizer_name_or_path

log = logging.getLogger(__name__)


def exact_dedup(rows: list[KVPRow]) -> list[KVPRow]:
    seen: set[str] = set()
    out: list[KVPRow] = []
    for r in rows:
        key = r.question.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def minhash_dedup(rows: list[KVPRow], threshold: float = 0.85,
                  num_perm: int = 128) -> list[KVPRow]:
    """MinHash + LSH dedup on question + answer. CPU via datasketch."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log.warning("datasketch not installed; skipping MinHash dedup")
        return rows

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    keep = []

    for i, r in enumerate(rows):
        m = MinHash(num_perm=num_perm)
        text = (r.question + " " + r.answer).lower()
        for token in text.split():
            m.update(token.encode("utf-8"))
        minhashes.append(m)
        matches = lsh.query(m)
        if matches:
            keep.append(False)
        else:
            keep.append(True)
            lsh.insert(str(i), m)

    return [r for r, k in zip(rows, keep) if k]


def load_stage3_tokenizer(tokenizer_name_or_path: str | None = None):
    tokenizer_name_or_path = (
        tokenizer_name_or_path or default_stage3_tokenizer_name_or_path()
    )
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - fail loudly with the configured tokenizer.
        raise RuntimeError(
            "Stage 3 token filtering requires the production tokenizer. "
            f"Could not load {tokenizer_name_or_path!r} locally. "
            "Set PIPELINE_STAGE3_TOKENIZER to an explicit tokenizer directory. "
            "If needed, download tokenizer artifacts with "
            "scripts/pipeline/download_stage3_tokenizer.py."
        ) from exc


def length_filter(
    rows: list[KVPRow],
    min_q_tokens: int = 12,
    min_a_tokens: int = 8,
    tokenizer=None,
) -> list[KVPRow]:
    tokenizer = tokenizer or load_stage3_tokenizer()
    out: list[KVPRow] = []
    for r in rows:
        if len(tokenizer.encode(r.question, add_special_tokens=False)) < min_q_tokens:
            continue
        if len(tokenizer.encode(r.answer, add_special_tokens=False)) < min_a_tokens:
            continue
        out.append(r)
    return out


def answer_subset_of_question_filter(rows: list[KVPRow]) -> list[KVPRow]:
    out: list[KVPRow] = []
    for r in rows:
        norm_a = re.sub(r"\W+", "", r.answer.lower())
        norm_q = re.sub(r"\W+", "", r.question.lower())
        if norm_a and norm_a in norm_q:
            continue
        out.append(r)
    return out


def train_val_split(rows: list[KVPRow], train_ratio: float = 0.9,
                    seed: int = 42) -> tuple[list[KVPRow], list[KVPRow]]:
    """Random split stratified by stage."""
    rng = random.Random(seed)
    by_stage: dict[str, list[KVPRow]] = defaultdict(list)
    for r in rows:
        by_stage[r.stage].append(r)

    train: list[KVPRow] = []
    val: list[KVPRow] = []
    for stage, stage_rows in by_stage.items():
        rng.shuffle(stage_rows)
        n_train = int(len(stage_rows) * train_ratio)
        train.extend(stage_rows[:n_train])
        val.extend(stage_rows[n_train:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def to_customizer_format(rows: list[KVPRow], system_prompt: str) -> list[dict]:
    return [
        {"prompt": r.question, "completion": r.answer, "system": system_prompt}
        for r in rows
    ]


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def _reduction_step(
    name: str,
    cause: str,
    before: int,
    after: int,
    input_rows: int,
) -> dict[str, int | float | str]:
    removed = before - after
    return {
        "name": name,
        "cause": cause,
        "before_rows": before,
        "after_rows": after,
        "removed_rows": removed,
        "removed_pct_of_step_input": _pct(removed, before),
        "removed_pct_of_stage_input": _pct(removed, input_rows),
    }


def build_stage3_reduction_summary(
    *,
    input_rows: int,
    after_exact_dedup: int,
    after_minhash_dedup: int,
    after_length_filter: int,
    after_subset_filter: int,
    training_rows: int,
    validation_rows: int,
    train_ratio: float,
    minhash_threshold: float,
    min_question_tokens: int,
    min_answer_tokens: int,
    tokenizer_name_or_path: str,
    tokenizer_class: str,
) -> dict[str, object]:
    reductions = [
        _reduction_step(
            "exact_question_dedup",
            "Duplicate questions after case/whitespace normalization",
            input_rows,
            after_exact_dedup,
            input_rows,
        ),
        _reduction_step(
            "minhash_fuzzy_dedup",
            "Near-duplicate question+answer pairs by MinHash LSH",
            after_exact_dedup,
            after_minhash_dedup,
            input_rows,
        ),
        _reduction_step(
            "length_filter",
            (
                f"Question tokens < {min_question_tokens} or "
                f"answer tokens < {min_answer_tokens}"
            ),
            after_minhash_dedup,
            after_length_filter,
            input_rows,
        ),
        _reduction_step(
            "answer_subset_filter",
            "Answer text normalizes to a substring of the question",
            after_length_filter,
            after_subset_filter,
            input_rows,
        ),
    ]
    removed_total = input_rows - after_subset_filter
    return {
        "stage": "3",
        "execution_surface": "python_curator_fallback",
        "parameters": {
            "train_ratio": train_ratio,
            "minhash_threshold": minhash_threshold,
            "min_question_tokens": min_question_tokens,
            "min_answer_tokens": min_answer_tokens,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "tokenizer_class": tokenizer_class,
        },
        "counts": {
            "input_rows": input_rows,
            "after_exact_dedup": after_exact_dedup,
            "after_minhash_dedup": after_minhash_dedup,
            "after_length_filter": after_length_filter,
            "after_subset_filter": after_subset_filter,
            "training_rows": training_rows,
            "validation_rows": validation_rows,
        },
        "reductions": reductions,
        "retention": {
            "final_curated_rows": after_subset_filter,
            "removed_rows": removed_total,
            "retained_pct_of_input": _pct(after_subset_filter, input_rows),
            "removed_pct_of_input": _pct(removed_total, input_rows),
        },
        "split": {
            "training_rows": training_rows,
            "validation_rows": validation_rows,
            "note": "Train/validation split is not a Curator reduction cause.",
        },
    }


def write_stage3_reduction_summary(output_dir: Path, summary: dict[str, object]) -> None:
    json_path = output_dir / "stage3_curator_summary.json"
    md_path = output_dir / "stage3_curator_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    counts = summary["counts"]
    retention = summary["retention"]
    lines = [
        "# Stage 3 Curator Reduction Summary",
        "",
        "Stage 3 applies structural curation after Stage 2 QA admission. "
        "The reductions below explain row loss before the train/validation split.",
        "",
        "## Parameters",
        "",
    ]
    for key, value in summary["parameters"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Reduction Steps",
        "",
        "| Step | Cause | Before | After | Removed | Removed % of Step | Removed % of Input |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for step in summary["reductions"]:
        lines.append(
            "| {name} | {cause} | {before_rows} | {after_rows} | {removed_rows} | "
            "{removed_pct_of_step_input:.2f}% | "
            "{removed_pct_of_stage_input:.2f}% |".format(**step)
        )
    lines.extend([
        "",
        "## Final Counts",
        "",
        f"- Stage 3 input rows: `{counts['input_rows']}`",
        f"- Final curated rows before split: `{retention['final_curated_rows']}`",
        f"- Total removed before split: `{retention['removed_rows']}` "
        f"(`{retention['removed_pct_of_input']:.2f}%` of input)",
        f"- Retained before split: `{retention['retained_pct_of_input']:.2f}%`",
        f"- Training rows: `{counts['training_rows']}`",
        f"- Validation rows: `{counts['validation_rows']}`",
        "",
        "The train/validation split is recorded for completeness but is not a row-loss cause.",
    ])
    md_path.write_text("\n".join(lines) + "\n")


def run_stage3(
    rows: list[KVPRow],
    output_dir: Path,
    system_prompt: str,
    train_ratio: float = 0.9,
    minhash_threshold: float = 0.85,
    min_q_tokens: int = 12,
    min_a_tokens: int = 8,
    tokenizer_name_or_path: str | None = None,
) -> tuple[list[dict], list[dict]]:
    tokenizer_name_or_path = (
        tokenizer_name_or_path or default_stage3_tokenizer_name_or_path()
    )
    tokenizer = load_stage3_tokenizer(tokenizer_name_or_path)
    input_rows = len(rows)
    log.info("Stage 3: input %d rows", input_rows)
    log.info(
        "Stage 3: tokenizer=%s (%s)",
        tokenizer_name_or_path,
        type(tokenizer).__name__,
    )

    rows = exact_dedup(rows)
    after_exact_dedup = len(rows)
    log.info("Stage 3: after exact dedup: %d", after_exact_dedup)

    rows = minhash_dedup(rows, threshold=minhash_threshold)
    after_minhash_dedup = len(rows)
    log.info("Stage 3: after MinHash dedup: %d", after_minhash_dedup)

    rows = length_filter(
        rows,
        min_q_tokens=min_q_tokens,
        min_a_tokens=min_a_tokens,
        tokenizer=tokenizer,
    )
    after_length_filter = len(rows)
    log.info("Stage 3: after length filter: %d", after_length_filter)

    rows = answer_subset_of_question_filter(rows)
    after_subset_filter = len(rows)
    log.info("Stage 3: after subset filter: %d", after_subset_filter)

    train_rows, val_rows = train_val_split(rows, train_ratio=train_ratio)
    log.info("Stage 3: split %d train / %d val", len(train_rows), len(val_rows))

    summary = build_stage3_reduction_summary(
        input_rows=input_rows,
        after_exact_dedup=after_exact_dedup,
        after_minhash_dedup=after_minhash_dedup,
        after_length_filter=after_length_filter,
        after_subset_filter=after_subset_filter,
        training_rows=len(train_rows),
        validation_rows=len(val_rows),
        train_ratio=train_ratio,
        minhash_threshold=minhash_threshold,
        min_question_tokens=min_q_tokens,
        min_answer_tokens=min_a_tokens,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_class=type(tokenizer).__name__,
    )
    write_stage3_reduction_summary(output_dir, summary)

    train_out = to_customizer_format(train_rows, system_prompt)
    val_out = to_customizer_format(val_rows, system_prompt)

    with (output_dir / "training.jsonl").open("w") as f:
        for x in train_out:
            f.write(json.dumps(x) + "\n")
    with (output_dir / "validation.jsonl").open("w") as f:
        for x in val_out:
            f.write(json.dumps(x) + "\n")

    return train_out, val_out
