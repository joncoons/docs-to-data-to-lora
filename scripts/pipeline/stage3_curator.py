"""Stage 3: Curator dedup/quality/split — pure-Python."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import tiktoken

from scripts.pipeline.models import KVPRow

log = logging.getLogger(__name__)
_enc = tiktoken.get_encoding("cl100k_base")


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


def length_filter(rows: list[KVPRow], min_q_tokens: int = 8,
                  min_a_tokens: int = 25) -> list[KVPRow]:
    out: list[KVPRow] = []
    for r in rows:
        if len(_enc.encode(r.question)) < min_q_tokens:
            continue
        if len(_enc.encode(r.answer)) < min_a_tokens:
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


def run_stage3(rows: list[KVPRow], output_dir: Path, system_prompt: str,
               train_ratio: float = 0.9, minhash_threshold: float = 0.85,
               min_q_tokens: int = 8, min_a_tokens: int = 25) -> tuple[list[dict], list[dict]]:
    log.info("Stage 3: input %d rows", len(rows))

    rows = exact_dedup(rows)
    log.info("Stage 3: after exact dedup: %d", len(rows))

    rows = minhash_dedup(rows, threshold=minhash_threshold)
    log.info("Stage 3: after MinHash dedup: %d", len(rows))

    rows = length_filter(rows, min_q_tokens=min_q_tokens, min_a_tokens=min_a_tokens)
    log.info("Stage 3: after length filter: %d", len(rows))

    rows = answer_subset_of_question_filter(rows)
    log.info("Stage 3: after subset filter: %d", len(rows))

    train_rows, val_rows = train_val_split(rows, train_ratio=train_ratio)
    log.info("Stage 3: split %d train / %d val", len(train_rows), len(val_rows))

    train_out = to_customizer_format(train_rows, system_prompt)
    val_out = to_customizer_format(val_rows, system_prompt)

    with (output_dir / "training.jsonl").open("w") as f:
        for x in train_out:
            f.write(json.dumps(x) + "\n")
    with (output_dir / "validation.jsonl").open("w") as f:
        for x in val_out:
            f.write(json.dumps(x) + "\n")

    return train_out, val_out
