"""Stage 4: external-judge validation gate."""
from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.pipeline.external_judge_client import ExternalJudge
from scripts.pipeline.models import KVPRow
from scripts.pipeline.prompts import JUDGE_SYSTEM, JUDGE_USER

log = logging.getLogger(__name__)


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()


def parse_judge_response(raw: str) -> dict:
    s = _strip_fences(raw)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return {"grounded": False, "answer_fidelity": False,
                    "no_hallucination": False, "all_three": False,
                    "reason": "JSON parse failed"}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"grounded": False, "answer_fidelity": False,
                    "no_hallucination": False, "all_three": False,
                    "reason": "JSON parse failed"}
    g = bool(obj.get("grounded", False))
    a = bool(obj.get("answer_fidelity", False))
    h = bool(obj.get("no_hallucination", False))
    return {
        "grounded": g,
        "answer_fidelity": a,
        "no_hallucination": h,
        "all_three": g and a and h,
        "reason": obj.get("reason", ""),
    }


def _customizer_key(question: str, answer: str) -> tuple[str, str]:
    return question.strip(), answer.strip()


def join_customizer_rows_to_kvp(
    customizer_rows: list[dict[str, Any]],
    source_rows: list[KVPRow],
) -> list[KVPRow]:
    """Restore KVP metadata/context for finalized Customizer-format rows.

    Stage 3 writes ``training.jsonl``/``validation.jsonl`` in Customizer format,
    which intentionally drops source context. Stage 4 needs the exact finalized
    rows plus source context, so it joins prompt/completion pairs back to the
    admitted Stage 2 KVP rows.
    """
    by_key: dict[tuple[str, str], list[KVPRow]] = defaultdict(list)
    for row in source_rows:
        by_key[_customizer_key(row.question, row.answer)].append(row)

    joined: list[KVPRow] = []
    missing: list[dict[str, Any]] = []
    for index, row in enumerate(customizer_rows):
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        matches = by_key.get(_customizer_key(prompt, completion), [])
        if not matches:
            missing.append({
                "index": index,
                "prompt": prompt[:200],
                "completion": completion[:200],
            })
            continue
        joined.append(matches[0])

    if missing:
        preview = json.dumps(missing[:5], ensure_ascii=False)
        raise ValueError(
            f"Could not restore context for {len(missing)} finalized rows; "
            f"first missing rows: {preview}"
        )
    return joined


def validate_pair(row: KVPRow, judge: ExternalJudge, max_tokens: int = 512) -> dict:
    raw = judge.grade(
        JUDGE_SYSTEM,
        JUDGE_USER.format(
            question=row.question,
            answer=row.answer,
            context=row.context,
        ),
        max_tokens=max_tokens,
    )
    if not raw:
        return {"grounded": False, "answer_fidelity": False,
                "no_hallucination": False, "all_three": False,
                "reason": "judge call returned no content"}
    return parse_judge_response(raw)


def sample_for_validation(rows: list[KVPRow], n: int, seed: int = 42) -> list[KVPRow]:
    rng = random.Random(seed)
    by_stage: dict[str, list[KVPRow]] = defaultdict(list)
    for r in rows:
        by_stage[r.stage].append(r)
    sample: list[KVPRow] = []
    total = len(rows)
    for stage, stage_rows in by_stage.items():
        share = max(1, round(len(stage_rows) / total * n))
        sample.extend(rng.sample(stage_rows, min(share, len(stage_rows))))
    rng.shuffle(sample)
    return sample[:n]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _sample_record(row: KVPRow, ordinal: int) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "sample_id": row.sample_id,
        "passage_id": row.passage_id,
        "source_url": row.source_url,
        "stage": row.stage,
        "qa_type": row.qa_type,
        "instr_type": row.instr_type,
        "question": row.question,
        "answer": row.answer,
        "context": row.context,
    }


def run_stage4(
    train_rows: list[KVPRow],
    judge: ExternalJudge,
    output_dir: Path,
    collection: str,
    sample_size: int = 100,
    threshold: float = 0.9,
    seed: int = 42,
    max_tokens: int = 512,
    judge_metadata: dict[str, Any] | None = None,
) -> dict:
    sample = sample_for_validation(train_rows, n=sample_size, seed=seed)
    log.info("Stage 4: sampling %d of %d for validation", len(sample), len(train_rows))

    sample_records = [_sample_record(row, ordinal) for ordinal, row in enumerate(sample)]
    _write_jsonl(output_dir / "validation_sample.jsonl", sample_records)

    judgments = []
    failures = []
    grounded_n = fidelity_n = no_hall_n = all_three_n = 0
    for ordinal, r in enumerate(sample):
        g = validate_pair(r, judge, max_tokens=max_tokens)
        judgment = {
            **_sample_record(r, ordinal),
            "grounded": g["grounded"],
            "answer_fidelity": g["answer_fidelity"],
            "no_hallucination": g["no_hallucination"],
            "all_three": g["all_three"],
            "reason": g["reason"],
        }
        judgments.append(judgment)
        if g["grounded"]:
            grounded_n += 1
        if g["answer_fidelity"]:
            fidelity_n += 1
        if g["no_hallucination"]:
            no_hall_n += 1
        if g["all_three"]:
            all_three_n += 1
        else:
            failures.append({
                "ordinal": ordinal,
                "sample_id": r.sample_id,
                "passage_id": r.passage_id,
                "source_url": r.source_url,
                "stage": r.stage,
                "question": r.question,
                "answer": r.answer,
                "reason": g["reason"],
            })

    _write_jsonl(output_dir / "validation_judgments.jsonl", judgments)

    pass_rate = all_three_n / len(sample) if sample else 0.0
    report = {
        "collection": collection,
        "sample_size": len(sample),
        "sample_seed": seed,
        "source_rows": len(train_rows),
        "grounded_count": grounded_n,
        "answer_fidelity_count": fidelity_n,
        "no_hallucination_count": no_hall_n,
        "all_three_count": all_three_n,
        "pass_rate": round(pass_rate, 3),
        "threshold": threshold,
        "passed": pass_rate >= threshold,
        "judge": judge_metadata or {},
        "sample_path": "validation_sample.jsonl",
        "judgments_path": "validation_judgments.jsonl",
        "failures": failures,
    }
    with (output_dir / "validation_report.json").open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    log.info("Stage 4: pass_rate=%.2f%% (threshold=%.0f%%), passed=%s",
             pass_rate * 100, threshold * 100, report["passed"])
    return report
