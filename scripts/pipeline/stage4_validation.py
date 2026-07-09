"""Stage 4: external-judge validation gate."""
from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

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


def validate_pair(row: KVPRow, judge: ExternalJudge) -> dict:
    raw = judge.grade(JUDGE_SYSTEM, JUDGE_USER.format(
        question=row.question, answer=row.answer, context=row.context,
    ))
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


def run_stage4(train_rows: list[KVPRow], judge: ExternalJudge,
               output_dir: Path, collection: str, sample_size: int = 100,
               threshold: float = 0.9) -> dict:
    sample = sample_for_validation(train_rows, n=sample_size)
    log.info("Stage 4: sampling %d of %d for validation", len(sample), len(train_rows))

    results = []
    failures = []
    grounded_n = fidelity_n = no_hall_n = all_three_n = 0
    for r in sample:
        g = validate_pair(r, judge)
        results.append(g)
        if g["grounded"]: grounded_n += 1
        if g["answer_fidelity"]: fidelity_n += 1
        if g["no_hallucination"]: no_hall_n += 1
        if g["all_three"]:
            all_three_n += 1
        else:
            failures.append({"question": r.question, "answer": r.answer,
                             "reason": g["reason"]})

    pass_rate = all_three_n / len(sample) if sample else 0.0
    report = {
        "collection": collection,
        "sample_size": len(sample),
        "grounded_count": grounded_n,
        "answer_fidelity_count": fidelity_n,
        "no_hallucination_count": no_hall_n,
        "all_three_count": all_three_n,
        "pass_rate": round(pass_rate, 3),
        "threshold": threshold,
        "passed": pass_rate >= threshold,
        "failures": failures,
    }
    with (output_dir / "validation_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    log.info("Stage 4: pass_rate=%.2f%% (threshold=%.0f%%), passed=%s",
             pass_rate * 100, threshold * 100, report["passed"])
    return report
