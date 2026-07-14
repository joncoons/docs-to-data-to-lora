#!/usr/bin/env python3
"""Roll up token consumption from evaluation summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens_raw",
    "completion_tokens_cleaned_est",
    "think_tokens_est",
    "total_tokens_raw",
)


@dataclass(frozen=True)
class SummaryRef:
    eval_type: str
    path: Path


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def token_totals(section: dict[str, Any] | None) -> dict[str, int]:
    section = section or {}
    return {field: _coerce_int(section.get(field)) for field in TOKEN_FIELDS}


def add_token_totals(target: dict[str, int], source: dict[str, int]) -> None:
    for field in TOKEN_FIELDS:
        target[field] += _coerce_int(source.get(field))


def pairwise_target_totals(section: dict[str, Any] | None) -> dict[str, int]:
    section = section or {}
    totals = {field: 0 for field in TOKEN_FIELDS}
    for side in ("left", "right"):
        add_token_totals(totals, token_totals(section.get(side) or {}))
    if not totals["total_tokens_raw"]:
        totals["total_tokens_raw"] = _coerce_int(section.get("combined_total_tokens_raw"))
    return totals


def parse_summary_metadata(eval_root: Path, summary_path: Path, eval_type: str) -> dict[str, Any]:
    rel = summary_path.parent.relative_to(eval_root)
    parts = rel.parts
    metadata: dict[str, Any] = {
        "eval_type": eval_type,
        "relative_dir": str(rel),
        "summary_path": str(summary_path),
    }
    if eval_type == "singleaxis-llm" and len(parts) >= 7:
        metadata.update(
            {
                "dataset_slug": parts[1],
                "base_slug": parts[2],
                "target_slug": parts[3],
                "rank_slug": parts[4],
                "completion_run_id": parts[5],
                "eval_run_id": parts[6],
            }
        )
    elif eval_type == "pairwise-llm" and len(parts) >= 2:
        metadata["eval_run_id"] = parts[-1]
        metadata["pair_slug"] = "/".join(parts[1:-1])
    else:
        metadata["eval_run_id"] = parts[-1] if parts else None
    return metadata


def summarize_one(eval_root: Path, ref: SummaryRef) -> dict[str, Any]:
    summary = read_json(ref.path)
    metadata = parse_summary_metadata(eval_root, ref.path, ref.eval_type)
    if ref.eval_type == "pairwise-llm":
        row_count = _coerce_int(summary.get("rows_compared"))
        target = pairwise_target_totals(summary.get("target_generation") or {})
    else:
        row_count = _coerce_int(summary.get("rows_scored"))
        target = token_totals(summary.get("target_generation") or {})
    judge = token_totals(summary.get("judge_scoring") or {})
    combined = _coerce_int(summary.get("combined_total_tokens_raw"))
    if not combined:
        combined = target["total_tokens_raw"] + judge["total_tokens_raw"]
    return {
        **metadata,
        "rows_scored_or_compared": row_count,
        "rows_failed": _coerce_int(summary.get("rows_failed")),
        "target_generation": target,
        "judge_scoring": judge,
        "combined_total_tokens_raw": combined,
        "avg_target_tokens_per_row": (
            target["total_tokens_raw"] / row_count if row_count else None
        ),
        "avg_judge_tokens_per_row": (
            judge["total_tokens_raw"] / row_count if row_count else None
        ),
        "avg_combined_tokens_per_row": combined / row_count if row_count else None,
    }


def find_summaries(eval_root: Path, eval_types: list[str], run_id: str | None) -> list[SummaryRef]:
    refs: list[SummaryRef] = []
    for eval_type in eval_types:
        for path in sorted((eval_root / eval_type).glob("**/summary.json")):
            if run_id and f"/{run_id}/" not in f"/{path.parent}/":
                continue
            refs.append(SummaryRef(eval_type=eval_type, path=path))
    return refs


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "rows_scored_or_compared": 0,
        "rows_failed": 0,
        "target_generation": {field: 0 for field in TOKEN_FIELDS},
        "judge_scoring": {field: 0 for field in TOKEN_FIELDS},
        "combined_total_tokens_raw": 0,
    }
    by_eval_type: dict[str, Any] = {}
    for row in rows:
        eval_type = row["eval_type"]
        bucket = by_eval_type.setdefault(
            eval_type,
            {
                "rows_scored_or_compared": 0,
                "rows_failed": 0,
                "target_generation": {field: 0 for field in TOKEN_FIELDS},
                "judge_scoring": {field: 0 for field in TOKEN_FIELDS},
                "combined_total_tokens_raw": 0,
            },
        )
        for target in (totals, bucket):
            target["rows_scored_or_compared"] += row["rows_scored_or_compared"]
            target["rows_failed"] += row["rows_failed"]
            add_token_totals(target["target_generation"], row["target_generation"])
            add_token_totals(target["judge_scoring"], row["judge_scoring"])
            target["combined_total_tokens_raw"] += row["combined_total_tokens_raw"]
    for target in (totals, *by_eval_type.values()):
        rows_count = target["rows_scored_or_compared"]
        target["avg_target_tokens_per_row"] = (
            target["target_generation"]["total_tokens_raw"] / rows_count
            if rows_count else None
        )
        target["avg_judge_tokens_per_row"] = (
            target["judge_scoring"]["total_tokens_raw"] / rows_count
            if rows_count else None
        )
        target["avg_combined_tokens_per_row"] = (
            target["combined_total_tokens_raw"] / rows_count if rows_count else None
        )
    return {"totals": totals, "by_eval_type": by_eval_type}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eval_type",
        "eval_run_id",
        "dataset_slug",
        "base_slug",
        "target_slug",
        "rank_slug",
        "completion_run_id",
        "pair_slug",
        "rows_scored_or_compared",
        "rows_failed",
        "target_total_tokens_raw",
        "judge_prompt_tokens",
        "judge_completion_tokens_raw",
        "judge_total_tokens_raw",
        "combined_total_tokens_raw",
        "avg_judge_tokens_per_row",
        "avg_combined_tokens_per_row",
        "relative_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "eval_type": row.get("eval_type"),
                    "eval_run_id": row.get("eval_run_id"),
                    "dataset_slug": row.get("dataset_slug"),
                    "base_slug": row.get("base_slug"),
                    "target_slug": row.get("target_slug"),
                    "rank_slug": row.get("rank_slug"),
                    "completion_run_id": row.get("completion_run_id"),
                    "pair_slug": row.get("pair_slug"),
                    "rows_scored_or_compared": row["rows_scored_or_compared"],
                    "rows_failed": row["rows_failed"],
                    "target_total_tokens_raw": row["target_generation"]["total_tokens_raw"],
                    "judge_prompt_tokens": row["judge_scoring"]["prompt_tokens"],
                    "judge_completion_tokens_raw": row["judge_scoring"]["completion_tokens_raw"],
                    "judge_total_tokens_raw": row["judge_scoring"]["total_tokens_raw"],
                    "combined_total_tokens_raw": row["combined_total_tokens_raw"],
                    "avg_judge_tokens_per_row": row["avg_judge_tokens_per_row"],
                    "avg_combined_tokens_per_row": row["avg_combined_tokens_per_row"],
                    "relative_dir": row["relative_dir"],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, default=Path("<EVAL_ROOT>"))
    ap.add_argument("--run-id", help="Only include summaries under this eval run id.")
    ap.add_argument(
        "--eval-type",
        action="append",
        choices=["singleaxis-llm", "pairwise-llm"],
        help="Evaluation output type to scan. Defaults to both.",
    )
    ap.add_argument("--out", type=Path, help="Write JSON rollup to this path.")
    ap.add_argument("--csv-out", type=Path, help="Write per-summary CSV rows to this path.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_types = args.eval_type or ["singleaxis-llm", "pairwise-llm"]
    refs = find_summaries(args.eval_root, eval_types, args.run_id)
    rows = [summarize_one(args.eval_root, ref) for ref in refs]
    payload = {
        "schema_version": "eval-token-usage-rollup/v1",
        "eval_root": str(args.eval_root),
        "run_id_filter": args.run_id,
        "summary_count": len(rows),
        **aggregate(rows),
        "rows": rows,
    }
    if args.out:
        write_json(args.out, payload)
    if args.csv_out:
        write_csv(args.csv_out, rows)
    if not args.out:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
