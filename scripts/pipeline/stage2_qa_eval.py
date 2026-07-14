"""Stage 2: QA admission/refinement with durable per-row decisions."""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError
from tqdm import tqdm

from scripts.pipeline.llm_client import LLMClient
from scripts.pipeline.models import KVPRow, QAEvaluation
from scripts.pipeline.prompts import QA_EVAL_SYSTEM, QA_EVAL_USER
from scripts.pipeline.provenance import sample_id_for_row, stable_id

log = logging.getLogger(__name__)

TERMINAL_QA_STATUSES = {"accepted", "refined", "dropped"}
RETRYABLE_QA_STATUSES = {"llm_empty", "parse_failed", "exception"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()


def parse_eval_response(raw: str) -> Optional[QAEvaluation]:
    s = _strip_fences(raw)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    try:
        return QAEvaluation(**obj)
    except ValidationError:
        return None


def _row_sample_id(row: KVPRow) -> str:
    return row.sample_id or sample_id_for_row(row)


def stage2_work_id_for_row(row: KVPRow, row_index: int | None = None) -> str:
    """Stable resume key for a Stage 2 input row."""
    return stable_id(
        "stage2qa",
        row_index,
        _row_sample_id(row),
        row.stage,
        row.passage_id,
        row.source_url,
        row.question,
        row.answer,
    )


def _llm_model(llm: LLMClient) -> str:
    return str(getattr(llm, "model", None) or "unknown")


def _llm_endpoints(llm: LLMClient) -> list[str]:
    endpoints = getattr(llm, "endpoints", None)
    if not endpoints:
        return []
    return [str(endpoint) for endpoint in endpoints]


def _qa_updates(
    *,
    row: KVPRow,
    work_id: str,
    status: str,
    admitted: bool,
    llm: LLMClient,
    execution_surface: str,
    ev: QAEvaluation | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": _row_sample_id(row),
        "qa_work_id": work_id,
        "qa_status": status,
        "qa_admitted": admitted,
        "qa_judge_model": _llm_model(llm),
        "qa_judge_endpoints": _llm_endpoints(llm) or None,
        "qa_execution_surface": execution_surface,
        "qa_grounded": ev.grounded if ev else None,
        "qa_answer_fidelity": ev.answer_fidelity if ev else None,
        "qa_no_hallucination": ev.no_hallucination if ev else None,
        "qa_repairable": ev.repairable if ev else None,
        "qa_reason": (reason if reason is not None else (ev.reason if ev else None)),
    }


def _decision_payload(
    row: KVPRow,
    *,
    work_id: str,
    status: str,
    admitted: bool,
    refined: bool,
    llm: LLMClient,
    execution_surface: str,
    ev: QAEvaluation | None = None,
    reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "finished_at": utc_now(),
        "work_id": work_id,
        "sample_id": _row_sample_id(row),
        "passage_id": row.passage_id,
        "source_url": row.source_url,
        "input_stage": row.stage,
        "status": status,
        "admitted": admitted,
        "refined": refined,
        "judge_model": _llm_model(llm),
        "judge_endpoints": _llm_endpoints(llm),
        "execution_surface": execution_surface,
        "grounded": ev.grounded if ev else None,
        "answer_fidelity": ev.answer_fidelity if ev else None,
        "no_hallucination": ev.no_hallucination if ev else None,
        "repairable": ev.repairable if ev else None,
        "reason": reason if reason is not None else (ev.reason if ev else None),
        "error": error,
    }


def _failure_result(
    row: KVPRow,
    *,
    work_id: str,
    status: str,
    reason: str,
    llm: LLMClient,
    execution_surface: str,
    error: str | None = None,
) -> tuple[str, None, KVPRow, dict[str, Any]]:
    dropped = row.model_copy(
        update=_qa_updates(
            row=row,
            work_id=work_id,
            status=status,
            admitted=False,
            llm=llm,
            execution_surface=execution_surface,
            reason=reason,
        )
    )
    decision = _decision_payload(
        row,
        work_id=work_id,
        status=status,
        admitted=False,
        refined=False,
        llm=llm,
        execution_surface=execution_surface,
        reason=reason,
        error=error,
    )
    return status, None, dropped, decision


def evaluate_row(
    row: KVPRow,
    llm: LLMClient,
    *,
    work_id: str | None = None,
    execution_surface: str = "direct_qa_eval",
    max_tokens: int = 2048,
) -> tuple[str, KVPRow | None, KVPRow | None, dict[str, Any]]:
    """Evaluate one row and return (status, admitted_row, dropped_row, decision)."""
    work_id = work_id or stage2_work_id_for_row(row)
    raw = llm.call(
        QA_EVAL_SYSTEM,
        QA_EVAL_USER.format(question=row.question, answer=row.answer, context=row.context),
        max_tokens=max_tokens,
    )
    if not raw:
        return _failure_result(
            row,
            work_id=work_id,
            status="llm_empty",
            reason="judge returned no content",
            llm=llm,
            execution_surface=execution_surface,
        )

    ev = parse_eval_response(raw)
    if not ev:
        return _failure_result(
            row,
            work_id=work_id,
            status="parse_failed",
            reason="judge response was not valid QAEvaluation JSON",
            llm=llm,
            execution_surface=execution_surface,
        )

    if not ev.admit:
        status = "dropped"
        dropped = row.model_copy(
            update=_qa_updates(
                row=row,
                work_id=work_id,
                status=status,
                admitted=False,
                llm=llm,
                execution_surface=execution_surface,
                ev=ev,
            )
        )
        decision = _decision_payload(
            row,
            work_id=work_id,
            status=status,
            admitted=False,
            refined=False,
            llm=llm,
            execution_surface=execution_surface,
            ev=ev,
        )
        return status, None, dropped, decision

    prompt = ev.prompt.strip()
    completion = ev.completion.strip()
    if not prompt or not completion:
        return _failure_result(
            row,
            work_id=work_id,
            status="parse_failed",
            reason="admitted judge response omitted prompt or completion",
            llm=llm,
            execution_surface=execution_surface,
        )

    changed = prompt != row.question.strip() or completion != row.answer.strip()
    status = "refined" if changed else "accepted"
    admitted = row.model_copy(
        update={
            **_qa_updates(
                row=row,
                work_id=work_id,
                status=status,
                admitted=True,
                llm=llm,
                execution_surface=execution_surface,
                ev=ev,
            ),
            "question": prompt,
            "answer": completion,
            "refined": changed,
        }
    )
    decision = _decision_payload(
        row,
        work_id=work_id,
        status=status,
        admitted=True,
        refined=changed,
        llm=llm,
        execution_surface=execution_surface,
        ev=ev,
    )
    return status, admitted, None, decision


def refine_row(row: KVPRow, llm: LLMClient) -> Optional[KVPRow]:
    """Compatibility wrapper returning only the admitted/refined row."""
    _, admitted, _, _ = evaluate_row(row, llm)
    return admitted


def read_stage2_rows(path: Path) -> list[KVPRow]:
    if not path.exists():
        return []
    return [KVPRow.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def latest_stage2_decisions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        work_id = str(row.get("work_id") or "")
        if work_id:
            latest[work_id] = row
    return latest


def _append_row(stream: Any, row: KVPRow) -> None:
    stream.write(row.model_dump_json() + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _append_decision(stream: Any, decision: dict[str, Any]) -> None:
    stream.write(json.dumps(decision, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _existing_work_ids(rows: list[KVPRow]) -> set[str]:
    return {row.qa_work_id for row in rows if row.qa_work_id}


def run_stage2(
    rows: list[KVPRow],
    llm: LLMClient,
    output_dir: Path,
    max_workers: int = 5,
    *,
    resume: bool = False,
    execution_surface: str = "direct_qa_eval",
    max_tokens: int = 2048,
) -> tuple[list[KVPRow], list[KVPRow]]:
    """Run durable Stage 2 QA admission. Returns (kept_rows, dropped_rows)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "stage2_eval.jsonl"
    drop_file = output_dir / "stage2_dropped.jsonl"
    quality_file = output_dir / "provenance" / "stage2_quality.jsonl"
    quality_file.parent.mkdir(parents=True, exist_ok=True)

    if not resume:
        out_file.unlink(missing_ok=True)
        drop_file.unlink(missing_ok=True)
        quality_file.unlink(missing_ok=True)

    existing_rows = read_stage2_rows(out_file)
    existing_dropped = read_stage2_rows(drop_file)
    latest_decisions = latest_stage2_decisions(quality_file)
    completed_work_ids = _existing_work_ids(existing_rows)
    completed_work_ids.update(
        row.qa_work_id
        for row in existing_dropped
        if row.qa_work_id and row.qa_status in TERMINAL_QA_STATUSES
    )
    completed_work_ids.update(
        work_id
        for work_id, decision in latest_decisions.items()
        if str(decision.get("status") or "") in TERMINAL_QA_STATUSES
    )

    indexed_rows = [
        (index, row, stage2_work_id_for_row(row, index))
        for index, row in enumerate(rows)
    ]
    pending = [(index, row, work_id) for index, row, work_id in indexed_rows if work_id not in completed_work_ids]
    skipped = len(indexed_rows) - len(pending)
    if skipped:
        log.info("Stage 2: resuming with %d skipped rows and %d pending", skipped, len(pending))

    kept: list[KVPRow] = list(existing_rows)
    dropped: list[KVPRow] = list(existing_dropped)
    if not pending:
        log.info("Stage 2: %d admitted rows already available at %s", len(kept), out_file)
        return kept, dropped

    with (
        out_file.open("a", encoding="utf-8") as kept_stream,
        drop_file.open("a", encoding="utf-8") as drop_stream,
        quality_file.open("a", encoding="utf-8") as quality_stream,
    ):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    evaluate_row,
                    row,
                    llm,
                    work_id=work_id,
                    execution_surface=execution_surface,
                    max_tokens=max_tokens,
                ): (index, row, work_id)
                for index, row, work_id in pending
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage 2"):
                _, row, work_id = futures[fut]
                try:
                    status, admitted, dropped_row, decision = fut.result()
                except Exception as exc:  # noqa: BLE001 - recorded for durable retry.
                    log.warning("Stage 2 row failed: %s: %s", work_id, exc)
                    status, admitted, dropped_row, decision = _failure_result(
                        row,
                        work_id=work_id,
                        status="exception",
                        reason="judge call raised an exception",
                        llm=llm,
                        execution_surface=execution_surface,
                        error=str(exc),
                    )
                if admitted is not None:
                    _append_row(kept_stream, admitted)
                    kept.append(admitted)
                if dropped_row is not None:
                    _append_row(drop_stream, dropped_row)
                    dropped.append(dropped_row)
                _append_decision(quality_stream, decision)

    log.info(
        "Stage 2: %d kept, %d dropped/retryable -> %s",
        len(kept),
        len(dropped),
        out_file,
    )
    return kept, dropped
