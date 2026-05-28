"""Stage 1A: logical entailment extraction to source-grounded KVP rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pipeline.llm_client import LLMClient  # noqa: E402
from scripts.pipeline.models import (  # noqa: E402
    KVPRow,
    LogEntailment,
    LogEntailmentList,
    Passage,
    QAKeyValuePair,
)
from scripts.pipeline.provenance import (  # noqa: E402
    entailment_id_for_passage,
    entailments_from_kvp_rows,
    passage_source_chunk_ids,
    passage_source_revision_id,
    sha256_text,
    utc_now,
)
from scripts.pipeline.provenance_io import write_jsonl  # noqa: E402
from scripts.pipeline.prompts import KVP_SYSTEM, KVP_USER, LE_SYSTEM, LE_USER  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_NIM_ENDPOINTS = os.getenv(
    "PIPELINE_NIM_ENDPOINTS",
    "http://nim-llm-super-120b-bw.runai-rag:8000/v1",
)
DEFAULT_LLM_MODEL = os.getenv("PIPELINE_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
DEFAULT_LLM_API_KEY = os.getenv("PIPELINE_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY") or "local"
DEFAULT_INPUT_PASSAGES = Path(os.getenv("PASSAGES_PATH", "/datasets/nim_curated/passages.jsonl"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/datasets/nim_curated"))
DEFAULT_OBSERVABILITY_DIR = Path(
    os.getenv("OBSERVABILITY_DIR", "/observability/stage1a-le-kvp")
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()


def _extract_json_object(raw: str) -> Optional[dict]:
    s = _strip_fences(raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def parse_le_response(raw: str) -> Optional[LogEntailmentList]:
    """Parse LLM response as a LogEntailmentList."""
    obj = _extract_json_object(raw)
    if not obj:
        return None
    try:
        return LogEntailmentList(**obj)
    except ValidationError:
        try:
            single = LogEntailment(**obj)
            return LogEntailmentList(entailments=[single])
        except ValidationError:
            return None


def parse_kvp_response(raw: str) -> Optional[QAKeyValuePair]:
    obj = _extract_json_object(raw)
    if not obj:
        return None
    try:
        return QAKeyValuePair(**obj)
    except ValidationError:
        return None


def process_passage_1a(passage: Passage, llm: LLMClient) -> list[KVPRow]:
    """Run logical entailment extraction and KVP expansion on one passage."""
    rows: list[KVPRow] = []

    le_raw = llm.call(LE_SYSTEM, LE_USER.format(text=passage.text), max_tokens=8192)
    if not le_raw:
        return rows
    le_list = parse_le_response(le_raw)
    if not le_list:
        log.debug("Stage 1A: no valid entailments for %s", passage.passage_id)
        return rows

    extractor_model = getattr(llm, "model", None)
    if not isinstance(extractor_model, str):
        extractor_model = None
    extractor_temperature = getattr(llm, "temperature", None)
    if not isinstance(extractor_temperature, (int, float)):
        extractor_temperature = None

    for ent_idx, ent in enumerate(le_list.entailments):
        ent_id = entailment_id_for_passage(
            passage, ent_idx, ent.conclusion, ent.premises
        )
        source_revision_id = passage_source_revision_id(passage)
        source_chunk_ids = passage_source_chunk_ids(passage)
        for prem_idx, premise in enumerate(ent.premises[:3]):
            kvp_raw = llm.call(
                KVP_SYSTEM,
                KVP_USER.format(premise=premise, conclusion=ent.conclusion, text=passage.text),
                max_tokens=4096,
            )
            if not kvp_raw:
                continue
            kvp = parse_kvp_response(kvp_raw)
            if not kvp:
                continue
            rows.append(KVPRow(
                passage_id=passage.passage_id,
                source_url=passage.url,
                product_family=passage.product_family,
                stage="1a",
                entailment_index=ent_idx,
                premise_index=prem_idx,
                sample_id=None,
                entailment_id=ent_id,
                entailment_claim=ent.conclusion,
                entailment_premises=ent.premises,
                source_revision_ids=[source_revision_id],
                source_chunk_ids=source_chunk_ids,
                extractor_model=extractor_model,
                extractor_prompt_hash=sha256_text(LE_SYSTEM + "\n" + LE_USER),
                extractor_temperature=extractor_temperature,
                question=kvp.question.strip(),
                answer=kvp.answer.strip(),
                context=passage.text,
                refined=False,
            ))
    return rows


def shard_label(shard_index: int, shard_count: int) -> str:
    return f"shard-{shard_index:05d}-of-{shard_count:05d}"


def validate_shard_args(shard_index: int, shard_count: int) -> None:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count - 1}], got {shard_index}"
        )


def passage_shard_index(passage_id: str, shard_count: int) -> int:
    digest = hashlib.sha256(passage_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % shard_count


def select_shard_passages(
    passages: list[Passage],
    *,
    shard_index: int,
    shard_count: int,
) -> list[Passage]:
    validate_shard_args(shard_index, shard_count)
    if shard_count == 1:
        return list(passages)
    return [
        passage for passage in passages
        if passage_shard_index(passage.passage_id, shard_count) == shard_index
    ]


def output_filenames_for_shard(shard_index: int, shard_count: int) -> tuple[str, str]:
    validate_shard_args(shard_index, shard_count)
    if shard_count == 1:
        return "stage1a_le.jsonl", "entailments.jsonl"
    label = shard_label(shard_index, shard_count)
    return f"stage1a_le.{label}.jsonl", f"entailments.{label}.jsonl"


def read_passages_jsonl(path: Path) -> list[Passage]:
    if not path.exists():
        raise FileNotFoundError(f"Passage input not found: {path}")
    return [Passage.model_validate_json(line) for line in path.read_text().splitlines() if line]


def run_stage1a(
    passages: list[Passage],
    llm: LLMClient,
    output_dir: Path,
    max_workers: int = 5,
    *,
    output_filename: str = "stage1a_le.jsonl",
    entailments_filename: str = "entailments.jsonl",
) -> list[KVPRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / output_filename
    entailments_file = output_dir / "provenance" / entailments_filename
    all_rows: list[KVPRow] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_passage_1a, p, llm) for p in passages]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage 1A"):
            all_rows.extend(fut.result())

    with out_file.open("w") as f:
        for row in all_rows:
            f.write(row.model_dump_json() + "\n")
    write_jsonl(entailments_file, entailments_from_kvp_rows(all_rows))
    log.info("Stage 1A: %d KVPs -> %s", len(all_rows), out_file)
    return all_rows


def safe_metric_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in cleaned.split("_") if part) or "unknown"


def count_jsonl_rows(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def file_manifest(path: Path, artifact_path: str, artifact_kind: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "path": str(path),
        "artifact_path": artifact_path,
        "artifact_kind": artifact_kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        manifest["rows"] = count_jsonl_rows(path)
    return manifest


def build_stage1a_observability_documents(
    *,
    input_passages_path: Path,
    output_dir: Path,
    output_file: Path,
    entailments_file: Path,
    input_passage_count: int,
    selected_passage_count: int,
    rows: list[KVPRow],
    shard_index: int,
    shard_count: int,
    max_workers: int,
    llm_model: str,
    llm_endpoints: list[str],
    pipeline_run_id: str | None,
    mlflow_tracking_uri: str | None,
    mlflow_experiment_name: str | None,
    mlflow_parent_run_id: str | None,
) -> dict[str, dict[str, Any]]:
    product_family_counts = Counter(row.product_family or "unknown" for row in rows)
    source_revision_ids = {
        source_revision_id
        for row in rows
        for source_revision_id in (row.source_revision_ids or [])
    }
    source_chunk_ids = {
        source_chunk_id
        for row in rows
        for source_chunk_id in (row.source_chunk_ids or [])
    }
    entailment_ids = {row.entailment_id for row in rows if row.entailment_id}

    metrics: dict[str, int | float] = {
        "stage1a.passages.input.count": input_passage_count,
        "stage1a.passages.selected.count": selected_passage_count,
        "stage1a.rows.count": len(rows),
        "stage1a.entailments.count": len(entailment_ids),
        "stage1a.source_revisions.count": len(source_revision_ids),
        "stage1a.source_chunks.count": len(source_chunk_ids),
        "stage1a.shard.index": shard_index,
        "stage1a.shard.count": shard_count,
        "stage1a.max_workers": max_workers,
    }
    for product_family, count in sorted(product_family_counts.items()):
        metrics[f"stage1a.product_family.{safe_metric_name(product_family)}.rows"] = count

    artifacts = [
        file_manifest(input_passages_path, "passages.jsonl", "input_passages"),
        file_manifest(output_file, output_file.name, "stage1a_rows"),
        file_manifest(
            entailments_file,
            f"provenance/{entailments_file.name}",
            "entailment_provenance",
        ),
    ]

    run_context: dict[str, Any] = {
        "schema_version": "observability.v1",
        "pipeline_stage": "stage1a-le-kvp",
        "created_at": utc_now(),
        "pipeline_run_id": pipeline_run_id,
        "output_dir": str(output_dir),
        "shard": {
            "index": shard_index,
            "count": shard_count,
            "label": shard_label(shard_index, shard_count),
        },
        "parameters": {
            "max_workers": max_workers,
        },
        "mlflow": {
            "tracking_uri": mlflow_tracking_uri,
            "experiment_name": mlflow_experiment_name,
            "parent_run_id": mlflow_parent_run_id,
        },
    }
    service_refs: dict[str, Any] = {
        "schema_version": "observability.v1",
        "services": {
            "llm": {
                "model": llm_model,
                "endpoints": llm_endpoints,
                "endpoint_count": len(llm_endpoints),
            },
        },
        "inputs": {
            "passages_path": str(input_passages_path),
        },
        "outputs": {
            "stage1a_rows": str(output_file),
            "entailments": str(entailments_file),
        },
    }
    artifacts_manifest: dict[str, Any] = {
        "schema_version": "observability.v1",
        "artifacts": artifacts,
    }
    return {
        "run_context.json": run_context,
        "metrics.json": metrics,
        "artifacts_manifest.json": artifacts_manifest,
        "service_refs.json": service_refs,
    }


def write_observability_documents(out_dir: Path, documents: dict[str, dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in documents.items():
        (out_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_endpoints(value: str) -> list[str]:
    endpoints = [endpoint.strip() for endpoint in value.split(",") if endpoint.strip()]
    if not endpoints:
        raise ValueError("At least one LLM endpoint is required")
    return endpoints


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run sharded Stage 1A entailment extraction.")
    ap.add_argument("--input-passages", type=Path, default=DEFAULT_INPUT_PASSAGES)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--observability-dir", type=Path, default=DEFAULT_OBSERVABILITY_DIR)
    ap.add_argument("--shard-index", type=int, default=_env_int("SHARD_INDEX", 0))
    ap.add_argument("--shard-count", type=int, default=_env_int("SHARD_COUNT", 1))
    ap.add_argument("--output-filename", default=None)
    ap.add_argument("--entailments-filename", default=None)
    ap.add_argument("--max-workers", type=int, default=_env_int("PIPELINE_MAX_WORKERS", 5))
    ap.add_argument("--nim-endpoints", default=DEFAULT_NIM_ENDPOINTS)
    ap.add_argument("--model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--api-key", default=DEFAULT_LLM_API_KEY)
    ap.add_argument(
        "--min-request-interval-s",
        type=float,
        default=_env_float("PIPELINE_MIN_REQUEST_INTERVAL_S", 0.5),
    )
    ap.add_argument("--retry-attempts", type=int, default=_env_int("PIPELINE_RETRY_ATTEMPTS", 3))
    ap.add_argument(
        "--retry-base-delay-s",
        type=float,
        default=_env_float("PIPELINE_RETRY_BASE_DELAY_S", 5.0),
    )
    ap.add_argument("--temperature", type=float, default=_env_float("PIPELINE_TEMPERATURE", 0.2))
    ap.add_argument(
        "--no-think",
        dest="no_think",
        action="store_true",
        default=_env_bool("PIPELINE_LLM_NO_THINK", True),
    )
    ap.add_argument("--allow-think", dest="no_think", action="store_false")
    ap.add_argument("--pipeline-run-id", default=os.getenv("PIPELINE_RUN_ID"))
    ap.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    ap.add_argument("--mlflow-experiment-name", default=os.getenv("MLFLOW_EXPERIMENT_NAME"))
    ap.add_argument("--mlflow-parent-run-id", default=os.getenv("MLFLOW_PARENT_RUN_ID"))
    args = ap.parse_args(argv)
    validate_shard_args(args.shard_index, args.shard_count)
    if args.max_workers < 1:
        ap.error("--max-workers must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args(argv)
    endpoints = parse_endpoints(args.nim_endpoints)
    output_filename, entailments_filename = output_filenames_for_shard(
        args.shard_index,
        args.shard_count,
    )
    if args.output_filename:
        output_filename = args.output_filename
    if args.entailments_filename:
        entailments_filename = args.entailments_filename

    passages = read_passages_jsonl(args.input_passages)
    selected_passages = select_shard_passages(
        passages,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    log.info(
        "Stage 1A: shard %d/%d selected %d of %d passages",
        args.shard_index,
        args.shard_count,
        len(selected_passages),
        len(passages),
    )

    llm = LLMClient(
        endpoints=endpoints,
        model=args.model,
        api_key=args.api_key,
        max_workers=args.max_workers,
        min_interval_s=args.min_request_interval_s,
        retry_attempts=args.retry_attempts,
        retry_base_delay_s=args.retry_base_delay_s,
        no_think=args.no_think,
        temperature=args.temperature,
    )
    rows = run_stage1a(
        selected_passages,
        llm,
        args.output_dir,
        max_workers=args.max_workers,
        output_filename=output_filename,
        entailments_filename=entailments_filename,
    )

    output_file = args.output_dir / output_filename
    entailments_file = args.output_dir / "provenance" / entailments_filename
    docs = build_stage1a_observability_documents(
        input_passages_path=args.input_passages,
        output_dir=args.output_dir,
        output_file=output_file,
        entailments_file=entailments_file,
        input_passage_count=len(passages),
        selected_passage_count=len(selected_passages),
        rows=rows,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        max_workers=args.max_workers,
        llm_model=args.model,
        llm_endpoints=endpoints,
        pipeline_run_id=args.pipeline_run_id,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_parent_run_id=args.mlflow_parent_run_id,
    )
    write_observability_documents(args.observability_dir, docs)
    log.info("Stage 1A: written observability files under %s", args.observability_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
