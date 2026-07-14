#!/usr/bin/env python3
"""Run Curator 26.04's native Nemotron-CC DiverseQA pipeline.

This script is intended to execute inside the official NeMo Curator container.
The NVIDIA API key is accepted only through the ``NVIDIA_API_KEY`` environment
variable and is never persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nemo_curator
from transformers import AutoTokenizer

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import DiverseQAStage
from nemo_curator.stages.synthetic.nemotron_cc.prompts import (
    DIVERSE_QA_PROMPT_TEMPLATE,
    NEMOTRON_CC_SYSTEM_PROMPT,
)
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter


TUTORIAL_DIR = Path("/opt/Curator/tutorials/synthetic/nemotron_cc")
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))
from nemotron_cc_pipelines import add_preprocessing_pipeline  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open() as stream:
        return sum(1 for line in stream if line.strip())


def output_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "run_manifest.json":
            continue
        artifact: dict[str, Any] = {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".jsonl":
            artifact["rows"] = count_jsonl(path)
        artifacts.append(artifact)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default="https://inference-api.nvidia.com/v1",
    )
    parser.add_argument("--model", default="nvidia/nvidia/nemotron-3-super-v3")
    parser.add_argument(
        "--tokenizer",
        default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    )
    parser.add_argument("--max-concurrent-requests", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--base-delay", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=600)
    parser.add_argument("--min-document-tokens", type=int, default=30)
    parser.add_argument("--min-segment-tokens", type=int, default=30)
    parser.add_argument("--max-input-tokens", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required")
    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    started_at = utc_now()
    start_time = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    helper_args = argparse.Namespace(tokenizer=tokenizer, hf_token=os.getenv("HF_TOKEN", ""))

    client = RayClient(include_dashboard=False)
    client.start()
    try:
        pipeline = Pipeline(
            name="nim_curator_diverse_qa",
            description="Native Curator Nemotron-CC DiverseQA over NIM HTML URL documents",
        )
        pipeline.add_stage(JsonlReader(file_paths=[str(args.input)], fields=None))
        pipeline = add_preprocessing_pipeline(
            pipeline=pipeline,
            text_field="text",
            system_prompt=NEMOTRON_CC_SYSTEM_PROMPT,
            user_prompt_template=DIVERSE_QA_PROMPT_TEMPLATE,
            min_document_tokens=args.min_document_tokens,
            min_segment_tokens=args.min_segment_tokens,
            max_input_tokens=args.max_input_tokens,
            args=helper_args,
        )

        llm_client = AsyncOpenAIClient(
            api_key=api_key,
            base_url=args.base_url,
            max_concurrent_requests=args.max_concurrent_requests,
            max_retries=args.max_retries,
            base_delay=args.base_delay,
        )
        generation_config = GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_output_tokens,
            seed=args.seed,
            extra_kwargs={
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            },
        )
        pipeline.add_stage(
            DiverseQAStage(
                client=llm_client,
                model_name=args.model,
                generation_config=generation_config,
                input_field="text",
                output_field="diverse_qa",
            )
        )
        raw_output_dir = args.output_dir / "raw"
        pipeline.add_stage(JsonlWriter(path=str(raw_output_dir), mode="overwrite"))
        print(pipeline.describe())
        results = pipeline.run(XennaExecutor())
    finally:
        client.stop()

    artifacts = output_artifacts(args.output_dir)
    manifest = {
        "schema_version": "curator_dataset.diverse_qa_run.v1",
        "status": "completed",
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
        "curator": {
            "version": getattr(nemo_curator, "__version__", "unknown"),
            "image": os.getenv("CURATOR_IMAGE_REF"),
            "image_digest": os.getenv("CURATOR_IMAGE_DIGEST"),
            "pipeline": "Nemotron-CC DiverseQAStage",
        },
        "input": {
            "path": str(args.input),
            "rows": count_jsonl(args.input),
            "bytes": args.input.stat().st_size,
            "sha256": sha256_file(args.input),
        },
        "provider": {
            "base_url": args.base_url,
            "model": args.model,
            "api_key_source": "Kubernetes Secret reference; value not persisted",
        },
        "tokenizer": args.tokenizer,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_output_tokens": args.max_output_tokens,
            "thinking": False,
            "max_concurrent_requests": args.max_concurrent_requests,
            "max_retries": args.max_retries,
            "base_delay": args.base_delay,
        },
        "preprocessing": {
            "native_helper": "nemotron_cc_pipelines.add_preprocessing_pipeline",
            "min_document_tokens": args.min_document_tokens,
            "min_segment_tokens": args.min_segment_tokens,
            "max_input_tokens": args.max_input_tokens,
        },
        "prompts": {
            "system_sha256": hashlib.sha256(
                NEMOTRON_CC_SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "diverse_qa_sha256": hashlib.sha256(
                DIVERSE_QA_PROMPT_TEMPLATE.encode()
            ).hexdigest(),
        },
        "result_task_count": len(results or []),
        "artifacts": artifacts,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
