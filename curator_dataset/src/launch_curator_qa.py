#!/usr/bin/env python3
"""Launch the pinned official Curator container without exposing API secrets."""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "nvcr.io/nvidia/nemo-curator:26.04"
DEFAULT_DIGEST = "sha256:11635749967ea52fb82ccc2b82d326a4de3dfc50a51605c95a993a007e8219ac"


def require_experiment_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.relative_to(EXPERIMENT_ROOT)
    return resolved


def container_path(path: Path) -> str:
    relative = require_experiment_path(path).relative_to(EXPERIMENT_ROOT)
    return str(Path("/workspace/curator_dataset") / relative)


def read_api_key() -> str:
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "secret",
            "nvidia-inference-key",
            "-n",
            "runai-rag",
            "-o",
            "jsonpath={.data.api-key}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return base64.b64decode(result.stdout.strip()).decode().strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--digest", default=DEFAULT_DIGEST)
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
    input_path = require_experiment_path(args.input)
    output_dir = require_experiment_path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")

    image_ref = f"{args.image.split(':', 1)[0]}@{args.digest}"
    environment = os.environ.copy()
    environment["NVIDIA_API_KEY"] = read_api_key()
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--shm-size",
        "8g",
        "--env",
        "NVIDIA_API_KEY",
        "--env",
        "HOME=/tmp",
        "--env",
        "HF_HOME=/workspace/curator_dataset/cache/huggingface",
        "--env",
        f"CURATOR_IMAGE_REF={args.image}",
        "--env",
        f"CURATOR_IMAGE_DIGEST={args.digest}",
        "--volume",
        f"{EXPERIMENT_ROOT}:/workspace/curator_dataset",
        "--workdir",
        "/workspace/curator_dataset",
        "--entrypoint",
        "python",
        image_ref,
        "src/run_curator_diverse_qa.py",
        "--input",
        container_path(input_path),
        "--output-dir",
        container_path(output_dir),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--max-concurrent-requests",
        str(args.max_concurrent_requests),
        "--max-retries",
        str(args.max_retries),
        "--base-delay",
        str(args.base_delay),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--min-document-tokens",
        str(args.min_document_tokens),
        "--min-segment-tokens",
        str(args.min_segment_tokens),
        "--max-input-tokens",
        str(args.max_input_tokens),
    ]
    completed = subprocess.run(command, env=environment, check=False)
    if output_dir.exists():
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--volume",
                f"{EXPERIMENT_ROOT}:/workspace/curator_dataset",
                "--entrypoint",
                "chown",
                image_ref,
                "-R",
                f"{os.getuid()}:{os.getgid()}",
                container_path(output_dir),
            ],
            check=True,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
