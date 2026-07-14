#!/usr/bin/env python3
"""Launch the pinned Curator image with enough shared memory for a full run."""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
IMAGE = "nvcr.io/nvidia/nemo-curator:26.04"
DIGEST = "sha256:11635749967ea52fb82ccc2b82d326a4de3dfc50a51605c95a993a007e8219ac"


def experiment_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.relative_to(EXPERIMENT_ROOT)
    return resolved


def mounted_path(path: Path) -> str:
    return str(
        Path("/workspace/curator_dataset")
        / experiment_path(path).relative_to(EXPERIMENT_ROOT)
    )


def api_key() -> str:
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
    parser.add_argument("--max-concurrent-requests", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = experiment_path(args.input)
    output_dir = experiment_path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    image_ref = f"nvcr.io/nvidia/nemo-curator@{DIGEST}"
    environment = os.environ.copy()
    environment["NVIDIA_API_KEY"] = api_key()
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--shm-size",
        "64g",
        "--env",
        "NVIDIA_API_KEY",
        "--env",
        "HOME=/tmp",
        "--env",
        "HF_HOME=/workspace/curator_dataset/cache/huggingface",
        "--env",
        f"CURATOR_IMAGE_REF={IMAGE}",
        "--env",
        f"CURATOR_IMAGE_DIGEST={DIGEST}",
        "--volume",
        f"{EXPERIMENT_ROOT}:/workspace/curator_dataset",
        "--workdir",
        "/workspace/curator_dataset",
        "--entrypoint",
        "python",
        image_ref,
        "src/run_curator_diverse_qa.py",
        "--input",
        mounted_path(input_path),
        "--output-dir",
        mounted_path(output_dir),
        "--max-concurrent-requests",
        str(args.max_concurrent_requests),
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
                mounted_path(output_dir),
            ],
            check=True,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
