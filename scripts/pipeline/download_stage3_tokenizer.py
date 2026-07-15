#!/usr/bin/env python3
"""Download the Stage 3 tokenizer artifacts without model weights."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_DIR = Path("outputs/tokenizers/llama-3.1-8b-instruct")
TOKENIZER_PATTERNS = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "added_tokens.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download tokenizer/config artifacts needed by Stage 3 local tests. "
            "Model weight files are intentionally excluded."
        )
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model repo to download from. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local tokenizer output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the local AutoTokenizer load check after download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install the project first with "
            'python -m pip install -e ".[dev]".'
        ) from exc

    snapshot_download(
        repo_id=args.model_id,
        local_dir=output_dir,
        allow_patterns=TOKENIZER_PATTERNS,
        token=args.token,
    )

    if not args.skip_validate:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: transformers. Install the project first with "
                'python -m pip install -e ".[dev]".'
            ) from exc
        AutoTokenizer.from_pretrained(output_dir, local_files_only=True)

    print(f"Tokenizer artifacts ready: {output_dir}")
    print(f"export PIPELINE_STAGE3_TOKENIZER={output_dir}")


if __name__ == "__main__":
    main()
