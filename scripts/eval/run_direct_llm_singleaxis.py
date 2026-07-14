#!/usr/bin/env python3
"""Run direct saved-response single-axis judging with a configurable LLM judge."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("JUDGE_API_URL", "https://inference-api.nvidia.com/v1")
os.environ.setdefault("EVALUATOR_JUDGE_MODEL", "azure/anthropic/claude-sonnet-4-6")
os.environ.setdefault("JUDGE_API_KEY_ENV", "NVIDIA_API_KEY")
os.environ.setdefault("SINGLEAXIS_OUTPUT_ROOT", "/mnt/nvme2/peft/evals/singleaxis-llm")

from scripts.eval.run_direct_kimi_singleaxis import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
