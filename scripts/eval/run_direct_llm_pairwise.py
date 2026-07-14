#!/usr/bin/env python3
"""Run direct saved-response pairwise judging with a configurable LLM judge."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("JUDGE_API_URL", "http://llm-judge.default.svc.cluster.local:8000/v1")
os.environ.setdefault("EVALUATOR_JUDGE_MODEL", "azure/anthropic/claude-sonnet-4-6")
os.environ.setdefault("JUDGE_API_KEY_ENV", "LLM_API_KEY")
os.environ.setdefault("PAIRWISE_OUTPUT_ROOT", "<EVAL_ROOT>/pairwise-llm")

from scripts.eval.direct_llm_pairwise_impl import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
