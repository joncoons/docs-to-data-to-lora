"""Stage 3 production tokenizer path resolution."""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_STAGE3_TOKENIZER_MODEL_CACHE = Path(
    "ngc/hub/models--nim--meta--llama-3.1-8b-instruct"
)
DEFAULT_STAGE3_TOKENIZER_SNAPSHOT = "fp8-tool-calling"


def default_stage3_tokenizer_name_or_path() -> str:
    configured = os.environ.get("PIPELINE_STAGE3_TOKENIZER")
    if configured:
        return configured

    local_nim_cache = os.environ.get("LOCAL_NIM_CACHE", "~/.cache/nim")
    snapshot = os.environ.get(
        "PIPELINE_STAGE3_TOKENIZER_SNAPSHOT",
        DEFAULT_STAGE3_TOKENIZER_SNAPSHOT,
    )
    return str(
        Path(local_nim_cache).expanduser()
        / DEFAULT_STAGE3_TOKENIZER_MODEL_CACHE
        / "snapshots"
        / snapshot
    )
