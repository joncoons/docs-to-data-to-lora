"""Config + credential accessors for the Stage 2 pipeline."""
from __future__ import annotations

import base64
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from scripts.pipeline.stage3_tokenizer import default_stage3_tokenizer_name_or_path


# Defaults assume in-cluster pod DNS. Override via env vars when running from
# a host that can't resolve cluster service names (e.g., from ubuntu-local-dev,
# which CAN reach ClusterIPs via k3s iptables routing but cannot resolve names).
DEFAULT_ES_HOST = os.environ.get(
    "PIPELINE_ES_HOST",
    "https://rag-eck-elasticsearch-es-http.runai-rag:9200",
)
DEFAULT_NIM_ENDPOINTS: tuple[str, ...] = tuple(
    os.environ.get(
        "PIPELINE_NIM_ENDPOINTS",
        "http://nim-llm-super-120b-bw.runai-rag:8000/v1",
    ).split(",")
)
DEFAULT_EXTERNAL_JUDGE_BASE = os.environ.get(
    "PIPELINE_EXTERNAL_JUDGE_BASE",
    "https://inference-api.nvidia.com/v1",
)
DEFAULT_EXTERNAL_JUDGE_MODEL = os.environ.get(
    "PIPELINE_EXTERNAL_JUDGE_MODEL",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
)
DEFAULT_STAGE2_QA_ENDPOINTS: tuple[str, ...] = tuple(
    endpoint.strip()
    for endpoint in os.environ.get(
        "PIPELINE_STAGE2_QA_ENDPOINTS",
        DEFAULT_EXTERNAL_JUDGE_BASE,
    ).split(",")
    if endpoint.strip()
)
# Stage 2 defaults to Super 120B-class QA for cost. Override with Ultra or
# another foundation/frontier-grade model only when the admission pass is
# operationally critical enough to justify the extra spend.
DEFAULT_STAGE2_QA_MODEL = os.environ.get(
    "PIPELINE_STAGE2_QA_MODEL",
    "nvidia/nvidia/nemotron-3-super-v3",
)


@dataclass
class Config:
    es_host: str = DEFAULT_ES_HOST
    nim_endpoints: list[str] = field(default_factory=lambda: list(DEFAULT_NIM_ENDPOINTS))
    external_judge_base_url: str = DEFAULT_EXTERNAL_JUDGE_BASE
    external_judge_model: str = DEFAULT_EXTERNAL_JUDGE_MODEL
    stage2_qa_endpoints: list[str] = field(default_factory=lambda: list(DEFAULT_STAGE2_QA_ENDPOINTS))
    stage2_qa_model: str = DEFAULT_STAGE2_QA_MODEL
    stage2_qa_temperature: float = float(os.environ.get("PIPELINE_STAGE2_QA_TEMPERATURE", "0.0"))
    stage2_qa_max_tokens: int = int(os.environ.get("PIPELINE_STAGE2_QA_MAX_TOKENS", "2048"))
    stage2_execution_surface: str = os.environ.get(
        "PIPELINE_STAGE2_EXECUTION_SURFACE",
        "curator_llm_quality",
    )
    super120b_model: str = "nvidia/nemotron-3-super-120b-a12b"

    base_output_dir: Path = field(default_factory=lambda: Path("/mnt/nvme2/peft/datasets/v2"))

    # Stage 0
    min_passage_tokens: int = 60
    max_passage_tokens: int = 999_999_999  # no cap for HTML by-URL merge

    # Stage 1B
    knn_k: int = 6
    knn_num_candidates: int = 50
    knn_top_neighbors: int = 3
    knn_max_context_tokens: int = 1200

    # Stage 1C
    stage1c_selection_mode: str = os.environ.get("PIPELINE_STAGE1C_SELECTION_MODE", "stratified")
    stage1c_top_percent: float = 0.25
    stage1c_min_passages: int = 100

    # Stage 1.5
    bias_threshold_factor: float = 0.5     # under-rep if density < median * 0.5
    bias_gapfill_target_factor: float = 0.8  # bring back up to median * 0.8
    gapfill_top_n_chunks: int = 20
    gapfill_pairs_per_call: int = 5
    gapfill_max_attempt_factor: int = 3    # max_attempts = pairs_needed * 3

    # Stage 3
    train_val_split: float = 0.90
    minhash_threshold: float = 0.85
    stage3_tokenizer_name_or_path: str = field(
        default_factory=default_stage3_tokenizer_name_or_path
    )
    min_question_tokens: int = int(os.environ.get("PIPELINE_STAGE3_MIN_QUESTION_TOKENS", "12"))
    min_answer_tokens: int = int(os.environ.get("PIPELINE_STAGE3_MIN_ANSWER_TOKENS", "8"))

    # Stage 4
    judge_sample_size: int = 100
    judge_pass_threshold: float = 0.90

    # Concurrency
    max_workers: int = 5
    min_request_interval_s: float = 0.5
    retry_attempts: int = 3
    retry_base_delay_s: float = 5.0

    def output_dir_for(self, collection: str) -> Path:
        return self.base_output_dir / collection


def get_k8s_secret(name: str, namespace: str, key: str) -> str:
    """Fetch a kubernetes secret value, base64-decoded.

    Raises RuntimeError with kubectl's stderr if the command fails (e.g.
    secret missing, wrong namespace, no cluster access).
    """
    try:
        result = subprocess.run(
            ["kubectl", "get", "secret", name, "-n", namespace,
             "-o", f"jsonpath={{.data.{key}}}"],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"kubectl failed fetching secret {name}/{key} in ns {namespace}: "
            f"{exc.stderr.strip()}"
        ) from exc
    return base64.b64decode(result.stdout.strip()).decode().strip()


def get_es_password() -> str:
    """Get the ECK Elasticsearch elastic-user password."""
    return get_k8s_secret(
        "rag-eck-elasticsearch-es-elastic-user", "runai-rag", "elastic"
    )


def get_external_judge_api_key() -> str:
    """Get the NVIDIA Inference API key used for external judge calls."""
    return get_k8s_secret("nvidia-inference-key", "runai-rag", "api-key")
