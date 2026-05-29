"""Shared NeMo Platform endpoint defaults for pipeline entrypoints.

Explicit legacy service variables still win so existing standalone
microservice deployments keep working, but Platform deployments can set the
NMP_* variables once and let the jobs inherit them.
"""

from __future__ import annotations

import os

DEFAULT_LOCAL_PLATFORM_URL = "http://localhost:8080"
DEFAULT_WORKSPACE = "default"
LEGACY_DATA_STORE_GIT_BASE = "http://nemo-data-store:3000"


def env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def default_nmp_base_url() -> str:
    return (
        env_first("NMP_BASE_URL", default=DEFAULT_LOCAL_PLATFORM_URL) or DEFAULT_LOCAL_PLATFORM_URL
    )


def default_nmp_workspace() -> str:
    return (
        env_first("NMP_WORKSPACE", "DATASET_NAMESPACE", default=DEFAULT_WORKSPACE)
        or DEFAULT_WORKSPACE
    )


def default_customizer_url() -> str:
    return (
        env_first("CUSTOMIZER_URL", "CUSTOMIZER_BASE_URL", "NMP_CUSTOMIZER_URL", "NMP_BASE_URL")
        or DEFAULT_LOCAL_PLATFORM_URL
    )


def default_evaluator_url() -> str:
    return (
        env_first("EVALUATOR_URL", "NMP_EVALUATOR_URL", "NMP_BASE_URL")
        or DEFAULT_LOCAL_PLATFORM_URL
    )


def default_data_designer_url() -> str:
    return (
        env_first("NEMO_MICROSERVICES_BASE_URL", "NMP_DATA_DESIGNER_URL", "NMP_BASE_URL")
        or DEFAULT_LOCAL_PLATFORM_URL
    )


def platform_openai_gateway_url(
    base_url: str | None = None,
    workspace: str | None = None,
) -> str:
    base = (base_url or default_nmp_base_url()).rstrip("/")
    ws = workspace or default_nmp_workspace()
    return f"{base}/v2/workspaces/{ws}/inference/gateway/openai/-"


def default_inference_gateway_url() -> str:
    explicit = env_first("NIM_PROXY_URL", "NMP_INFERENCE_GATEWAY_URL")
    if explicit:
        return explicit
    return platform_openai_gateway_url()


def default_entity_store_url() -> str:
    return (
        env_first("ENTITY_STORE_URL", "NMP_ENTITY_STORE_URL", "NMP_BASE_URL")
        or DEFAULT_LOCAL_PLATFORM_URL
    )


def default_data_store_url() -> str:
    return (
        env_first("DATA_STORE_URL", "NMP_DATASTORE_URL", "NMP_BASE_URL")
        or DEFAULT_LOCAL_PLATFORM_URL
    )


def default_data_store_hf_endpoint() -> str:
    explicit = env_first("DATA_STORE_HF_ENDPOINT", "NMP_DATASTORE_HF_ENDPOINT")
    if explicit:
        return explicit
    return f"{default_data_store_url().rstrip('/')}/v1/hf"


def default_data_store_git_base() -> str:
    return env_first("DATA_STORE_GIT_BASE", "NMP_DATASTORE_GIT_BASE") or LEGACY_DATA_STORE_GIT_BASE
