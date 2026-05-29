"""NeMo Platform Customizer spec builders.

The Platform API creates customization jobs from a single `spec` that refers to
Model Entities and FileSets. These helpers keep that shape isolated from the
legacy standalone Customizer payload builders.
"""
from __future__ import annotations

import os
from typing import Any

from scripts.nemo_platform import default_nmp_workspace

DEFAULT_MAX_SEQ_LENGTH = int(os.getenv("CUSTOMIZER_MAX_SEQ_LENGTH", "2048"))
DEFAULT_LORA_DROPOUT = float(os.getenv("CUSTOMIZER_LORA_DROPOUT", "0.0"))
DEFAULT_PLATFORM_PAYLOAD_FORMAT = os.getenv("CUSTOMIZER_PAYLOAD_FORMAT", "platform")

_PLATFORM_MODEL_ENTITY_NAME_FOR_BASE = {
    "meta/llama-3.2-1b-instruct": "llama-3.2-1b-instruct",
    "meta/llama-3.2-3b-instruct": "llama-3.2-3b-instruct",
    "meta/llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "nvidia/nemotron-3-nano-30b-a3b": "nemotron-3-nano-30b-a3b",
}


def split_entity_ref(ref: str, default_workspace: str | None = None) -> tuple[str, str]:
    """Return (workspace, name) for `workspace/name` or a bare name."""
    if "/" in ref:
        workspace, name = ref.split("/", 1)
        return workspace, name
    return default_workspace or default_nmp_workspace(), ref


def fileset_uri_from_ref(ref: str, default_workspace: str | None = None) -> str:
    """Convert a dataset/entity ref to a Platform fileset URI."""
    if ref.startswith("fileset://"):
        return ref
    workspace, name = split_entity_ref(ref, default_workspace=default_workspace)
    return f"fileset://{workspace}/{name}"


def model_entity_for_base(base_model: str, workspace: str | None = None) -> str:
    """Map a source model slug to the expected Platform Model Entity ref."""
    if "/" in base_model and base_model.split("/", 1)[0] in {"default", "system"}:
        return base_model
    name = _PLATFORM_MODEL_ENTITY_NAME_FOR_BASE.get(base_model)
    if name is None:
        name = base_model.split("/", 1)[-1]
    return f"{workspace or default_nmp_workspace()}/{name}"


def build_mlflow_integration(
    *,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    description: str | None = None,
) -> dict[str, Any] | None:
    """Build the Customizer MLflow integration block when configured."""
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME")
    run_name = run_name or os.getenv("MLFLOW_RUN_NAME")
    if not any((tracking_uri, experiment_name, run_name)):
        return None
    mlflow: dict[str, Any] = {}
    if experiment_name:
        mlflow["experiment_name"] = experiment_name
    if tracking_uri:
        mlflow["tracking_uri"] = tracking_uri
    if run_name:
        mlflow["run_name"] = run_name
    if tags:
        mlflow["tags"] = tags
    if description:
        mlflow["description"] = description
    return {"mlflow": mlflow}


def build_lora_training_spec(
    *,
    rank: int,
    alpha: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    dropout: float = DEFAULT_LORA_DROPOUT,
) -> dict[str, Any]:
    """Build the Platform `spec.training` block for LoRA SFT."""
    return {
        "type": "sft",
        "peft": {
            "type": "lora",
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
        },
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
    }


def build_platform_customizer_spec(
    *,
    model_entity: str,
    dataset_fileset_uri: str,
    output_name: str,
    training: dict[str, Any],
    deployment_lora_enabled: bool = True,
    integrations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the NeMo Platform Customizer job spec."""
    spec: dict[str, Any] = {
        "model": model_entity,
        "dataset": dataset_fileset_uri,
        "training": training,
        "output": {"name": output_name},
    }
    if deployment_lora_enabled:
        spec["deployment_config"] = {"lora_enabled": True}
    if integrations:
        spec["integrations"] = integrations
    return spec


def build_platform_customizer_job(
    *,
    name: str,
    workspace: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Build SDK create-job arguments as a dry-run friendly payload."""
    return {
        "name": name,
        "workspace": workspace,
        "spec": spec,
    }
