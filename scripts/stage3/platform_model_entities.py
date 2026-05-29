"""Plan and verify NeMo Platform Model Entities for Stage 3 training.

Customizer Platform jobs require a base Model Entity that references a model
FileSet. This helper writes an audit manifest for MLflow/Kubernetes and can
optionally verify or create those resources through the NeMo Platform SDK.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.nemo_platform import default_nmp_base_url, default_nmp_workspace  # noqa: E402
from scripts.stage3.platform_customizer import model_entity_for_base  # noqa: E402


@dataclass(frozen=True)
class PlatformModelEntitySpec:
    """Expected Platform resources for one trainable base model."""

    source_model: str
    model_entity_name: str
    fileset_name: str
    hf_repo_id: str
    description: str
    repo_type: str = "model"
    requires_token: bool = False
    token_secret: str | None = None

    @property
    def storage_type(self) -> str:
        return "huggingface"


DEFAULT_MODEL_SPECS: dict[str, PlatformModelEntitySpec] = {
    "meta/llama-3.2-1b-instruct": PlatformModelEntitySpec(
        source_model="meta/llama-3.2-1b-instruct",
        model_entity_name="llama-3.2-1b-instruct",
        fileset_name="llama-3.2-1b-instruct",
        hf_repo_id="meta-llama/Llama-3.2-1B-Instruct",
        description="Llama 3.2 1B Instruct base model for Stage 3 LoRA training",
        requires_token=True,
    ),
    "meta/llama-3.2-3b-instruct": PlatformModelEntitySpec(
        source_model="meta/llama-3.2-3b-instruct",
        model_entity_name="llama-3.2-3b-instruct",
        fileset_name="llama-3.2-3b-instruct",
        hf_repo_id="meta-llama/Llama-3.2-3B-Instruct",
        description="Llama 3.2 3B Instruct base model for Stage 3 LoRA training",
        requires_token=True,
    ),
    "meta/llama-3.1-8b-instruct": PlatformModelEntitySpec(
        source_model="meta/llama-3.1-8b-instruct",
        model_entity_name="llama-3.1-8b-instruct",
        fileset_name="llama-3.1-8b-instruct",
        hf_repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        description="Llama 3.1 8B Instruct base model for Stage 3 LoRA training",
        requires_token=True,
    ),
    "nvidia/nemotron-3-nano-30b-a3b": PlatformModelEntitySpec(
        source_model="nvidia/nemotron-3-nano-30b-a3b",
        model_entity_name="nemotron-3-nano-30b-a3b",
        fileset_name="nemotron-3-nano-30b-a3b",
        hf_repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        description="Nemotron 3 Nano 30B-A3B BF16 base model for Stage 3 MoE LoRA training",
    ),
}


@dataclass(frozen=True)
class PlatformModelEntityConfig:
    workspace: str
    nmp_base_url: str
    models: tuple[str, ...] = ()
    hf_token_secret: str | None = None
    out: Path | None = None
    verify: bool = False
    create_missing: bool = False
    dry_run: bool = False


def selected_model_specs(
    models: Iterable[str] | None = None,
    *,
    hf_token_secret: str | None = None,
) -> list[PlatformModelEntitySpec]:
    """Return configured specs for the requested source model slugs."""
    requested = list(models or DEFAULT_MODEL_SPECS.keys())
    unknown = sorted(set(requested).difference(DEFAULT_MODEL_SPECS))
    if unknown:
        known = ", ".join(sorted(DEFAULT_MODEL_SPECS))
        raise ValueError(f"Unknown base model(s): {unknown}. Known models: {known}")

    specs: list[PlatformModelEntitySpec] = []
    for model in requested:
        spec = DEFAULT_MODEL_SPECS[model]
        if hf_token_secret and spec.storage_type == "huggingface":
            spec = replace(spec, token_secret=hf_token_secret)
        specs.append(spec)
    return specs


def build_model_entity_plan(spec: PlatformModelEntitySpec, workspace: str) -> dict[str, Any]:
    """Build a dry-run friendly resource plan for one base model."""
    return {
        "source_model": spec.source_model,
        "workspace": workspace,
        "model_entity_name": spec.model_entity_name,
        "model_entity_ref": f"{workspace}/{spec.model_entity_name}",
        "customizer_model_ref": model_entity_for_base(spec.source_model, workspace),
        "fileset_name": spec.fileset_name,
        "fileset_ref": f"{workspace}/{spec.fileset_name}",
        "fileset_uri": f"fileset://{workspace}/{spec.fileset_name}",
        "description": spec.description,
        "source": {
            "type": spec.storage_type,
            "repo_id": spec.hf_repo_id,
            "repo_type": spec.repo_type,
            "requires_token": spec.requires_token,
            "token_secret": spec.token_secret,
        },
        "status": "planned",
        "fileset_exists": None,
        "model_entity_exists": None,
        "model_spec_populated": None,
    }


def build_manifest(
    plans: list[dict[str, Any]],
    *,
    workspace: str,
    nmp_base_url: str,
    dry_run: bool,
    create_missing: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "platform_model_entities.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "workspace": workspace,
        "nmp_base_url": nmp_base_url,
        "dry_run": dry_run,
        "create_missing": create_missing,
        "models": plans,
    }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _obj_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    return status_code


def _is_not_found(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return _status_code(exc) == 404 or "notfound" in name or "not_found" in name


def _is_conflict(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return _status_code(exc) == 409 or "conflict" in name


def _create_huggingface_fileset(client: Any, spec: PlatformModelEntitySpec, workspace: str) -> Any:
    from nemo_platform.types.files import HuggingfaceStorageConfigParam  # type: ignore

    storage_args = {
        "type": "huggingface",
        "repo_id": spec.hf_repo_id,
        "repo_type": spec.repo_type,
    }
    if spec.token_secret:
        storage_args["token_secret"] = spec.token_secret
    return client.files.filesets.create(
        workspace=workspace,
        name=spec.fileset_name,
        description=spec.description,
        storage=HuggingfaceStorageConfigParam(**storage_args),
    )


def _verify_or_create_plan(
    client: Any,
    spec: PlatformModelEntitySpec,
    workspace: str,
    *,
    create_missing: bool,
) -> dict[str, Any]:
    plan = build_model_entity_plan(spec, workspace)
    errors: list[str] = []

    fileset = None
    try:
        fileset = client.files.filesets.retrieve(workspace=workspace, name=spec.fileset_name)
    except Exception as exc:  # noqa: BLE001 - SDK exception types are version-specific.
        if not _is_not_found(exc):
            errors.append(f"fileset retrieve failed: {exc}")
        elif create_missing:
            try:
                fileset = _create_huggingface_fileset(client, spec, workspace)
            except Exception as create_exc:  # noqa: BLE001
                if _is_conflict(create_exc):
                    try:
                        fileset = client.files.filesets.retrieve(
                            workspace=workspace,
                            name=spec.fileset_name,
                        )
                    except Exception as retrieve_exc:  # noqa: BLE001
                        errors.append(f"fileset conflict retrieve failed: {retrieve_exc}")
                else:
                    errors.append(f"fileset create failed: {create_exc}")

    plan["fileset_exists"] = fileset is not None
    if fileset is not None:
        plan["fileset"] = _jsonable(fileset)

    model = None
    try:
        model = client.models.retrieve(workspace=workspace, name=spec.model_entity_name)
    except Exception as exc:  # noqa: BLE001
        if not _is_not_found(exc):
            errors.append(f"model retrieve failed: {exc}")
        elif create_missing and fileset is not None:
            try:
                model = client.models.create(
                    workspace=workspace,
                    name=spec.model_entity_name,
                    fileset=f"{workspace}/{spec.fileset_name}",
                    description=spec.description,
                )
            except Exception as create_exc:  # noqa: BLE001
                if _is_conflict(create_exc):
                    try:
                        model = client.models.retrieve(
                            workspace=workspace,
                            name=spec.model_entity_name,
                        )
                    except Exception as retrieve_exc:  # noqa: BLE001
                        errors.append(f"model conflict retrieve failed: {retrieve_exc}")
                else:
                    errors.append(f"model create failed: {create_exc}")

    plan["model_entity_exists"] = model is not None
    if model is not None:
        model_dict = _jsonable(model)
        plan["model_entity"] = model_dict
        plan["model_spec_populated"] = _obj_get(model_dict, "spec") is not None
    else:
        plan["model_spec_populated"] = False

    if errors:
        plan["status"] = "error"
        plan["errors"] = errors
    elif model is None or fileset is None:
        plan["status"] = "missing"
    elif not plan["model_spec_populated"]:
        plan["status"] = "waiting_for_spec"
    else:
        plan["status"] = "ready"
    return plan


def run(config: PlatformModelEntityConfig) -> dict[str, Any]:
    specs = selected_model_specs(
        config.models,
        hf_token_secret=config.hf_token_secret,
    )
    live = (config.verify or config.create_missing) and not config.dry_run

    if live:
        try:
            from nemo_platform import NeMoPlatform  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "NeMo Platform SDK is not installed. Install nemo-platform or rerun "
                "with --dry-run to generate a model entity plan."
            ) from exc
        client = NeMoPlatform(base_url=config.nmp_base_url, workspace=config.workspace)
        plans = [
            _verify_or_create_plan(
                client,
                spec,
                config.workspace,
                create_missing=config.create_missing,
            )
            for spec in specs
        ]
    else:
        plans = [build_model_entity_plan(spec, config.workspace) for spec in specs]

    manifest = build_manifest(
        plans,
        workspace=config.workspace,
        nmp_base_url=config.nmp_base_url,
        dry_run=not live,
        create_missing=config.create_missing and live,
    )
    if config.out:
        config.out.parent.mkdir(parents=True, exist_ok=True)
        config.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or verify NeMo Platform Model Entities for Customizer jobs."
    )
    parser.add_argument("--workspace", default=default_nmp_workspace())
    parser.add_argument("--nmp-base-url", default=default_nmp_base_url())
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(DEFAULT_MODEL_SPECS),
        default=(),
        help="Source model slugs to include. Defaults to all Stage 3 train bases.",
    )
    parser.add_argument(
        "--hf-token-secret",
        default=os.getenv("NMP_HF_TOKEN_SECRET"),
        help="NeMo Platform secret name containing a Hugging Face token.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Create missing Hugging Face FileSets and Model Entities through the SDK.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = run(
        PlatformModelEntityConfig(
            workspace=args.workspace,
            nmp_base_url=args.nmp_base_url,
            models=tuple(args.models),
            hf_token_secret=args.hf_token_secret,
            out=args.out,
            verify=args.verify,
            create_missing=args.create_missing,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
