#!/usr/bin/env python3
"""Inspect a PEFT LoRA adapter checkpoint without importing torch.

This is a structural check for merged adapter directories. It verifies required
files, reads `adapter_config.json`, parses the safetensors header, and reports
tensor counts/dtypes. It does not prove that NIM can load the adapter; live NIM
loading remains the final viability check.
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")
OPTIONAL_AUX_FILES = (
    "automodel_peft_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def _product(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read a safetensors header without loading tensor data."""
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError("safetensors file is too short to contain a header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError("safetensors header is truncated")
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    header["_header_len"] = header_len
    return header


def inspect_adapter_dir(path: Path) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []

    files = {name: path / name for name in REQUIRED_FILES + OPTIONAL_AUX_FILES}
    for name in REQUIRED_FILES:
        if not files[name].exists():
            errors.append(f"missing required file: {name}")

    config: dict[str, Any] = {}
    if files["adapter_config.json"].exists():
        try:
            config = json.loads(files["adapter_config.json"].read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"adapter_config.json is invalid JSON: {exc}")

    if config:
        if not config.get("base_model_name_or_path"):
            warnings.append("adapter_config.json has no base_model_name_or_path")
        if not config.get("task_type"):
            warnings.append("adapter_config.json has no task_type")
        target_modules = config.get("target_modules")
        if not target_modules:
            warnings.append("adapter_config.json has no target_modules")
        elif isinstance(target_modules, list) and len(target_modules) < 16:
            warnings.append(
                "adapter_config.json has a small generic target_modules list; "
                "loadability depends on base model name matching"
            )

    tensor_summary: dict[str, Any] = {}
    if files["adapter_model.safetensors"].exists():
        try:
            header = read_safetensors_header(files["adapter_model.safetensors"])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"adapter_model.safetensors header is unreadable: {exc}")
        else:
            tensor_items = [(k, v) for k, v in header.items()
                            if k not in {"__metadata__", "_header_len"}]
            dtypes = Counter(v.get("dtype") for _, v in tensor_items)
            total_elements = sum(_product(v.get("shape", [])) for _, v in tensor_items)
            lora_a = [k for k, _ in tensor_items if "lora_A" in k]
            lora_b = [k for k, _ in tensor_items if "lora_B" in k]
            non_lora = [
                k for k, _ in tensor_items
                if "lora_A" not in k and "lora_B" not in k
            ]
            if not lora_a or not lora_b:
                errors.append("safetensors file does not contain both lora_A and lora_B tensors")
            if non_lora:
                warnings.append(
                    f"safetensors contains {len(non_lora)} non-LoRA tensors; "
                    "confirm these are expected for the base architecture"
                )
            tensor_summary = {
                "header_bytes": header["_header_len"],
                "tensor_count": len(tensor_items),
                "dtype_counts": dict(sorted(dtypes.items())),
                "total_elements": total_elements,
                "lora_a_count": len(lora_a),
                "lora_b_count": len(lora_b),
                "non_lora_count": len(non_lora),
                "first_tensor_keys": [k for k, _ in tensor_items[:8]],
            }

    status = "error" if errors else "warn" if warnings else "ok"
    return {
        "path": str(path),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "required_files": {
            name: files[name].exists() for name in REQUIRED_FILES
        },
        "aux_files": {
            name: files[name].exists() for name in OPTIONAL_AUX_FILES
        },
        "config": {
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "target_modules_count": (
                len(config.get("target_modules"))
                if isinstance(config.get("target_modules"), list)
                else None
            ),
            "config_key_count": len(config),
        },
        "tensors": tensor_summary,
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"Adapter: {report['path']}")
    print(f"Status:  {report['status']}")
    if report["errors"]:
        print("Errors:")
        for item in report["errors"]:
            print(f"  - {item}")
    if report["warnings"]:
        print("Warnings:")
        for item in report["warnings"]:
            print(f"  - {item}")
    print("Required files:")
    for name, exists in report["required_files"].items():
        print(f"  - {name}: {'present' if exists else 'missing'}")
    print("Config:")
    for key, value in report["config"].items():
        print(f"  - {key}: {value}")
    if report["tensors"]:
        print("Tensors:")
        for key in (
            "tensor_count",
            "dtype_counts",
            "total_elements",
            "lora_a_count",
            "lora_b_count",
            "non_lora_count",
            "header_bytes",
        ):
            print(f"  - {key}: {report['tensors'][key]}")


def write_json_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("adapter_dir", type=Path)
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    ap.add_argument(
        "--output-json",
        type=Path,
        help="Write machine-readable JSON report to this path",
    )
    ap.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero for warnings as well as errors",
    )
    args = ap.parse_args()

    report = inspect_adapter_dir(args.adapter_dir)
    if args.output_json:
        write_json_report(report, args.output_json)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_summary(report)
    if report["status"] == "error" or (args.fail_on_warn and report["status"] == "warn"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
