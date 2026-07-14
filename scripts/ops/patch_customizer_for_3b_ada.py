#!/usr/bin/env python3
"""Enable Llama 3.2 3B BF16 Customizer training on the Ada node."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml


TARGET_KEY = "meta/llama-3.2-3b-instruct@2.0"
TEMPLATE_KEY = "meta/llama-3.2-3b-instruct@v1.0.0+80GB"


def kubectl(args: list[str]) -> str:
    return subprocess.check_output(["kubectl", *args], text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="nemo-peft")
    parser.add_argument("--configmap", default="nemo-platform-customizer-config")
    parser.add_argument("--node", default="<ADA_NODE>")
    parser.add_argument("--backup-dir", default="archive/cluster-backups")
    args = parser.parse_args()

    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = backup_dir / f"{args.configmap}-pre-3b-ada-{ts}.yaml"

    cm_yaml = kubectl(
        ["get", "configmap", args.configmap, "-n", args.namespace, "-o", "yaml"]
    )
    backup_path.write_text(cm_yaml)

    cm = json.loads(
        kubectl(["get", "configmap", args.configmap, "-n", args.namespace, "-o", "json"])
    )
    cfg = yaml.safe_load(cm["data"]["config.yaml"])

    target = cfg["customizationTargets"]["targets"][TARGET_KEY]
    target["enabled"] = True

    node_selector = {"kubernetes.io/hostname": args.node}
    training = cfg["training"]
    training["nodeSelectors"] = node_selector
    training.setdefault("container_defaults", {})["nodeSelector"] = node_selector

    patch = {"data": {"config.yaml": yaml.safe_dump(cfg, sort_keys=False)}}
    subprocess.run(
        [
            "kubectl",
            "patch",
            "configmap",
            args.configmap,
            "-n",
            args.namespace,
            "--type",
            "merge",
            "-p",
            json.dumps(patch),
        ],
        check=True,
    )

    training_option = cfg["customizationConfigTemplates"]["templates"][TEMPLATE_KEY][
        "training_options"
    ][0]
    print(f"backup={backup_path}")
    print(f"{TARGET_KEY}.enabled={target['enabled']}")
    print(f"{TARGET_KEY}.precision={target['precision']}")
    print(f"{TEMPLATE_KEY}.num_gpus={training_option.get('num_gpus')}")
    print(f"training.nodeSelectors={training['nodeSelectors']}")
    print(
        "training.container_defaults.nodeSelector="
        f"{training['container_defaults']['nodeSelector']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
