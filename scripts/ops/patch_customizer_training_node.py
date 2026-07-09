#!/usr/bin/env python3
"""Patch NeMo Customizer training placement to a specific Kubernetes node."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml


def kubectl(args: list[str]) -> str:
    return subprocess.check_output(["kubectl", *args], text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="nemo-peft")
    parser.add_argument("--configmap", default="nemo-platform-customizer-config")
    parser.add_argument("--node", required=True)
    parser.add_argument("--backup-dir", default="archive/cluster-backups")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--deployment", default="nemo-platform-customizer")
    args = parser.parse_args()

    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = backup_dir / f"{args.configmap}-pre-node-{args.node}-{ts}.yaml"

    cm_yaml = kubectl(
        ["get", "configmap", args.configmap, "-n", args.namespace, "-o", "yaml"]
    )
    backup_path.write_text(cm_yaml)

    cm = json.loads(
        kubectl(["get", "configmap", args.configmap, "-n", args.namespace, "-o", "json"])
    )
    cfg = yaml.safe_load(cm["data"]["config.yaml"])

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

    if args.restart:
        subprocess.run(
            [
                "kubectl",
                "rollout",
                "restart",
                f"deployment/{args.deployment}",
                "-n",
                args.namespace,
            ],
            check=True,
        )

    print(f"backup={backup_path}")
    print(f"training.nodeSelectors={training['nodeSelectors']}")
    print(
        "training.container_defaults.nodeSelector="
        f"{training['container_defaults']['nodeSelector']}"
    )
    if args.restart:
        print(f"restarted=deployment/{args.deployment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
