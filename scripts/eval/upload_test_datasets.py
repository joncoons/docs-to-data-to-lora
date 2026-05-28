"""One-off uploader for the Stage 3 test-set datasets.

For each of the two Stage 3 collections, this script:
  1. Creates a HF dataset repo in NeMo Data Store (Gitea) via the HF Hub API
  2. Pushes the test_set.jsonl (from holdout_split output on NFS) to that repo
     as `test.jsonl`
  3. Registers the dataset entity in NeMo Entity Store

Idempotent: existing repo / entity = patch + skip.

Source files (already on NFS, produced by holdout_split.py):
  /mnt/nvme2/peft/datasets/v2/nim_curated/test_set.jsonl       (540 rows)
  /mnt/nvme2/peft/datasets/v2/nemo_usvcs_curated/test_set.jsonl (462 rows)

Resulting entities (in entity-store):
  default/stage3-nim-curated-test
  default/stage3-nemo-usvcs-curated-test
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

DATASET_NAMESPACE = "default"
ENTITY_STORE = "http://192.168.1.187:30911"
DATA_STORE = "http://192.168.1.187:30912"
GITEA_USER = "datastore_admin"
GITEA_PASS = "nemo-peft-ds"

COLLECTIONS = [
    # Originals (bare questions, no retrieved context).
    {
        "src":        Path("/mnt/nvme2/peft/datasets/v2/nim_curated/test_set.jsonl"),
        "dataset":    "stage3-nim-curated-test",
        "collection": "nim_curated",
        "context":    False,
    },
    {
        "src":        Path("/mnt/nvme2/peft/datasets/v2/nemo_usvcs_curated/test_set.jsonl"),
        "dataset":    "stage3-nemo-usvcs-curated-test",
        "collection": "nemo_usvcs_curated",
        "context":    False,
    },
    # Context-baked variants (output of bake_context_into_testset.py).
    # vdb_top_k=25, rerank_top_k=5, soft cutoff at sigmoid score >= 0.5.
    # Each row's prompt is `Context:\n[1] <url>\n<chunk>\n...\n\nQuestion: <q>`.
    {
        "src":        Path("/mnt/nvme2/peft/datasets/v2/nim_curated/test_set_with_context.jsonl"),
        "dataset":    "stage3-nim-curated-test-with-context",
        "collection": "nim_curated",
        "context":    True,
    },
    {
        "src":        Path("/mnt/nvme2/peft/datasets/v2/nemo_usvcs_curated/test_set_with_context.jsonl"),
        "dataset":    "stage3-nemo-usvcs-curated-test-with-context",
        "collection": "nemo_usvcs_curated",
        "context":    True,
    },
]


def _sh(*args: str, cwd: str | None = None, allow_fail: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(args, cwd=cwd, capture_output=True)
    if r.returncode != 0 and not allow_fail:
        sys.stderr.write(r.stderr.decode())
        sys.exit(r.returncode)
    return r


def create_repo(name: str) -> None:
    r = httpx.post(
        f"{DATA_STORE}/v1/hf/api/repos/create",
        auth=(GITEA_USER, GITEA_PASS),
        json={
            "type": "dataset",
            "name": name,
            "organization": DATASET_NAMESPACE,
            "private": False,
        },
        timeout=15,
    )
    if r.status_code in (200, 201):
        print(f"  created {DATASET_NAMESPACE}/{name}")
    elif r.status_code == 409 or "already created" in r.text:
        print(f"  {DATASET_NAMESPACE}/{name} already exists")
    else:
        print(f"  create_repo failed [{r.status_code}]: {r.text}")
        sys.exit(1)


def push_test_file(name: str, src: Path) -> None:
    repo_id = f"{DATASET_NAMESPACE}/{name}"
    push_url = f"http://{GITEA_USER}:{GITEA_PASS}@192.168.1.187:30912/{repo_id}.git"
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _sh("git", "clone", push_url, str(td_path / "repo"))
        repo = td_path / "repo"
        shutil.copyfile(src, repo / "test.jsonl")
        _sh("git", "config", "user.email", "stage3-test-uploader@nemo-peft.local", cwd=str(repo))
        _sh("git", "config", "user.name", "stage3-test-uploader", cwd=str(repo))
        _sh("git", "add", "test.jsonl", cwd=str(repo))
        with src.open() as f:
            n_rows = sum(1 for _ in f)
        commit_msg = f"stage3 test-set: {n_rows} rows from holdout_split (10% KVP holdout, seed=42)"
        r = _sh("git", "commit", "-m", commit_msg, cwd=str(repo), allow_fail=True)
        if r.returncode != 0:
            if b"nothing to commit" in r.stdout or b"nothing to commit" in r.stderr:
                print(f"  no changes to push for {name}")
                return
            sys.stderr.write(r.stderr.decode())
            sys.exit(1)
        _sh("git", "push", "origin", "main", cwd=str(repo))
    print(f"  pushed {repo_id}: test.jsonl ({n_rows} rows)")


def register_in_entity_store(name: str, collection: str, n_rows: int,
                              context_baked: bool = False) -> None:
    if context_baked:
        desc = (
            f"Stage 3 held-out test set for {collection} ({n_rows} rows). "
            f"Retrieval-baked variant: each row's prompt has top-5 reranked "
            f"chunks (vdb_top_k=25, sigmoid>=0.5) prepended as Context. "
            f"Produced by bake_context_into_testset.py."
        )
    else:
        desc = (
            f"Stage 3 held-out test set for {collection} ({n_rows} rows, "
            f"10% KVP holdout, seed 42; bare questions; from holdout_split.py)"
        )
    payload = {
        "name": name,
        "namespace": DATASET_NAMESPACE,
        "description": desc,
        "format": "hf",
        "files_url": f"hf://datasets/{DATASET_NAMESPACE}/{name}",
        "hf_endpoint": "http://nemo-data-store:3000/v1/hf",
    }
    r = httpx.post(f"{ENTITY_STORE}/v1/datasets", json=payload, timeout=30)
    if r.status_code in (200, 201):
        print(f"  registered in entity-store")
    elif r.status_code == 409:
        r2 = httpx.patch(
            f"{ENTITY_STORE}/v1/datasets/{DATASET_NAMESPACE}/{name}",
            json={"format": "hf", "hf_endpoint": "http://nemo-data-store:3000/v1/hf"},
            timeout=15,
        )
        print(f"  already in entity-store, patched ({r2.status_code})")
    else:
        print(f"  entity-store register failed [{r.status_code}]: {r.text}")
        sys.exit(1)


def main() -> int:
    for spec in COLLECTIONS:
        src = spec["src"]
        if not src.exists():
            print(f"FATAL: source missing: {src}")
            return 1
        print(f"=== {spec['dataset']} ({spec['collection']}) ===")
        create_repo(spec["dataset"])
        push_test_file(spec["dataset"], src)
        with src.open() as f:
            n_rows = sum(1 for _ in f)
        register_in_entity_store(spec["dataset"], spec["collection"], n_rows,
                                  context_baked=spec.get("context", False))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
