"""Build 2-way {prompt, completion} shards for Nemotron-3-Nano-30B-A3B TIES SFT.

Source: adapter_{train,val}.jsonl produced by holdout_split.py under
  /mnt/nvme2/peft/datasets/v2/<collection>/
Each row has {prompt, completion, system}.

  shard-a: first 50% of stratified-shuffled train rows
  shard-b: second 50% of stratified-shuffled train rows
  validation: full val set — SHARED across both shards so val_loss curves
              are directly comparable

Datasets named stage3-<corpus>-shard-{a,b} in the data-store.

Ported from nim-sft-final/scripts/build_2way_pc_full.py — adapted for:
  - stage3 collection paths (adapter_train.jsonl / adapter_val.jsonl)
  - stratified shuffle by `stage` field (falls back to plain shuffle)
  - Platform-aware service configuration via NMP_* or legacy service env vars

Usage:
  python3 scripts/stage3/build_moe_shards.py --collection nim_curated [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.nemo_platform import (  # noqa: E402
    default_data_store_git_base,
    default_data_store_hf_endpoint,
    default_data_store_url,
    default_entity_store_url,
    default_nmp_workspace,
)

SHARDS = ["a", "b"]

DEFAULT_DATASET_NAMESPACE = default_nmp_workspace()
DEFAULT_ENTITY_STORE_URL = default_entity_store_url()
DEFAULT_DATA_STORE_URL = default_data_store_url()
DEFAULT_DATA_STORE_GIT_BASE = default_data_store_git_base()
DEFAULT_DATA_STORE_HF_ENDPOINT = default_data_store_hf_endpoint()

# Canonical source paths
_CORPUS_BASE = Path("/mnt/nvme2/peft/datasets/v2")
_COLLECTION_TO_DIR = {
    "nim_curated": _CORPUS_BASE / "nim_curated",
    "nemo_usvcs_curated": _CORPUS_BASE / "nemo_usvcs_curated",
}


@dataclass(frozen=True)
class ServiceConfig:
    namespace: str
    entity_store_url: str
    data_store_url: str
    data_store_git_base: str
    data_store_hf_endpoint: str
    data_store_user: str | None
    data_store_password: str | None

    @property
    def data_store_auth(self) -> tuple[str, str] | None:
        if not self.data_store_user:
            return None
        return (self.data_store_user, self.data_store_password or "")

    @property
    def authenticated_git_base(self) -> str:
        return inject_basic_auth(
            self.data_store_git_base,
            self.data_store_user,
            self.data_store_password,
        )


def inject_basic_auth(base_url: str, user: str | None, password: str | None) -> str:
    base_url = base_url.rstrip("/")
    if not user:
        return base_url
    parts = urlsplit(base_url)
    if parts.username:
        return base_url
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"Data Store Git base must be an absolute URL, got: {base_url!r}")
    auth = quote(user, safe="")
    if password is not None:
        auth = f"{auth}:{quote(password, safe='')}"
    return urlunsplit(
        (
            parts.scheme,
            f"{auth}@{parts.netloc}",
            parts.path.rstrip("/"),
            parts.query,
            parts.fragment,
        )
    )


# ---------------------------------------------------------------------------
# Pure functions (extractable, tested independently)
# ---------------------------------------------------------------------------

def shard_dataset_name(collection: str, shard: str) -> str:
    """Return the data-store dataset name for a collection+shard combo.

    Examples:
      nim_curated, a  → stage3-nim-curated-shard-a
      nemo_usvcs_curated, b → stage3-nemo-usvcs-curated-shard-b
    """
    corpus_slug = collection.replace("_", "-")
    return f"stage3-{corpus_slug}-shard-{shard}"


def fold_system_into_prompt(rows: list[dict]) -> list[dict]:
    """{prompt, completion, system} → {prompt: f'{system}\\n\\n{prompt}', completion}.

    System field folded into prompt with '\\n\\n' separator because the
    Nemotron Customizer template '{prompt} {completion}' doesn't reference a
    system field. Empty/missing system → prompt unchanged.
    Returns a new list; original rows are not mutated.
    """
    out = []
    for r in rows:
        system = r.get("system", "").strip()
        user_prompt = r["prompt"].strip()
        prompt = f"{system}\n\n{user_prompt}" if system else user_prompt
        out.append({"prompt": prompt, "completion": r["completion"]})
    return out


def stratified_2way_split(
    rows: list[dict],
    stratify_key: str = "stage",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified 2-way split returning (shard_a, shard_b).

    Per-stratum even split preserves class distribution across both shards.
    Falls back to plain random shuffle when `stratify_key` is absent from all rows.

    The split is deterministic for a given seed.
    """
    has_key = any(stratify_key in r for r in rows)

    if not has_key:
        # Plain random shuffle fallback
        rng = random.Random(seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        return shuffled[:half], shuffled[half:]

    # Stratified split: bucket by stratum, shuffle each bucket independently,
    # then interleave even/odd halves into shard-a and shard-b.
    by_stratum: dict[str, list[dict]] = {}
    for r in rows:
        stratum = r.get(stratify_key, "__missing__")
        by_stratum.setdefault(stratum, []).append(r)

    rng = random.Random(seed)
    shard_a: list[dict] = []
    shard_b: list[dict] = []
    for stratum in sorted(by_stratum.keys()):
        bucket = list(by_stratum[stratum])
        rng.shuffle(bucket)
        half = len(bucket) // 2
        shard_a.extend(bucket[:half])
        shard_b.extend(bucket[half:])

    # Final shuffle of each shard so rows from different strata are mixed
    rng.shuffle(shard_a)
    rng.shuffle(shard_b)
    return shard_a, shard_b


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def _sh(*args, cwd=None, allow_fail=False):
    r = subprocess.run(args, cwd=cwd, capture_output=True)
    if r.returncode != 0 and not allow_fail:
        sys.stderr.write(r.stderr.decode())
        raise subprocess.CalledProcessError(r.returncode, args)
    return r


# ---------------------------------------------------------------------------
# Data-store operations (integration; not unit-tested)
# ---------------------------------------------------------------------------

def create_dataset_repo(name: str, config: ServiceConfig) -> None:
    repo_id = f"{config.namespace}/{name}"
    r = httpx.post(
        f"{config.data_store_url.rstrip('/')}/v1/hf/api/repos/create",
        auth=config.data_store_auth,
        json={
            "type": "dataset",
            "name": name,
            "organization": config.namespace,
            "private": False,
        },
        timeout=15,
    )
    if r.status_code in (200, 201):
        print(f"  created {repo_id}")
    elif r.status_code == 409 or "already created" in r.text:
        # Gitea returns 409 on duplicate repo; keep the text-sniff as a fallback
        # in case the API version differs.
        print(f"  {repo_id} already exists")
    else:
        print(f"  create_repo failed [{r.status_code}]: {r.text}")
        sys.exit(1)


def push_shard_files(
    name: str,
    train_rows: list[dict],
    val_rows: list[dict],
    config: ServiceConfig,
) -> None:
    repo_id = f"{config.namespace}/{name}"
    push_url = f"{config.authenticated_git_base.rstrip('/')}/{repo_id}.git"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _sh("git", "clone", push_url, str(td / "repo"))
        repo = td / "repo"
        with (repo / "training.jsonl").open("w") as f:
            for r in train_rows:
                f.write(json.dumps(r) + "\n")
        with (repo / "validation.jsonl").open("w") as f:
            for r in val_rows:
                f.write(json.dumps(r) + "\n")
        _sh("git", "config", "user.email", "moe-shards@nemo-peft.local", cwd=repo)
        _sh("git", "config", "user.name", "moe-shard-uploader", cwd=repo)
        _sh("git", "add", "training.jsonl", "validation.jsonl", cwd=repo)
        commit_msg = (
            f"stage3 MoE shard {name}: {len(train_rows)} train + "
            f"{len(val_rows)} val (stratified, shared val)"
        )
        r = _sh("git", "commit", "-m", commit_msg, cwd=repo, allow_fail=True)
        if r.returncode != 0:
            if b"nothing to commit" in r.stdout or b"nothing to commit" in r.stderr:
                print(f"  no changes to push for {name}")
                return
            sys.stderr.write(r.stderr.decode())
            sys.exit(1)
        _sh("git", "push", "origin", "main", cwd=repo)
    print(f"  pushed {repo_id}: {len(train_rows)} train + {len(val_rows)} val")


def register_in_entity_store(
    name: str,
    n_train: int,
    n_val: int,
    collection: str,
    config: ServiceConfig,
) -> None:
    shard_letter = name.split("-")[-1].upper()
    payload = {
        "name": name,
        "namespace": config.namespace,
        "description": (
            f"Nemotron-3-Nano-30B-A3B 2-way TIES SFT prompt/completion — "
            f"shard {shard_letter} ({n_train} train rows, stratified by stage, "
            f"seed=42, system folded into prompt); full validation set shared "
            f"across shards ({n_val} val rows) for directly comparable val_loss "
            f"curves. Collection: {collection}."
        ),
        "format": "hf",
        "files_url": f"hf://datasets/{config.namespace}/{name}",
        "hf_endpoint": config.data_store_hf_endpoint,
    }
    r = httpx.post(f"{config.entity_store_url.rstrip('/')}/v1/datasets", json=payload, timeout=30)
    if r.status_code in (200, 201):
        print("  registered in entity-store")
    elif r.status_code == 409:
        r2 = httpx.patch(
            f"{config.entity_store_url.rstrip('/')}/v1/datasets/{config.namespace}/{name}",
            json={"format": "hf", "hf_endpoint": config.data_store_hf_endpoint},
            timeout=15,
        )
        print(f"  already in entity-store, patched ({r2.status_code})")
    else:
        print(f"  entity-store register failed [{r.status_code}]: {r.text}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build 2-way MoE shards and register in entity-store + data-store."
    )
    ap.add_argument(
        "--collection",
        required=True,
        choices=list(_COLLECTION_TO_DIR.keys()),
        help="Which stage3 corpus to shard",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--namespace", default=DEFAULT_DATASET_NAMESPACE)
    ap.add_argument("--entity-store-url", default=DEFAULT_ENTITY_STORE_URL)
    ap.add_argument("--data-store-url", default=DEFAULT_DATA_STORE_URL)
    ap.add_argument("--data-store-git-base", default=DEFAULT_DATA_STORE_GIT_BASE)
    ap.add_argument("--data-store-hf-endpoint", default=DEFAULT_DATA_STORE_HF_ENDPOINT)
    ap.add_argument("--data-store-user", default=os.getenv("DATA_STORE_USER"))
    ap.add_argument("--data-store-password", default=os.getenv("DATA_STORE_PASSWORD"))
    args = ap.parse_args()

    config = ServiceConfig(
        namespace=args.namespace,
        entity_store_url=args.entity_store_url,
        data_store_url=args.data_store_url,
        data_store_git_base=args.data_store_git_base,
        data_store_hf_endpoint=args.data_store_hf_endpoint,
        data_store_user=args.data_store_user,
        data_store_password=args.data_store_password,
    )

    collection = args.collection
    seed = args.seed
    corpus_dir = _COLLECTION_TO_DIR[collection]

    train_path = corpus_dir / "adapter_train.jsonl"
    val_path = corpus_dir / "adapter_val.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run holdout_split.py first.")
        return 1
    if not val_path.exists():
        print(f"ERROR: {val_path} not found. Run holdout_split.py first.")
        return 1

    train_rows_raw = _load_jsonl(train_path)
    val_rows_raw = _load_jsonl(val_path)
    print(
        f"Source pools: {len(train_rows_raw)} train, {len(val_rows_raw)} val "
        f"({{prompt,completion,system}} format)"
    )

    # Verify row format — peek at first row
    sample_row = train_rows_raw[0] if train_rows_raw else {}
    if "prompt" not in sample_row or "completion" not in sample_row:
        print(f"ERROR: unexpected row format: {list(sample_row.keys())}")
        return 1
    has_system = "system" in sample_row
    print(f"Row fields: {list(sample_row.keys())} (has_system={has_system})")

    # Fold system into prompt, convert to {prompt, completion}
    train_pc = fold_system_into_prompt(train_rows_raw)
    val_pc = fold_system_into_prompt(val_rows_raw)
    print(f"After fold: {len(train_pc)} train, {len(val_pc)} val ({{prompt,completion}})")

    # Stratified 2-way split on the original rows (which carry `stage` if present)
    # We split train_rows_raw so we keep the `stage` field for stratification,
    # then map indices back to the folded version.
    shard_a_raw, shard_b_raw = stratified_2way_split(
        train_rows_raw, stratify_key="stage", seed=seed
    )
    # Re-fold the split rows (stratification used train_rows_raw which has system)
    shard_a = fold_system_into_prompt(shard_a_raw)
    shard_b = fold_system_into_prompt(shard_b_raw)
    print(f"Shard sizes: a={len(shard_a)}, b={len(shard_b)} (sum={len(shard_a)+len(shard_b)})")
    print(f"Validation set: {len(val_pc)} rows — SHARED on both shards")

    # Sample row preview
    if shard_a:
        s = shard_a[0]
        print("\nSample shard-a row 0:")
        print(f"  prompt    ({len(s['prompt'])} chars): {s['prompt'][:200]!r}...")
        print(f"  completion ({len(s['completion'])} chars): {s['completion'][:200]!r}...")

    train_shards = {"a": shard_a, "b": shard_b}

    for letter in SHARDS:
        name = shard_dataset_name(collection, letter)
        print(f"\n=== {name} ===")
        create_dataset_repo(name, config)
        push_shard_files(name, train_shards[letter], val_pc, config)
        register_in_entity_store(
            name,
            len(train_shards[letter]),
            len(val_pc),
            collection,
            config,
        )

    print(f"\nBoth MoE shards ready for {collection} TIES run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
