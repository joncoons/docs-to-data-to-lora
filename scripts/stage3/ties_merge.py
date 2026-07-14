"""TIES merge of N LoRA adapters trained via NeMo Customizer parallel-shard SFT.

For each LoRA tensor (lora_A or lora_B matrix) across the N adapters:
  1. Trim:  keep only the top (1 - trim_ratio) magnitude entries; zero the rest
  2. Sign-elect: per-parameter, pick the sign with the largest total magnitude across adapters
  3. Disjoint merge: average ONLY the entries that agree with the elected sign
Reference: Yadav et al. 2023, "TIES-Merging" (arXiv 2306.01708)

Usage, cloning source adapters from NeMo Data Store:
  DATA_STORE_GIT_BASE=http://nemo-data-store:3000 \
  DATA_STORE_USER=... DATA_STORE_PASSWORD=... \
  python3 scripts/stage3/ties_merge.py \
    --adapters \
        default/lora-nim-nemotron-nano-30b-r16-shard-a@cust-AAAA \
        default/lora-nim-nemotron-nano-30b-r16-shard-b@cust-BBBB \
    --out <ARTIFACT_ROOT>/checkpoints/lora/nemotron-nano-30b-stage3-tiesmerged \
    --trim-ratio 0.2

Usage, reading already-mounted adapter directories:
  python3 scripts/stage3/ties_merge.py \
    --adapter-dirs /inputs/shard-a /inputs/shard-b \
    --out /outputs/nemotron-nano-30b-stage3-tiesmerged \
    --trim-ratio 0.2

Both modes load adapter_model.safetensors, run TIES, and write a merged adapter
at --out with auxiliary files copied from the first adapter so it loads
identically in a NIM via NIM_PEFT_SOURCE.

Ported from nim-sft-final/scripts/ties_merge.py and adapted for:
  - K8s-native Data Store or mounted-PVC source adapters
  - stage3 context (adapter naming, doc strings)
  - Algorithm and math are preserved verbatim from the validated original.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import safetensors.torch
import torch


DEFAULT_DATA_STORE_GIT_BASE = "http://nemo-data-store:3000"


# ---------------------------------------------------------------------------
# Pure function - tested in test_stage3_ties_merge.py
# ---------------------------------------------------------------------------

def ties_merge_one_tensor(stacked: torch.Tensor, trim_ratio: float) -> torch.Tensor:
    """stacked shape: (N, ...); returns merged tensor of shape (...)

    Algorithm:
      1. Trim - per-adapter, zero everything below (1 - trim_ratio) quantile of |value|
      2. Sign-elect - sign with largest summed magnitude across adapters wins per parameter
      3. Disjoint merge - average only entries whose sign agrees with the elected sign
    """
    n = stacked.shape[0]

    # 1. Trim - per-adapter, zero everything below (1 - trim_ratio) quantile of |value|
    flat = stacked.reshape(n, -1)
    if trim_ratio > 0:
        k = int(round((1 - trim_ratio) * flat.shape[1]))   # keep top k entries per adapter
        k = max(1, k)
        # threshold = the k-th largest |value| per adapter
        absvals = flat.abs()
        thresh, _ = torch.kthvalue(absvals, flat.shape[1] - k + 1, dim=1, keepdim=True)
        mask = absvals >= thresh
        flat = flat * mask

    # 2. Sign elect - sign with largest summed magnitude across adapters wins per parameter
    sign_score = flat.sum(dim=0)        # signed sum; sign is the consensus sign
    elected_sign = torch.sign(sign_score)
    # if score is exactly 0, fall back to majority-sign (or just 0)

    # 3. Disjoint merge - average only entries whose sign agrees with the elected sign
    elected_sign_b = elected_sign.unsqueeze(0).expand_as(flat)
    agree_mask = (torch.sign(flat) == elected_sign_b) & (flat != 0)
    sums = (flat * agree_mask).sum(dim=0)
    counts = agree_mask.sum(dim=0).clamp(min=1)            # avoid divide-by-zero
    merged = sums / counts

    return merged.reshape(stacked.shape[1:])


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def _inject_basic_auth(base_url: str, user: str | None, password: str | None) -> str:
    """Return base_url with URL-encoded basic auth if credentials are supplied."""
    base_url = base_url.rstrip("/")
    if not user:
        return base_url

    parts = urlsplit(base_url)
    if parts.username:
        return base_url
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"DATA_STORE_GIT_BASE must be an absolute URL, got: {base_url!r}")

    auth = quote(user, safe="")
    if password is not None:
        auth = f"{auth}:{quote(password, safe='')}"
    netloc = f"{auth}@{parts.netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), parts.query, parts.fragment))


def data_store_git_base(cli_value: str | None = None) -> str:
    """Resolve the Data Store Git base URL from CLI/env without hardcoded secrets."""
    base_url = cli_value or os.getenv("DATA_STORE_GIT_BASE") or DEFAULT_DATA_STORE_GIT_BASE
    return _inject_basic_auth(
        base_url,
        os.getenv("DATA_STORE_USER"),
        os.getenv("DATA_STORE_PASSWORD"),
    )


def pull_adapter(spec: str, dst: Path, data_store_base: str) -> Path:
    """spec format: namespace/name@revision (e.g., default/foo@cust-XXX)"""
    if "@" in spec:
        full_name, revision = spec.split("@", 1)
    else:
        full_name, revision = spec, "main"
    url = f"{data_store_base.rstrip('/')}/{full_name}.git"
    print(f"  cloning {full_name} @ {revision}")
    subprocess.run(["git", "clone", url, str(dst)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(dst), "checkout", revision], check=True, capture_output=True)
    return dst


def validate_adapter_dirs(adapter_dirs: list[Path]) -> list[Path]:
    if len(adapter_dirs) < 2:
        raise ValueError("TIES merge requires at least two source adapters")
    for adapter_dir in adapter_dirs:
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
        if not (adapter_dir / "adapter_model.safetensors").exists():
            raise FileNotFoundError(
                f"Adapter directory is missing adapter_model.safetensors: {adapter_dir}"
            )
    return adapter_dirs


def load_lora_tensors(path: Path) -> dict:
    sft = path / "adapter_model.safetensors"
    return safetensors.torch.load_file(str(sft))


def ties_merge(adapter_dirs: list[Path], out_dir: Path, trim_ratio: float) -> None:
    # Load all
    all_tensors = [load_lora_tensors(d) for d in adapter_dirs]
    n = len(all_tensors)
    print(f"Loaded {n} adapter tensor dicts")

    # Verify identical keys
    key_sets = [set(t.keys()) for t in all_tensors]
    if not all(k == key_sets[0] for k in key_sets):
        from functools import reduce
        diff = reduce(lambda a, b: a ^ b, key_sets)
        raise ValueError(
            f"Adapters have mismatched keys; symmetric difference (first 5): {list(diff)[:5]}"
        )

    keys = sorted(key_sets[0])
    print(f"Merging {len(keys)} tensors with trim_ratio={trim_ratio}")

    merged = {}
    for i, k in enumerate(keys):
        # Stack across adapters
        tensors = [t[k] for t in all_tensors]
        # Verify shapes match
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) > 1:
            raise ValueError(f"Tensor {k}: mismatched shapes {shapes}")
        # cast to float32 for the merge math, then back to original dtype
        orig_dtype = tensors[0].dtype
        stacked = torch.stack([t.to(torch.float32) for t in tensors], dim=0)
        merged_t = ties_merge_one_tensor(stacked, trim_ratio).to(orig_dtype)
        merged[k] = merged_t
        if (i + 1) % 200 == 0 or i + 1 == len(keys):
            print(f"  merged {i+1}/{len(keys)}")

    # Write out
    out_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(merged, str(out_dir / "adapter_model.safetensors"))
    print(
        f"Wrote {out_dir / 'adapter_model.safetensors'} "
        f"({sum(v.numel() for v in merged.values()):,} params)"
    )

    # Copy aux files from the first adapter.
    # adapter_config.json is REQUIRED: NIM_PEFT_SOURCE cannot load the merged
    # adapter without it - raise hard rather than producing a silently broken dir.
    REQUIRED_AUX = ["adapter_config.json"]
    OPTIONAL_AUX = [
        "automodel_peft_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    for f in REQUIRED_AUX:
        src = adapter_dirs[0] / f
        if not src.exists():
            raise FileNotFoundError(
                f"Required aux file {f!r} missing from first source adapter "
                f"({adapter_dirs[0]}); a merged adapter without {f} cannot be "
                f"loaded by NIM_PEFT_SOURCE."
            )
        shutil.copy2(src, out_dir / f)
        print(f"  copied {f}")
    for f in OPTIONAL_AUX:
        src = adapter_dirs[0] / f
        if src.exists():
            shutil.copy2(src, out_dir / f)
            print(f"  copied {f}")
        else:
            print(f"  (optional aux {f} not present, skipping)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="TIES-merge N LoRA adapters into a single merged adapter."
    )
    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--adapters",
        nargs="+",
        help="Adapter specs to clone from NeMo Data Store: namespace/name@revision",
    )
    source_group.add_argument(
        "--adapter-dirs",
        nargs="+",
        type=Path,
        help="Mounted source adapter directories containing adapter_model.safetensors",
    )
    ap.add_argument(
        "--data-store-git-base",
        help=(
            "Base Git URL for NeMo Data Store clone mode. Defaults to "
            "DATA_STORE_GIT_BASE or http://nemo-data-store:3000. Optional "
            "DATA_STORE_USER/DATA_STORE_PASSWORD env vars are injected as basic auth."
        ),
    )
    ap.add_argument("--out", required=True, help="Output directory for merged adapter")
    ap.add_argument(
        "--trim-ratio",
        type=float,
        default=0.2,
        help="Fraction of |value|-smallest entries to zero per adapter (default 0.2)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    if args.adapter_dirs:
        ties_merge(validate_adapter_dirs(args.adapter_dirs), out, args.trim_ratio)
    else:
        base_url = data_store_git_base(args.data_store_git_base)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dirs = []
            for i, spec in enumerate(args.adapters):
                d = td / f"a{i}"
                pull_adapter(spec, d, base_url)
                dirs.append(d)
            ties_merge(validate_adapter_dirs(dirs), out, args.trim_ratio)
    print(f"\nMerged adapter ready at: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
