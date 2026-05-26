"""TIES merge of N LoRA adapters trained via NeMo Customizer parallel-shard SFT.

For each LoRA tensor (lora_A or lora_B matrix) across the N adapters:
  1. Trim:  keep only the top (1 - trim_ratio) magnitude entries; zero the rest
  2. Sign-elect: per-parameter, pick the sign with the largest total magnitude across adapters
  3. Disjoint merge: average ONLY the entries that agree with the elected sign
Reference: Yadav et al. 2023, "TIES-Merging" (arXiv 2306.01708)

Usage:
  python3 scripts/stage3/ties_merge.py \\
    --adapters \\
        default/lora-nim-nemotron-nano-30b-r16-shard-a@cust-AAAA \\
        default/lora-nim-nemotron-nano-30b-r16-shard-b@cust-BBBB \\
    --out  /mnt/nvme2/peft/checkpoints/lora/nemotron-nano-30b-stage3-tiesmerged \\
    --trim-ratio 0.2

Pulls each adapter via git clone from the data-store (default/<name>@<revision>),
loads adapter_model.safetensors, runs TIES, writes merged adapter at --out
(with all auxiliary files copied from the first adapter so it loads identically
in a NIM via NIM_PEFT_SOURCE).

Ported from nim-sft-final/scripts/ties_merge.py — adapted for:
  - NodePort endpoint (192.168.1.187:30912 instead of localhost:30912)
  - stage3 context (adapter naming, doc strings)
  - Algorithm and math are preserved verbatim from the validated original.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import safetensors.torch
import torch


GITEA_USER = "datastore_admin"
GITEA_PASS = "nemo-peft-ds"
DATA_STORE_BASE = f"http://{GITEA_USER}:{GITEA_PASS}@192.168.1.187:30912"


# ---------------------------------------------------------------------------
# Pure function — tested in test_stage3_ties_merge.py
# ---------------------------------------------------------------------------

def ties_merge_one_tensor(stacked: torch.Tensor, trim_ratio: float) -> torch.Tensor:
    """stacked shape: (N, ...); returns merged tensor of shape (...)

    Algorithm:
      1. Trim — per-adapter, zero everything below (1 - trim_ratio) quantile of |value|
      2. Sign-elect — sign with largest summed magnitude across adapters wins per parameter
      3. Disjoint merge — average only entries whose sign agrees with the elected sign
    """
    n = stacked.shape[0]

    # 1. Trim — per-adapter, zero everything below (1 - trim_ratio) quantile of |value|
    flat = stacked.reshape(n, -1)
    if trim_ratio > 0:
        k = int(round((1 - trim_ratio) * flat.shape[1]))   # keep top k entries per adapter
        k = max(1, k)
        # threshold = the k-th largest |value| per adapter
        absvals = flat.abs()
        thresh, _ = torch.kthvalue(absvals, flat.shape[1] - k + 1, dim=1, keepdim=True)
        mask = absvals >= thresh
        flat = flat * mask

    # 2. Sign elect — sign with largest summed magnitude across adapters wins per parameter
    sign_score = flat.sum(dim=0)        # signed sum; sign is the consensus sign
    elected_sign = torch.sign(sign_score)
    # if score is exactly 0, fall back to majority-sign (or just 0)

    # 3. Disjoint merge — average only entries whose sign agrees with the elected sign
    elected_sign_b = elected_sign.unsqueeze(0).expand_as(flat)
    agree_mask = (torch.sign(flat) == elected_sign_b) & (flat != 0)
    sums = (flat * agree_mask).sum(dim=0)
    counts = agree_mask.sum(dim=0).clamp(min=1)            # avoid divide-by-zero
    merged = sums / counts

    return merged.reshape(stacked.shape[1:])


# ---------------------------------------------------------------------------
# Integration helpers (not unit-tested — covered by operational verification)
# ---------------------------------------------------------------------------

def pull_adapter(spec: str, dst: Path) -> Path:
    """spec format: namespace/name@revision (e.g., default/foo@cust-XXX)"""
    if "@" in spec:
        full_name, revision = spec.split("@", 1)
    else:
        full_name, revision = spec, "main"
    url = f"{DATA_STORE_BASE}/{full_name}.git"
    print(f"  cloning {full_name} @ {revision}")
    subprocess.run(["git", "clone", url, str(dst)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(dst), "checkout", revision], check=True, capture_output=True)
    return dst


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
        diff = key_sets[0].symmetric_difference(*key_sets[1:])
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

    # Copy aux files (adapter_config, tokenizer, chat_template) from the first adapter
    aux_files = [
        "adapter_config.json",
        "automodel_peft_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    for f in aux_files:
        src = adapter_dirs[0] / f
        if src.exists():
            shutil.copy2(src, out_dir / f)
            print(f"  copied {f}")
        else:
            print(f"  WARN: {f} not present in first adapter")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="TIES-merge N LoRA adapters from data-store into a single merged adapter."
    )
    ap.add_argument(
        "--adapters",
        nargs="+",
        required=True,
        help="Adapter specs: namespace/name@revision",
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
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dirs = []
        for i, spec in enumerate(args.adapters):
            d = td / f"a{i}"
            pull_adapter(spec, d)
            dirs.append(d)
        ties_merge(dirs, out, args.trim_ratio)
    print(f"\nMerged adapter ready at: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
