# 8B Non-Augmented 5-Epoch LoRA Training

Date: 2026-06-09

Purpose: test whether the grounded, non-augmented datasets show value from a longer 5-epoch LoRA SFT run on Llama 3.1 8B before spending more time on augmented 8B variants.

## Scope

- Base model: `meta/llama-3.1-8b-instruct`
- Customizer template: `meta/llama-3.1-8b-instruct@v1.0.0+80GB`
- Precision: `bf16-mixed`
- Datasets:
  - `default/stage3-nim-curated`
  - `default/stage3-nemo-usvcs-curated`
- Ranks:
  - `r16`
  - `r32`
- Epochs: `5`
- Batch size: `16`
- Learning rate: `1e-4`
- Sequence packing: disabled
- Required placement: Blackwell node `<BLACKWELL_NODE>`

## Scheduling Correction

Initial 2026-06-09 submissions landed on `<ADA_NODE>` because the live Customizer ConfigMap still had:

- `training.container_defaults.nodeSelector.kubernetes.io/hostname: <ADA_NODE>`
- `training.nodeSelectors.kubernetes.io/hostname: <ADA_NODE>`

Those jobs were cancelled or failed. The ConfigMap was patched to `<BLACKWELL_NODE>`, Customizer was restarted, and placement was verified with a two-job canary before queueing r32.

Reusable command:

```bash
python scripts/ops/patch_customizer_training_node.py --node <BLACKWELL_NODE> --restart
```

## Active Jobs

| Dataset | Rank | Job ID | Output model | Status |
|---|---:|---|---|---|
| NIM grounded | 16 | `cust-GQkpTXu3frr2PnWSnbcgKH` | `default/lora-nim-e5-llama-3.1-8b-r16` | running on `<BLACKWELL_NODE>` |
| NeMo Microservices grounded | 16 | `cust-EDmfZ5Hi9HaCw4Ko4wyLpH` | `default/lora-nemo-usvcs-e5-llama-3.1-8b-r16` | running on `<BLACKWELL_NODE>` |
| NIM grounded | 32 | `cust-CiTm8oforcC38Rw67D3RVH` | `default/lora-nim-e5-llama-3.1-8b-r32` | queued for `<BLACKWELL_NODE>` |
| NeMo Microservices grounded | 32 | `cust-XZMopAuFBH57vNpVJZzBvj` | `default/lora-nemo-usvcs-e5-llama-3.1-8b-r32` | queued for `<BLACKWELL_NODE>` |

## Follow-On

After training completes:

1. Sync the four adapters into the 8B LoRA model-store directory.
2. Restart the 8B LoRA NIM deployment with `--max-loras 4`.
3. Collect completions for both non-augmented test sets at `max_tokens=8192`.
4. Run Phase 1, Phase 2, and Phase 3 eval with Claude Sonnet and Nemotron 3 Ultra.
