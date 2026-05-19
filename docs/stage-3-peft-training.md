# Stage 3: PEFT Adapter Training

> **Status**: outline only. The pipeline is under active development; this
> document captures the intended approach. Stage 2 (dataset creation) is
> the input.

## What this stage produces

A LoRA (Low-Rank Adaptation) adapter trained on the Stage 2 dataset.
Loaded onto a base model at inference time, the adapter biases responses
toward the source vendor's product domain — without retraining the base.

## Why LoRA / PEFT

For a single-product expert adapter, LoRA is the right tool:

- **Small artifact.** Tens to hundreds of MB vs. full-model checkpoints
  measured in GB. Easy to swap, version, and ship alongside the base.
- **Composable.** Multiple per-product adapters can live alongside one base
  model; switch adapters per request based on the user's query intent.
- **Preserves base capabilities.** Full SFT can degrade generic
  capabilities; LoRA's narrow update path makes domain specialization
  without catastrophic forgetting more achievable.

The trade-off: LoRA learns less per training pass than full SFT, so signal
quality (Stage 2) matters more than for full fine-tuning.

## Two adapters, one base

This project trains **two adapters** from the same base model:

| Adapter | Trained on | Use case |
|---|---|---|
| `nim_expert` | Stage 2 dataset derived from `nim_curated` collection | Questions about NIM deployment, API shapes, profiles |
| `nemo_usvcs_expert` | Stage 2 dataset derived from `nemo_usvcs_curated` | Questions about NeMo Microservices platform operations |

Routing between them happens at the application layer (or via a small
classifier upstream). The base model handles questions outside either
domain unchanged.

## Planned training workflow

```
Stage 2 dataset (JSONL)
        │
        │  load + tokenize
        ▼
LoRA config
   - rank r (typically 8-32)
   - alpha (typically 2x rank)
   - target modules (q_proj, v_proj, etc.)
        │
        ▼
SFT trainer
   - base model frozen
   - only LoRA adapter weights trained
   - mixed-precision (bf16 or fp8 depending on hardware)
        │
        ▼
Checkpoint (LoRA adapter weights)
        │
        ▼
Evaluation
   - vs. base model (no adapter)
   - vs. RAG-only baseline
   - product-specific QA accuracy
        │
        ▼
Adapter artifact (publishable)
```

## Key choices, not yet final

- **Base model.** Options: Llama 3.x 8B / 70B, NeMotron-3-Nano-30B-A3B,
  others. The choice affects training compute, adapter size, and inference
  cost. Likely a small dense model first (8B-class) for fast iteration,
  with a larger MoE option later.
- **LoRA rank.** Common starting point is rank=16, alpha=32. Will tune
  based on validation metrics.
- **Training framework.** NVIDIA NeMo Customizer (part of NeMo Microservices)
  is the planned target — keeps the whole pipeline on NVIDIA-native tools
  and makes adapter deployment via NIM straightforward.
- **Evaluation strategy.** External frontier judge for response quality;
  domain-specific QA benchmark for factual accuracy. Eval split held out
  from the source collection so it represents the same distribution.

## Hardware notes

- Dense Llama 3.x training requires CUDA-capable GPUs with sufficient
  memory for the base model + LoRA gradients (16-24 GB for an 8B model
  with rank=16 in bf16).
- MoE models (Nemotron, gpt-oss-20b) have specific constraints — sequence
  packing is not supported, and certain optimizers don't work cleanly
  with the routing layer. Plan accordingly.
- Blackwell (sm_120) GPUs have some current incompatibilities with
  sequence-packed flash-attention. If your training framework offers
  sequence packing as an optimization, disable it on Blackwell hardware.

## Adapter deployment

Once trained, adapters are loaded into a serving NIM via `NIM_PEFT_SOURCE`.
On recent NIM releases (2.0.3+), the value must be a filesystem path; URL
mode is parsed but silently ignored. Practical deployment:

- Filesystem mount the adapter directory into the NIM pod (NFS, S3 CSI,
  local PV, etc.).
- Set `NIM_PEFT_SOURCE` to that mount path.
- Verify the adapter is detected via the NIM's `/v1/models` endpoint —
  the adapter appears as a selectable model variant.

## Open work

- [ ] Lock base model choice based on size/cost/perf tradeoff.
- [ ] Build the training pipeline against NeMo Customizer.
- [ ] Define eval benchmark per adapter.
- [ ] Document the adapter-publication workflow (HF Hub format? NGC?).

When this stage lands, this doc will be replaced by a worked walkthrough
end-to-end: from JSONL dataset through adapter weights to a deployed NIM
serving the adapter.
