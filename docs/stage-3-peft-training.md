# Stage 3: PEFT Adapter Training

> **Status**: outline only. The pipeline is under active development; this
> document captures the intended approach. Stage 2 (dataset creation) is
> the input.

## What this stage produces

A LoRA (Low-Rank Adaptation) adapter trained on the Stage 2 dataset.
Loaded onto a base model at inference time, the adapter biases responses
toward a target enterprise domain or sub-domain without retraining the base.
The adapter is meant to improve how the model handles domain nomenclature,
procedures, abbreviations, constraints, and expected answer style.

## Why LoRA / PEFT

For a domain-specific adapter, LoRA is the right tool:

- **Small artifact.** Tens to hundreds of MB vs. full-model checkpoints
  measured in GB. Easy to swap, version, and ship alongside the base.
- **Composable.** Multiple domain or sub-domain adapters can live alongside
  one base model; switch adapters per request based on the user's query
  intent, tenant, workflow, or collection boundary.
- **Preserves base capabilities.** Full SFT can degrade generic
  capabilities; LoRA's narrow update path makes domain specialization
  without catastrophic forgetting more achievable.

The trade-off: LoRA learns less per training pass than full SFT, so signal
quality (Stage 2) matters more than for full fine-tuning.

## Model Downselection Rationale

The case study evaluates LoRA adapters across multiple dense model sizes rather
than assuming one model class is correct. Training 1B, 3B, and 8B dense adapters
with matched ranks and datasets shows how much domain specificity each size can
absorb, where the smaller models saturate, and whether a larger adapter earns
its serving cost.

The dense Llama 3.3 70B target is used as a reference comparison target, not as
the judge. Single-axis scoring first measures standalone answer quality for each
LoRA candidate. Pairwise scoring then compares the strongest LoRA candidates in
each size class against the 70B reference. This supports practical selection of
the smallest dense model that meets the quality bar for the target domain,
latency budget, and deployment footprint.

## Domain adapters, one base

The case study trains two adapters from the same base model, one per scoped
collection. These adapters happen to use NIM and NeMo Microservices
documentation, but the pattern is general: each collection can represent a
department, workflow, compliance area, engineering system, field-service
procedure set, or other enterprise sub-domain.

| Adapter pattern | Trained on | Use case |
|---|---|---|
| `domain_a_expert` | Stage 2 dataset derived from one scoped collection | Questions requiring that collection's terminology, workflows, and constraints |
| `domain_b_expert` | Stage 2 dataset derived from a second scoped collection | Questions requiring a different process, policy, or technical vocabulary |

Routing between adapters happens at the application layer (or via a small
classifier upstream). The base model handles questions outside the adapted
domains unchanged.

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
   - domain-specific QA accuracy
        │
        ▼
Adapter artifact (publishable)
```

## Key choices, not yet final

- **Base model.** The case study uses dense Llama 1B, 3B, and 8B models for
  LoRA training, then compares the strongest LoRA candidates with a dense
  Llama 3.3 70B no-adapter reference target.
- **LoRA rank.** Common starting point is rank=16, alpha=32. Will tune
  based on validation metrics.
- **Training framework.** NVIDIA NeMo Customizer (part of NeMo Microservices)
  is the planned target — keeps the whole pipeline on NVIDIA-native tools
  and makes adapter deployment via NIM straightforward.
- **Evaluation strategy.** External frontier judge for response quality;
  domain-specific golden QA benchmark for factual accuracy; single-axis
  scoring for standalone efficacy; pairwise scoring for LE-vs-Curator and
  best-LoRA-vs-70B comparisons. Optional RAG/RAGAS diagnostics measure
  retrieval-augmented behavior separately and should not decide the primary
  no-RAG LoRA winner.

## Hardware notes

- Dense Llama 3.x training requires CUDA-capable GPUs with sufficient
  memory for the base model + LoRA gradients (16-24 GB for an 8B model
  with rank=16 in bf16).
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
