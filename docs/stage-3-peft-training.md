# Stage 3: PEFT Adapter Training

> **Status**: implemented reference methodology. Stage 2 finalized datasets are
> the input; `scripts/stage3/` and `scripts/eval/` provide the Customizer,
> completion-capture, and evaluation surfaces. `docs/integrations/` documents
> deployable MLflow export/observability support plus WIP orchestration notes.

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
collection. The retained examples use two representative public documentation
corpora, but the pattern is general: each collection can represent a
department, workflow, compliance area, engineering system, field-service
procedure set, or other enterprise sub-domain.

| Adapter pattern | Trained on | Use case |
|---|---|---|
| `domain_a_expert` | Stage 2 dataset derived from one scoped collection | Questions requiring that collection's terminology, workflows, and constraints |
| `domain_b_expert` | Stage 2 dataset derived from a second scoped collection | Questions requiring a different process, policy, or technical vocabulary |

Routing between adapters happens at the application layer (or via a small
classifier upstream). The base model handles questions outside the adapted
domains unchanged.

## Training and evaluation workflow

```
Stage 2 training.jsonl + validation.jsonl
        │
        │  register dataset entity / Data Store repo
        ▼
NeMo Customizer LoRA SFT
   - dense base model selected for the deployment target
   - rank 16 and rank 32 comparison runs
   - alpha typically 2x rank
   - base model frozen; only adapter weights trained
        │
        ▼
Adapter artifact / model entity
        │
        ├── no-RAG completion capture on immutable golden QA
        ├── optional RAG completion capture on the reduced comparison set
        └── MLflow-ready metadata and evaluation export for publishable lineage
        │
        ▼
Evaluation
   - single-axis scoring for standalone model efficacy
   - pairwise scoring for LE vs. Curator and best LoRA vs. 70B reference
   - optional RAGAS diagnostics for retrieval behavior
```

## Reference choices

- **Base models.** The case study uses dense Llama 1B, 3B, and 8B models for
  LoRA training, then compares the strongest LoRA candidates with a dense
  Llama 3.3 70B no-adapter reference target.
- **LoRA ranks.** Rank 16 and rank 32 are trained for each dataset/model pair
  so the evaluation can show whether additional adapter capacity is useful.
- **Training framework.** NVIDIA NeMo Customizer consumes the Stage 2 SFT JSONL
  and produces LoRA adapter artifacts that can be served through LoRA-capable
  NIM deployments.
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

## Adaptation checklist

For a new domain corpus:

- Choose the dense base model sizes that match the latency, cost, and quality
  target for the deployment.
- Train at least one LoRA rank baseline, and use rank 16/rank 32 comparisons
  when adapter capacity is part of the experiment.
- Keep no-RAG evaluation separate from optional RAG evaluation so model
  adaptation and retrieval quality are measured independently.
- Capture dataset, adapter job IDs, evaluation outputs, and MLflow-ready
  metadata together so the selected adapter can be reproduced or rolled back.
