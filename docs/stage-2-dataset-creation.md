# Stage 2: Dataset Creation

> **Status**: outline only. The pipeline is under active development; this
> document captures the intended approach. Stage 1 (curated crawl) is the
> input. Stage 3 (PEFT training) is the consumer.

## What this stage produces

A supervised fine-tuning dataset — `(instruction, response)` pairs, in
whatever format your training framework expects (JSONL, Parquet, etc.) —
derived from the curated documentation corpus in Stage 1.

## Two-source design

Training data comes from two complementary sources:

1. **Ground-truth (logical entailment) extraction** from corpus chunks.
   Reads each chunk and produces claims it definitively supports. These
   become the spine of the dataset — every pair is grounded in vendor docs.
2. **Synthetic augmentation** via [NVIDIA NeMo Data Designer](https://docs.nvidia.com/nemo/microservices/latest/about/index.html).
   Generates variations, edge cases, and balancing examples on top of the
   ground-truth seed. Addresses corpus skew (see Stage 1's NIM corpus-bias
   discussion) and adds adversarial coverage.

## Why entailment specifically

For a single-product SFT adapter, the training signal needs to be
**factually grounded in the source docs** — not paraphrased, not
generalized, not inferred from training data the base model already saw.
Entailment-style extraction enforces this: a pair is only included if the
response is provably entailed by a chunk in the corpus.

Alternative approaches (instruction-tuning from generic Q&A datasets,
self-instruct over the base model) tend to leak base-model priors into the
adapter, which defeats the purpose of training a product expert.

## Planned workflow

```
Stage 1 collection (ES)
        │
        │  retrieve chunks
        ▼
Chunk filter / quality gate
   - drop boilerplate, _static, API stubs
   - drop chunks < ~500 chars main text
        │
        ▼
Entailment extraction
   - LLM-as-judge produces (claim, source-chunk) pairs
   - reject pairs where claim is not entailed
        │
        ▼
Seed dataset (small, grounded)
        │
        │  feed to Data Designer
        ▼
Synthetic augmentation
   - paraphrases, multi-turn variations
   - adversarial / edge-case examples
   - per-product stratified rebalancing (Stage 1 skew correction)
        │
        ▼
Final SFT dataset (JSONL)
```

## Key choices, not yet final

The following are decision points that will be locked down when the
pipeline lands:

- **Judge model for entailment.** Likely an external frontier model
  accessed via API, not an in-cluster NIM (independence from the system
  under evaluation matters).
- **Chunk size / overlap.** Driven by what the retrieval-augmented
  variant of the adapter will eventually see at inference time.
- **Augmentation ratio.** Open question whether 1:1 seed:augmented or
  1:5 is the right balance for SFT signal-to-noise.
- **Stratification.** For corpora with skew (NIM example: 88%
  LLM-NIM), per-product caps before augmentation prevent the adapter
  from over-fitting to the dominant product.

## Related references

- The Stage 1 ["filter at extraction, not at crawl"](stage-1-curated-crawl.md#filter-at-extraction-not-at-crawl)
  guidance — those filters apply here.
- The Stage 1 ["corpus bias to know about"](../examples/nim.md#corpus-bias-to-know-about)
  callout for NIM — relevant for the stratification decision above.

## Open work

- [ ] Choose and benchmark the entailment-judge model.
- [ ] Build the chunk-quality-gate filters (Stage 1 lists these informally;
      need a concrete script).
- [ ] Wire up NeMo Data Designer with seed → augmentation prompts.
- [ ] Validate output dataset: spot-check 100 random pairs for factual
      grounding.

When this stage lands, this doc will replace the outline with a worked
walkthrough on the same two case studies (NIM + NeMo Microservices).
