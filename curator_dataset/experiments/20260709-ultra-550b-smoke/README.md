# Nemotron 3 Ultra 550B Minimal Smoke Test Design

Date: 2026-07-09

Purpose: test whether generating training data with Nemotron 3 Ultra 550B is likely to improve downstream LoRA validation behavior compared with the current Nemotron 3 Super 120B-equivalent generation, without paying for a full-corpus regeneration first.

## Hypothesis

If Ultra 550B improves grounded question-answer quality, then a source-fixed smoke slice should show at least one of these signals before full scale-out:

- Lower validation-best loss when trained with the same base model, rank, epochs, and Customizer settings.
- Better independent judge scores for groundedness, answer correctness, and non-hallucination.
- Fewer invalid, duplicated, trivial, or context-leaking samples after normalization.

The current baseline is Super 120B-equivalent generation:

- LE pipeline: `nvidia/nemotron-3-super-120b-a12b` through `nim-llm-super-120b-bw`.
- Curator pipeline: actual manifests record `nvidia/nvidia/nemotron-3-super-v3`; the catalog shorthand `nvidia/nemotron-3-super-v3` is treated as exact-equivalent to `nvidia/nemotron-3-super-120b-a12b` per the 2026-07-09 project clarification.

## Minimal matrix

Use a fixed source slice and regenerate only that slice.

| Dimension | Values |
| --- | --- |
| Corpora | NIM, NeMo Microservices |
| Pipelines | Logical entailment, NeMo Curator DiverseQA |
| Generator models | Super 120B-equivalent baseline, Ultra 550B candidate |
| Training proxy | Llama 3.1 8B LoRA r16 first; add r32 only if r16 is promising |
| Judge | Claude Sonnet 4.6 via `azure/anthropic/claude-sonnet-4-6` on `inference.nvidia.com` |

Start with r16 because the current Curator r16/r32 spread is small relative to the LE gap. A r32 follow-up is useful only if Ultra changes the data-quality signal enough to justify the extra training edge.

## Source slice

Freeze a deterministic, source-balanced slice before generation:

- NIM: 40 to 60 source documents, stratified across product families and document types already present in the Curator corpus.
- NeMo Microservices: 40 to 60 source documents, stratified across customization, datastore, evaluation, deployment, and troubleshooting topics.
- Keep source document IDs, URLs, text hashes, chunk IDs, and selection seed in a slice manifest.
- Use the same source slice for LE and Curator. Do not let Ultra see a different document mix than Super.

Target output size per cell: about 500 training rows and 100 validation rows after normalization. That is large enough to expose format and grounding problems but small enough to run quickly.

### Frozen slice edge

The 2026-07-09 smoke slice is now frozen in `slice_manifest.json` using seed `20260709-ultra-550b-smoke-v1`. It selects 48 documents per corpus, balanced as 12 short, 12 medium, 12 long, and 12 very-long documents. Exact duplicate document text is excluded before selection so repeated generated docs do not overweight the smoke signal.

Frozen inputs:

- Curator NIM: `inputs/nim_curated.curator_input.jsonl`
- LE NIM: `inputs/nim_curated.passages.jsonl`
- Curator NeMo Microservices: `inputs/nemo_usvcs_curated.curator_input.jsonl`
- LE NeMo Microservices: `inputs/nemo_usvcs_curated.passages.jsonl`

The Super Curator model ID remains `nvidia/nvidia/nemotron-3-super-v3`. The Ultra generator model ID used by existing repo evaluation jobs is `nvidia/nvidia/nemotron-3-ultra`; verify endpoint availability before launch and record any endpoint-resolved alias in the generation manifest.

### Micro execution slice

The first Curator Super run on the full 48-document NIM slice was stopped after roughly 17 minutes with one active `DiverseQAStage` task, zero completed DiverseQA tasks, and no raw JSONL output. That attempt is preserved under `runs/curator/nim_curated/super-v3-attempt2.*`.

For the initial endpoint and quality hypothesis check, use `micro_slice_manifest.json` and `inputs_micro/`. This derived slice keeps the same seed and source-selection policy but uses 8 documents per corpus: 2 short, 2 medium, 2 long, and 2 very-long documents. Use this micro slice for the first Super-vs-Ultra generation comparison before spending time on the larger frozen slice.


## Generation controls

- Hold prompts, temperature, max tokens, context packing, and filtering constant except for the generator model.
- Keep the existing Data Store to HF compatibility shim in the registration path; Customizer should only see HF-style dataset entities.
- Preserve sample lineage for every row: source document ID, source URL, chunk IDs, generator model, pipeline stage, and normalization version.
- Persist rejected rows with rejection reasons, not just accepted rows.

## Evaluation signals

### Training signal

Train proxy adapters sequentially through Customizer:

1. LE Super r16.
2. LE Ultra r16.
3. Curator Super r16.
4. Curator Ultra r16.

Run this once per corpus. Use the same base model, batch size, learning rate, epoch count, validation cadence, and placement. Capture best validation loss, first-validation loss, final validation loss, train loss, steps, runtime, output revision, and worker-log evidence.

### Judge signal

Use Claude Sonnet 4.6 through the NVIDIA endpoint as the independent judge. Credentials should come only from the existing Kubernetes secret; do not write secrets into repo files.

Judge a paired sample from each cell on:

- Groundedness: answer supported by provided context.
- Correctness: answer resolves the question accurately.
- Specificity: answer is neither vague nor overbroad.
- Leakage: answer does not depend on hidden context, generation artifacts, or source metadata not presented to the model.
- Format validity: prompt/completion follows the expected SFT schema.

Use paired comparisons where possible: Super-generated row vs Ultra-generated row for the same source segment and pipeline.

### Dataset quality signal

Compute lightweight deterministic checks:

- Exact and near-duplicate rate.
- Empty, too-short, and too-long answer rate.
- Context-copy ratio.
- Question triviality rate using lexical heuristics.
- Validation/train source overlap.
- Invalid JSONL or missing required fields.

## Success gates

Proceed to a larger generation only if Ultra clears these gates on at least one corpus/pipeline pair:

- Validation-best loss improves by at least 5% versus the source-fixed Super baseline, or improves judge composite by at least 0.15 on a 1-5 scale with no validation-loss regression.
- Groundedness and correctness do not regress by more than 2 percentage points.
- Invalid/rejected row rate is not higher than Super by more than 2 percentage points.
- The gain is visible at the first validation checkpoint, not only after overfitting later epochs.
- Runtime and endpoint cost are acceptable for full-corpus scale-out.

If Ultra improves Curator quality but still trails LE materially, keep LE as the production dataset path and use Ultra only for targeted gap-fill or adversarial examples.

## Action plan

1. Create a source-slice manifest for NIM and NeMo Microservices with hashes and selection seed.
2. Generate the Super baseline on the exact slice if no slice-equivalent baseline already exists.
3. Generate the Ultra candidate on the same slice for LE and Curator.
4. Normalize all outputs through the same JSONL schema and register them through the HF shim-backed dataset path.
5. Train r16 proxy adapters sequentially on Blackwell, committing after each training edge.
6. Collect worker-log validation checkpoints and Customizer status into per-job manifests.
7. Run Claude Sonnet 4.6 judge evaluation through `inference.nvidia.com` using the Kubernetes secret for credentials.
8. Produce a smoke comparison report with validation curves, judge deltas, row-quality checks, and a go/no-go recommendation.
9. Commit the smoke report and manifests before deciding on full-corpus Ultra generation.

## Expected directory layout

```text
curator_dataset/experiments/20260709-ultra-550b-smoke/
  README.md
  slice_manifest.json
  generation_manifest.json
  customizer_runs/
  judge_runs/
  smoke_report.md
  smoke_report.svg
```

Only `README.md` exists now; the remaining files should be created by the smoke-test run.
