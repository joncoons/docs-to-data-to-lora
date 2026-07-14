# NIM 1B Data Designer Augmentation Experiment

## Objective

Test whether a small, lineage-preserving synthetic augmentation set from NeMo
Data Designer improves the 1B LoRA adapter trained on the grounded NIM dataset.
The grounded logical-entailment dataset remains the source of truth; synthetic
rows are an additive ablation, not a replacement.

Target collection: `nim_curated`.

Base model: `meta/llama-3.2-1b-instruct`.

Baseline adapters already trained and evaluated:

- `lora-nim-llama-3.2-1b-r16`
- `lora-nim-llama-3.2-1b-r32`

New experimental adapters:

- `lora-nim-dd-kimi-llama-3.2-1b-r16`
- `lora-nim-dd-kimi-llama-3.2-1b-r32`

The `nim-dd-kimi` corpus slug keeps the adapter names compatible with the
existing proxy/collector parser while making the augmentation source visible.

## Non-Negotiables

- Do not modify the existing `nim_curated` dataset in place.
- Do not use validation or test rows as Kimi seed input.
- Keep the existing validation and test sets unchanged for comparison.
- Preserve per-row lineage from synthetic row back to seed rows, Kimi seed
  generation, Data Designer job, and source grounded samples.
- Register the augmented dataset as a new NeMo Data Store / Entity Store entity.
- Train new adapters under new output model entity names; do not overwrite the
  baseline adapters.

## Current Inputs

Grounded dataset directory:

```text
<DATASET_ROOT>/nim_curated/
  training.jsonl
  validation.jsonl
  test_set.jsonl
  test_set_with_context.jsonl
  adapter_train.jsonl
  adapter_val.jsonl
  stage2_eval.jsonl
  validation_report.json
  bias_report.json
```

Existing Data Designer gap-fill scaffolding expects:

```text
data_designer/gapfill_requests.jsonl
provenance/gap_manifest.json
```

Those files are not present for `nim_curated`, so this experiment should use a
new seed-from-grounded preparation path rather than forcing the gap-fill path.

## Experiment Dataset Layout

Create a new immutable experiment directory:

```text
<DATASET_ROOT>/experiments/nim_curated_dd_kimi_1b_<date>/
  source_snapshot/
  data_designer/
    kimi_seed_requests.jsonl
    seed_dataset.csv
    submission_plan.json
    generated_raw.jsonl
    generated_samples.jsonl
    result_manifest.json
  provenance/
    synthetic_samples.jsonl
    merge_manifest.json
  training.jsonl
  validation.jsonl
  test_set.jsonl
  test_set_with_context.jsonl
  manifests/dataset_version_manifest.json
```

`source_snapshot/` records checksums and relative paths for the source grounded
files. It should not duplicate multi-GB files unless needed for archival.

## Phase 1: Build Kimi Seed Set

Use Kimi K2 to create seed records from grounded `training.jsonl` only.
Kimi should generate seed briefs, coverage intents, and variation instructions,
not final accepted training rows.

Kimi endpoint:

```text
https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/kimi-k2-6/v1
```

Key source:

```text
$KIMI_KEY
```

Recommended initial scope:

- sample 150-250 grounded training rows
- stratify by product family, source URL/domain when available, prompt length,
  answer length, and low-confidence themes from `validation_report.json`
- request 2-4 synthetic pairs per seed record
- use `max_tokens=8192` for Kimi K2 seed generation because reasoning output can
  consume a smaller cap before the final JSON is emitted
- strip inline `<think>...</think>` blocks from seed responses and generated
  synthetic rows before writing seed CSVs or training rows
- cap accepted synthetic rows to roughly 10-20% of grounded train rows for the
  first ablation

Seed output schema:

```json
{
  "seed_id": "seed-...",
  "source_dataset": "nim_curated",
  "source_split": "training",
  "source_row_indices": [123, 456],
  "source_sample_ids": ["..."],
  "source_prompt_excerpt": "...",
  "source_completion_excerpt": "...",
  "coverage_axis": "deployment|profiles|api|security|operations|troubleshooting|other",
  "augmentation_intent": "Generate grounded QA variants that test ...",
  "pairs_requested": 3,
  "constraints": [
    "answer must be supported by provided source text",
    "do not introduce product claims absent from source",
    "prefer operational questions over generic documentation summaries"
  ],
  "kimi": {
    "endpoint": "https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/kimi-k2-6/v1",
    "model": "kimi-k2-6",
    "prompt_sha256": "sha256:...",
    "response_sha256": "sha256:..."
  }
}
```

Planned implementation:

```bash
python scripts/pipeline/build_data_designer_seed_from_grounded.py \
  --dataset-dir <DATASET_ROOT>/nim_curated \
  --output-dir <DATASET_ROOT>/experiments/nim_curated_dd_kimi_1b_<date> \
  --collection nim_curated \
  --judge-api-url https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/kimi-k2-6/v1 \
  --judge-model kimi-k2-6 \
  --judge-api-key-env KIMI_KEY \
  --seed-count 200 \
  --pairs-per-seed 3 \
  --max-tokens 8192
```

## Phase 2: Submit NeMo Data Designer Job

Upload `data_designer/seed_dataset.csv` to NeMo Data Store and create a Data
Designer job using native NeMo Data Designer APIs/SDK. The existing
`data_designer_gapfill.py` submit logic can be reused after adding a mode that
accepts the seed-from-grounded schema.

Generation controls for the first run:

- `temperature`: `0.3`
- `top_p`: `1.0`
- `max_tokens`: `8192` when using a reasoning-capable generator; `2048` is
  acceptable only for non-reasoning models that consistently emit final JSON
- strip inline `<think>...</think>` blocks during result collection before
  schema parsing and admission filtering
- requested synthetic pairs: target 600-900 raw pairs, expecting filtering to
  accept fewer

Required artifacts:

```text
data_designer/submission_plan.json
data_designer/generated_raw.jsonl
data_designer/result_manifest.json
```

Every artifact must include the Data Designer job ID, seed dataset reference,
model/provider settings, and input/output checksums.

## Phase 3: Synthetic Admission And Provenance

Normalize generated rows into the same prompt/completion shape expected by
Customizer, then run admission filters before merging.

Minimum admission checks:

- JSON/schema validity
- non-empty question and answer
- exact and near-duplicate removal against grounded train/validation/test
- source-support check against the seed source text
- no test/validation leakage
- answer length and prompt length bounds

Recommended NVIDIA-native quality step:

- pass normalized synthetic rows through NeMo Curator when available, recording
  accepted/rejected samples and rejection reasons

Kimi may be used as an external groundedness judge for this experiment, but the
result must be recorded as an experiment dependency and not confused with a
native NeMo service.

Accepted synthetic sample schema should include:

```json
{
  "origin": "data_designer_synthetic",
  "prompt": "...",
  "completion": "...",
  "lineage": {
    "seed_id": "seed-...",
    "source_dataset": "nim_curated",
    "source_split": "training",
    "source_row_indices": [123],
    "source_sample_ids": ["..."],
    "data_designer_job_id": "...",
    "kimi_seed_prompt_sha256": "sha256:..."
  },
  "quality": {
    "accepted": true,
    "grounding_judge": "kimi-k2-6",
    "grounding_score": 0.0
  }
}
```

## Phase 4: Merge Augmented Dataset

Create the augmented training split:

```text
training.jsonl = grounded training.jsonl + accepted synthetic rows
validation.jsonl = original validation.jsonl unchanged
test_set*.jsonl = original test files unchanged
```

Write `provenance/merge_manifest.json` with:

- source grounded dataset checksums
- synthetic generated/accepted/rejected counts
- final train/validation/test row counts
- synthetic ratio
- Kimi seed run metadata
- Data Designer job metadata
- admission filter metrics

Finalize the dataset and register it as a new entity:

```text
default/stage3-nim-curated-dd-kimi-v1
```

The dataset registration artifact should preserve all Data Designer and Kimi
seed sidecars so MLflow can later log them as lineage artifacts.

## Phase 5: Train 1B Augmented Adapters

Use the same Customizer recipe as the baseline 1B runs, changing only the
registered dataset entity and output model entity.

Required small code change:

- extend `scripts/stage3/train_adapter.py` to accept `--dataset-entity` and an
  adapter/corpus override, instead of relying only on the hard-coded
  `default/stage3-nim-curated` mapping.

Planned submissions:

```bash
python scripts/stage3/train_adapter.py \
  --collection nim_curated \
  --base-model meta/llama-3.2-1b-instruct \
  --rank 16 \
  --dataset-entity default/stage3-nim-curated-dd-kimi-v1 \
  --adapter-name lora-nim-dd-kimi-llama-3.2-1b-r16 \
  --wait

python scripts/stage3/train_adapter.py \
  --collection nim_curated \
  --base-model meta/llama-3.2-1b-instruct \
  --rank 32 \
  --dataset-entity default/stage3-nim-curated-dd-kimi-v1 \
  --adapter-name lora-nim-dd-kimi-llama-3.2-1b-r32 \
  --wait
```

Customizer observability to capture:

- Customizer job ID
- dataset entity
- output model entity
- rank/alpha
- train loss and validation loss
- wall-clock duration
- final adapter artifact URI

## Phase 6: Serve And Evaluate

Add the two augmented adapters to the 1B LoRA-capable NIM PEFT source, then add
proxy routes for:

```text
lora-nim-dd-kimi-llama-3.2-1b-r16
lora-nim-dd-kimi-llama-3.2-1b-r32
```

Collect completions using the same held-out test prompts and generation budget:

```text
--max-tokens 8192
```

Evaluation comparisons:

- augmented r16 vs baseline r16 on `nim_curated` held-out test
- augmented r32 vs baseline r32 on `nim_curated` held-out test
- augmented r16/r32 vs base 1B on `nim_curated` held-out test
- optional: augmented adapters on `nemo_usvcs_curated` test to detect harmful
  cross-domain behavior

Keep token ROI metrics:

- prompt tokens
- raw completion tokens
- cleaned completion tokens
- stripped reasoning tokens, if any
- judge token overhead separately from target inference tokens

## Success Criteria

Promote the approach only if the augmented adapter improves quality without
large regressions:

- better pairwise preference against same-rank baseline, or materially better
  single-axis rubric score
- no increase in hallucination/unsupported-answer failures
- no validation-loss regression that indicates synthetic drift
- synthetic rows accepted at a reasonable rate after filtering
- no test/validation leakage found in audit

If the result is neutral or worse, document the negative result and keep the
baseline grounded-only method as the default.

## Documentation Updates After Results

If the augmented adapters improve quality, update:

- `docs/stage-2-dataset-creation.md`: describe Data Designer as optional
  synthetic augmentation after grounded entailment extraction
- `docs/stage-3-peft-training.md`: add augmented adapter recipe and results
- `evals/training_session.log`: add Customizer jobs, losses, and artifact refs
- `deploy/evaluation-matrix/README.md`: add augmented 1B completion/eval target
- MLflow integration docs: log Kimi seed, Data Designer job, dataset entity,
  Customizer jobs, adapter artifacts, and eval metrics under one parent run

## Open Implementation Tasks

1. Add seed builder: `scripts/pipeline/build_data_designer_seed_from_grounded.py`.
2. Add or extend Data Designer collect mode for seed-from-grounded outputs.
3. Add synthetic admission/merge script for `training.jsonl` plus provenance.
4. Extend dataset finalization/registration docs for experiment dataset entity.
5. Extend `train_adapter.py` to accept custom dataset and adapter output names.
6. Add proxy routes and completion collection job for augmented 1B targets.
7. Run r16/r32 training and collect held-out completions.
8. Run evaluator/judge comparison against existing 1B baseline artifacts.

## Completed Synthetic Generation Run: 2026-05-29

Data Designer job: `job-lksewy1b4owhpevuhjcesx`

Generator: Kimi K2 through the Data Designer `llm-structured` column path.

Result summary:

- seed records requested: 200
- seed records returned: 193
- seed records omitted by provider failures: 7
- raw synthetic QA pairs: 579
- accepted synthetic QA pairs after exact duplicate filtering: 577
- rejected duplicate pairs: 2
- trainable think-tag hits after stripping: 0
- reasoning side column excluded from trainable artifacts: `qa_pairs_json__reasoning_trace`
- estimated Data Designer generation tokens: 290,629 prompt, 59,378 completion, 350,007 total

Important artifact paths:

```text
<DATASET_ROOT>/experiments/nim_curated_dd_kimi_1b_20260529_kimi/data_designer/result_manifest.json
<DATASET_ROOT>/experiments/nim_curated_dd_kimi_1b_20260529_kimi/data_designer/generated_raw.jsonl
<DATASET_ROOT>/experiments/nim_curated_dd_kimi_1b_20260529_kimi/data_designer/accepted_samples.jsonl
<DATASET_ROOT>/experiments/nim_curated_dd_kimi_1b_20260529_kimi/stage1_5_data_designer_synthetic.jsonl
<DATASET_ROOT>/experiments/nim_curated_dd_kimi_1b_20260529_kimi/provenance/synthetic_samples.jsonl
<DATASET_ROOT>/experiments/nim_curated_dd_kimi_1b_20260529_kimi/data_designer/omitted_seed_records.jsonl
```

Operational notes:

- The standard Data Designer dataset download endpoint returned HTTP 500 because the service-side download helper lacked NeMo Data Store credentials and received a 401 from Data Store.
- The result repo was recovered directly from NeMo Data Store as `default/job-results-job-lksewy1b4owhpevuhjcesx`.
- The cloned parquet result is preserved at `data_designer/results/job-lksewy1b4owhpevuhjcesx/repo/dataset/batch_00000.parquet`.
- Source-support judging and NeMo Curator admission have not yet been run; the accepted set has passed structural, split, exact duplicate, and think-tag checks only.

## Deferred Follow-Up: Full-Seed Dense Augmentation

Decision date: 2026-05-30

This follow-up is intentionally deferred until the interrupted single-axis and
pairwise evaluation matrix completes. The current evaluation establishes the
baseline for grounded-only LoRA adapters, dense base models, Nano, and 49B. The
full-seed Data Designer experiment would materially expand both generation and
training workloads, so it should not replace the current evaluation checkpoint.

### Objective

Generate a large canonical synthetic pool from the full grounded training split
for each collection, then derive model-sized training mixes for 1B, 3B, 8B, and
Nano/30B-style adapters. The goal is to measure whether synthetic scale improves
parametric, no-RAG behavior and whether the ROI differs by model size.

This is not a replacement for RAG. The expected value is a learning curve for
small-model specialization versus 49B parametric-only behavior, context-baked
behavior, and RAG behavior.

### Full-Seed Sizing

Use training rows only as seeds. Keep validation and test sets unchanged.

```text
NIM train seeds:            4,870
NeMo Microservices seeds:   4,162
```

At `pairs_per_seed=5`, the raw synthetic request size is:

```text
NIM:                  4,870 x 5 = 24,350 raw pairs
NeMo Microservices:   4,162 x 5 = 20,810 raw pairs
Combined:                         45,160 raw pairs
```

Using the first NIM Data Designer run as the rough acceptance baseline, expected
accepted rows are approximately:

```text
NIM accepted synthetic:                ~23.4k
NeMo Microservices accepted synthetic: ~20.0k
Combined accepted synthetic:           ~43.4k
```

Approximate merged train sizes would be:

```text
NIM augmented train:                ~28.3k rows
NeMo Microservices augmented train: ~24.2k rows
Combined augmented train:           ~52.5k rows
```

These estimates must be recomputed from actual `result_manifest.json` and
admission-filter outputs after generation.

### Canonical Pool, Not One Monolithic Training Set

Generate one canonical accepted synthetic pool per collection, but do not train
every model on the full pool by default. Instead, create deterministic training
variants from the same accepted pool:

```text
grounded_only
synthetic_1x
synthetic_2x
synthetic_4x
synthetic_full
```

The multiplier is relative to the grounded training-row count. Sampling should
be deterministic, stratified by source URL/product family/task style when those
fields are available, and recorded in a mix manifest.

Recommended mapping:

```text
1B:   grounded_only, synthetic_1x, synthetic_2x
3B:   grounded_only, synthetic_2x, synthetic_4x
8B:   grounded_only, synthetic_4x, synthetic_full
Nano: grounded_only, synthetic_4x, synthetic_full
```

Do not let low-quality or redundant synthetic data dominate smaller models. The
1B and 3B variants should be more selective than the 8B/Nano variants.

### Generation Controls

Recommended Data Designer generation settings for the full-seed experiment:

```text
generator: Kimi K2 through Data Designer
pairs_per_seed: 5
max_tokens: 8192
temperature: 0.3
top_p: 1.0
max_parallel_requests: 1-2 initially
```

`max_parallel_requests` should remain conservative. The Kimi endpoint showed
occasional disconnects during evaluation even at low concurrency, so maximum
level means maximum generation breadth and token budget, not maximum request
pressure.

### Implementation Plan

1. Finish the interrupted single-axis and pairwise evaluation matrix.
2. Freeze the grounded-only baseline results and token ROI summaries.
3. Build full-seed Data Designer seed CSVs from `training.jsonl` only:

```bash
python scripts/pipeline/build_data_designer_seed_from_grounded.py \
  --dataset-dir <DATASET_ROOT>/nim_curated \
  --output-dir <DATASET_ROOT>/experiments/nim_curated_dd_kimi_fullseed_<date> \
  --collection nim_curated \
  --mode prepare \
  --seed-count 4870 \
  --pairs-per-seed 5 \
  --max-tokens 8192

python scripts/pipeline/build_data_designer_seed_from_grounded.py \
  --dataset-dir <DATASET_ROOT>/nemo_usvcs_curated \
  --output-dir <DATASET_ROOT>/experiments/nemo_usvcs_dd_kimi_fullseed_<date> \
  --collection nemo_usvcs_curated \
  --mode prepare \
  --seed-count 4162 \
  --pairs-per-seed 5 \
  --max-tokens 8192
```

4. Submit Data Designer jobs in shards if the service cannot comfortably handle
   one full collection at once. Prefer 2.5k-3k seed records per shard if needed.
5. Collect raw outputs and preserve Data Designer job IDs, result repo IDs,
   seed dataset refs, token metrics, and omitted/failed seed records.
6. Run admission filtering: schema, exact duplicate, near duplicate,
   train/validation/test leakage, source support, and optional NeMo Curator.
7. Materialize canonical accepted synthetic pools per collection.
8. Build deterministic model-sized mix variants (`synthetic_1x`, `2x`, `4x`,
   `full`) with mix manifests and MLflow-ready metrics.
9. Register each variant as a distinct NeMo Data Store / Entity Store dataset.
10. Train selected LoRA ranks per dense model size and evaluate against the
    frozen grounded-only baseline.

### Evaluation Gate

Only promote the full-seed augmentation approach if it improves held-out quality
without worsening unsupported-answer rates or validation loss. Required
comparisons:

```text
parametric-only: test_set.jsonl, no context, no RAG
context-baked:   test_set_with_context.jsonl, no live retrieval
RAG:             question-only prompt with live retrieval
```

Primary hypothesis:

```text
Small LoRA + RAG should beat 49B no-RAG first.
Pure no-RAG LoRA beating 49B consistently requires much larger, high-quality,
diversified synthetic data and is likely domain-local rather than general
reasoning parity.
```

