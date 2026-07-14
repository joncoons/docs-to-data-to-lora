# Execution plan: NeMo Curator versus logical-entailment extraction

Date scoped: 2026-06-29

Status: proposed; implementation and execution have not started.

## 1. Decision to make

Determine whether NeMo Curator's native document-to-QA workflow produces a
better corpus-derived SFT dataset than the current custom LE extraction, where
"better" is evaluated at three distinct levels:

1. **Extractor quality:** groundedness, answerability, usefulness, diversity,
   coverage, and duplicate rate before common filtering.
2. **Curated dataset quality:** the same measures after applying an identical
   Curator quality/deduplication policy to both arms.
3. **Downstream utility:** performance of otherwise identical LoRA adapters
   trained on count-matched datasets.

The experiment must not claim that Curator is better merely because it emits
more rows, filters more aggressively, or uses a different model.

This is not a universal replacement test. Curator remains the preferred NVAIE
foundation for massive-scale curation, filtering, synthetic workflows, and
full-SFT dataset preparation. The experiment asks a narrower operational
question: for small, corpus-specific LoRA adaptation, does an LE-generated QA
KVP dataset yield better adapter utility than Curator DiverseQA on the same
source material under matched training and evaluation conditions?

## 2. What is and is not being compared

### In scope

- Two source collections: `nim_curated` and `nemo_usvcs_curated`.
- The corpus-to-QA portion corresponding to Stage 1A.
- NeMo Curator 26.04 / v1.1.2 as the proposed pinned release, subject to a
  successful image/API preflight and capture of the immutable image digest.
- Curator's native `DiverseQAStage` and published Nemotron-CC preprocessing and
  postprocessing helpers.
- Common Curator answerability/easiness scoring, exact deduplication, and fuzzy
  deduplication applied symmetrically to both arms.
- Intrinsic, human-reviewed, operational, and downstream-model evaluation.

### Out of scope

- Replacing or modifying the production dataset pipeline.
- Stage 1B cross-passage synthesis, Stage 1C instruction diversification,
  Stage 1.5 gap filling, or Data Designer augmentation.
- Comparing the existing `deploy/curator/native-filter-job.yaml` with LE. That
  job is post-generation filtering, not corpus-to-QA extraction.
- Semantic deduplication in the first experiment; it adds an embedding/model
  choice and is unnecessary at this corpus size.
- Promoting generated artifacts to canonical datasets or deploying a winning
  adapter.
- Treating an inconclusive result as evidence of equivalence.

## 3. Frozen inputs and known baselines

The following files are inputs only:

| Collection | Passages | Passage SHA-256 | Existing Stage 1A rows |
|---|---:|---|---:|
| `nim_curated` | 498 | `ea949c2c738ffaaa98716fe303af99f3f75db058244b1d1531bc9164dfaa783f` | 4,804 |
| `nemo_usvcs_curated` | 486 | `5c4586499c3e77def8c14b2231bc49d57d56777a36eea1f9efdb1a964017d5a7` | 4,783 |

Canonical paths:

```text
/mnt/nvme2/peft/datasets/v2/nim_curated/passages.jsonl
/mnt/nvme2/peft/datasets/v2/nim_curated/stage1a_le.jsonl
/mnt/nvme2/peft/datasets/v2/nemo_usvcs_curated/passages.jsonl
/mnt/nvme2/peft/datasets/v2/nemo_usvcs_curated/stage1a_le.jsonl
```

The execution manifest must recompute and verify these hashes. A mismatch
creates a new experiment version; it must not silently update this plan.

The existing Stage 1A artifacts are the full-run control for quality and model
utility. A small control rerun is included in the pilot only to collect
comparable latency/token telemetry and verify that the current model endpoint
still reproduces acceptable behavior. This avoids paying for a full duplicate
LE run while still exposing model or prompt drift.

## 4. Experimental arms

### Arm A: logical-entailment control

- Use rows from the frozen canonical `stage1a_le.jsonl`, selected only by
  `passage_id` according to the experiment's source split.
- Preserve the original question, answer, context, passage ID, and source URL.
- Do not run Stage 2 refinement for the primary comparison; it could mask
  extractor differences.
- In the pilot telemetry rerun, use the current LE prompts and the same pinned
  inference model used by Arm B. Record the prompt content and hash rather than
  assuming the checked-in prompt was the one used historically.

### Arm B: Curator native treatment

- Read the same frozen passages and preserve `passage_id`, source URL, product,
  document kind, source chunk IDs, and source revision ID through every stage.
- Use the release-pinned native `DiverseQAStage` with NVIDIA's published
  preprocessing and `DiverseQAPostProcessingStage` behavior.
- Do not replace the native prompt with the LE prompt. The native generation
  strategy is the treatment.
- Use the same OpenAI-compatible NIM model endpoint and model identifier as the
  pilot control rerun so the primary causal difference is extraction method.
- Use a fixed generation configuration for both pilot arms where supported.
  Proposed settings are temperature `0.2`, fixed maximum output tokens, and no
  reasoning output. If Curator requires a materially different setting, record
  that deviation and add a sensitivity run using Curator's documented example
  (`temperature=0.5`, `top_p=0.9`) rather than hiding the confound.
- Retain raw LLM responses, parsed pairs, parse errors, and rejected pairs in
  separate append-only artifacts.

### Common curation layer

Run the same policy against normalized output from both arms:

1. Schema and non-empty-field validation.
2. Exact deduplication on normalized `(question, answer)` and a second report on
   question-only duplicates.
3. Curator fuzzy deduplication with one frozen threshold/config.
4. Curator `AnswerabilityFilter` using `(context, question)`.
5. Curator `EasinessFilter` as a score, with the rejection percentile calibrated
   on the pooled pilot rather than independently per arm.
6. Neutral length and malformed-output checks.

Every filter emits scores and reason codes. Report raw results and commonly
curated results separately. Do not apply a treatment-only cleanup step.

## 5. Source split and leakage controls

Create the split before looking at generated QA:

- Group by `passage_id`; no passage may cross train, calibration, or test.
- Stratify as far as sample size permits by collection, product family,
  `doc_kind`, and passage token-count bin.
- Use a committed seed and a stable SHA-256 rank of `seed + passage_id`.
- Proposed full split: 70% train, 10% calibration, 20% held-out source test.
- Use calibration passages for filter thresholds and development only.
- Build the human-reviewed benchmark solely from held-out source passages.
- Scan train versus benchmark for exact and fuzzy question/answer overlap.

The existing repository test sets are retained as a secondary operational
benchmark, but they cannot be the sole primary benchmark because they were
created by the incumbent pipeline and may favor the LE control.

## 6. Two-phase execution

### Phase 0: implementation and preflight

Create only under `curator_dataset/`:

```text
curator_dataset/
  configs/                 # pinned generation, filter, split, and training configs
  deploy/                  # isolated Job templates with unique names and output PVC paths
  src/                     # input adapter, Curator runner, normalization, metrics, manifests
  tests/                   # path safety, schema, split, provenance, and determinism tests
  reports/                 # generated summaries; no secrets or raw licensed corpus text
```

Required preflight checks:

- Resolve and record the Curator container digest; reject `:latest`.
- Confirm the pinned release imports `DiverseQAStage`, its postprocessor, the
  selected executor, and the chosen filters.
- Run one synthetic fixture through the full pipeline without cluster access.
- Verify the NIM endpoint/model with one non-corpus request.
- Verify every output path is outside canonical dataset roots.
- Verify the container runs as a distinct job name and does not mount canonical
  data read-write.
- Capture package inventory/SBOM and all license/NOTICE files.

Exit criterion: all isolation and fixture tests pass and a dry-run manifest is
complete.

### Phase 1: paired pilot

- Select 50 passages per collection from the future training partition using
  the frozen stratified hash rule (100 total).
- Run Curator generation on all 100.
- Rerun LE on the same 100 only for telemetry/reproducibility analysis.
- Also select the matching rows from the historical LE artifacts and compare
  the historical and rerun control distributions.
- Apply the common curation layer to both extraction arms.
- Blindly review at least 100 arm-balanced pairs plus all sampled disagreement
  cases where one arm passes answerability and the other fails.

Pilot go/no-go gates:

- 100% passage/source-ID preservation for parseable outputs.
- No writes outside the experiment output root.
- At least 95% parse success.
- At least 90% of reviewed accepted rows pass groundedness and answer fidelity.
- No critical safety, secret, or license-evidence gap.
- Estimated full-run cost and wall time are explicitly approved.

If a gate fails, fix only experiment-local code/config and rerun the pilot under
a new run ID. Do not tune against held-out test passages.

### Phase 2: full extraction and intrinsic evaluation

- Run Curator over the train and calibration partitions of all 984 passages.
- Materialize the corresponding historical LE rows from those same passage IDs.
- Produce raw and commonly curated datasets for both collections and arms.
- Run automatic scoring and blinded review.
- Freeze the accepted datasets and all manifests before model training.

### Phase 3: downstream adapter evaluation

For each collection, set `N = min(accepted_LE, accepted_Curator)` and create
deterministic, count-matched training sets. Sample in passage-balanced rounds so
one prolific passage cannot dominate an arm. Use the same system prompt,
tokenization, base checkpoint, LoRA rank/alpha/dropout, optimizer, learning-rate
schedule, batch size, epochs or token budget, and seed list.

Recommended ladder:

1. One inexpensive smoke adapter per arm to validate formatting and training.
2. One matched primary training run per arm using the model family/config that
   best represents the current project.
3. Add two more seeds per arm only if the first result is close enough that
   run-to-run variance could change the decision.

Evaluate both adapters on:

- the new source-held-out, human-reviewed benchmark (primary);
- the existing context-baked operational test set (secondary);
- closed-book and supplied-context variants, reported separately;
- position-swapped pairwise judging to reduce ordering bias.

Do not merge either adapter into an existing adapter and do not register it as
production-ready.

## 7. Metrics and decision rule

### Dataset metrics

- QA pairs per input passage and passages with zero output.
- Product/doc-kind/source coverage.
- Parse and API failure rates.
- Exact, fuzzy, and question-only duplicate rates.
- Answerability and easiness distributions.
- Independent groundedness, answer fidelity, hallucination, technical
  usefulness, and cognitive-depth ratings.
- Question/answer token lengths and context-copy ratios.
- Lexical and semantic diversity.
- Yield after each common filter.

### Operational metrics

- Input/output tokens, requests, retries, failures, and estimated inference cost.
- End-to-end wall time and throughput.
- CPU/GPU hours and peak memory.
- Manual review minutes per accepted 1,000 rows.
- Reproducibility and resume behavior.

### Downstream metrics

- Existing single-axis evaluator scores by collection and context mode.
- Pairwise Curator-adapter versus LE-adapter win/tie/loss rates with both orderings.
- Per-product and per-question-type slices.
- Mean and confidence interval across seeds when multiple seeds are run.

### Predeclared conclusion

The primary downstream statistic is the paired Curator-versus-LE win rate on
the new held-out benchmark, with ties split as half a win and a passage-clustered
95% bootstrap confidence interval.

- **Curator better:** lower CI bound is above 0.50, Curator meets the 90%
  groundedness gate, and no major product slice regresses by more than 5 points.
- **Curator worse:** upper CI bound is below 0.50, or it fails the groundedness
  gate while LE passes it.
- **Inconclusive:** all other outcomes.

Operational efficiency is a secondary decision axis and is reported even when
quality is inconclusive. A materially cheaper method may justify a separate
non-inferiority decision, but that threshold must be approved before examining
full-run model results.

## 8. Run artifact contract

Each immutable run directory should contain:

```text
<run-id>/
  manifest.json                 # run ID, timestamps, hashes, seeds, git state
  environment/                  # image digest, packages, SBOM, licenses/notices
  inputs/                       # manifests and hashes; references, not source copies
  splits/                       # passage IDs and split audit
  le/raw/                       # selected immutable control rows
  curator/raw/                  # raw responses and native parsed output
  normalized/{le,curator}/      # common schema
  curated/{le,curator}/         # accepted, rejected, scores, reason codes
  review/                       # blinded sample IDs, rubric, adjudicated labels
  training/{le,curator}/        # count-matched train/validation files and configs
  evaluation/                   # job IDs, raw results, aggregate statistics
  report/                       # final Markdown/JSON comparison
```

Secrets, bearer tokens, and full endpoint credentials must never appear in a
manifest. Store secret references and redacted endpoint identities instead.

## 9. Principal risks and mitigations

| Risk | Mitigation |
|---|---|
| Curator and LE use different models | Same pinned model for the primary run; separate labeled sensitivity run only. |
| Existing LE artifact lacks complete modern telemetry | Pilot control rerun; report historical and rerun controls separately. |
| Higher Curator yield makes training comparison unfair | Count-match accepted rows and balance by passage. |
| Common filters favor Curator output | Freeze thresholds on pooled calibration data and publish per-arm score distributions. |
| Existing tests favor the incumbent | New source-held-out human benchmark is primary; existing tests are secondary. |
| LLM judge shares generator bias | Blind human review plus an independent judge; preserve disagreement cases. |
| Curator API drift | Pin 26.04 and image digest; fixture/API tests before corpus execution. |
| The current Curator deployment uses `latest` and legacy CLI assumptions | Do not reuse or edit it; build an isolated release-pinned Ray pipeline. |
| Canonical artifact overwrite | Read-only mounts, path-deny preflight, unique job names, separate output root. |
| Legal conclusions overreach | Preserve evidence and separate software, model, corpus, and output terms. |

## 10. Deliverables

1. Isolated, tested Curator experiment runner and deployment templates.
2. Frozen split and input manifests.
3. Raw and commonly curated datasets for both arms with row-level provenance.
4. Blinded review set, rubric, labels, and adjudication record.
5. Count-matched adapter datasets and training manifests.
6. Intrinsic, operational, and downstream comparison report.
7. License/SBOM/evidence bundle described in `LEGAL_AND_PROVENANCE.md`.
8. A final recommendation of better, worse, or inconclusive, with limitations.
