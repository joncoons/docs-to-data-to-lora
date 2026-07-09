# Full-Seed Kimi 5x Data Designer Augmentation - 2026-06-04

## Objective

Generate 5x synthetic augmentation pools for both grounded training collections, then merge accepted synthetic rows into new training datasets while preserving validation/test splits unchanged.

This is the larger follow-up to the earlier NIM 1B augmentation experiment. The 2x and 3x multipliers remain useful later ablation points, but this run intentionally targets the maximum 5x pool first.

Update: the initial Kimi-authored seed-brief phase was cancelled because it added too much wall-clock time before native Data Designer generation. The active replacement uses deterministic seed rows built directly from every eligible grounded training row. Kimi remains the native Data Designer generation model, but there is no separate Kimi seed-brief pass.

## Collections

| Collection | Source dataset | Training rows | Eligible seed rows | Requested synthetic pairs | Merged dataset path |
|---|---:|---:|---:|---:|---|
| `nim_curated` | `/mnt/nvme2/peft/datasets/v2/nim_curated` | 4,870 | 4,154 | 20,770 | `/mnt/nvme2/peft/datasets/experiments/nim_curated_dd_kimi_5x_20260604` |
| `nemo_usvcs_curated` | `/mnt/nvme2/peft/datasets/v2/nemo_usvcs_curated` | 4,162 | 3,557 | 17,785 | `/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_kimi_5x_20260604` |

Eligible seed rows are training rows after exact prompt/completion pairs present in validation/test artifacts are excluded to prevent synthetic leakage into held-out evaluation.

Active deterministic replacement directories:

| Collection | Deterministic experiment dir | Data Designer job id | Status |
|---|---|---|---|
| `nim_curated` | `/mnt/nvme2/peft/datasets/experiments/nim_curated_dd_deterministic_5x_20260604` | `job-fhtk8u175n6d5yah7vvmww` | active |
| `nemo_usvcs_curated` | `/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_deterministic_5x_20260604` | `job-cqubk2j9fw3xcz7ztpr6zv` | active |

## Kimi Configuration

The cancelled seed-brief run used the NVIDIA endpoint directly:

- endpoint: `https://inference-api.nvidia.com/v1/chat/completions`
- model: `nvidia/moonshotai/kimi-k2.6`
- max tokens: `8192`
- local key source: ephemeral local file, not persisted in artifacts

Native NeMo Data Designer generation uses provider `kimi-k2`, updated in `nemo-peft/nemo-data-designer-config`:

- endpoint: `https://inference-api.nvidia.com/v1`
- allowed model: `nvidia/moonshotai/kimi-k2.6`
- secret env: `KIMI_KEY` from `nemo-peft/kimi-judge-api`

Backups were written under `archive/cluster-backups/` before the ConfigMap/deployment change.

## Running State

The Kimi-authored seed generation processes were stopped on 2026-06-04 after partial progress:

- NIM: about 281 of 4,154 seed briefs, 1 fallback
- NeMo-USVCS: about 332 of 3,557 seed briefs, 0 fallbacks

Those partial artifacts remain in the original `*_dd_kimi_5x_20260604` directories for audit only.

The active deterministic replacement has no local seed-generation process. It built local seed tables with:

```bash
python scripts/pipeline/build_data_designer_seed_from_grounded.py \
  --mode prepare \
  --seed-count 0 \
  --pairs-per-seed 5
```

The native Data Designer jobs are active and own the long-running generation work.

Original cancelled process files:

- NIM PID file: `/mnt/nvme2/peft/datasets/experiments/nim_curated_dd_kimi_5x_20260604/seed_generation.pid`
- NeMo-USVCS PID file: `/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_kimi_5x_20260604/seed_generation.pid`
- per-collection progress: `data_designer/kimi_seed_progress.json`
- per-collection append log: `data_designer/kimi_seed_requests.jsonl`
- per-collection failures: `data_designer/kimi_seed_failures.jsonl` if any failures require heuristic fallback

The original post-seed workflow orchestrator was also stopped:

- PID file: `/mnt/nvme2/peft/datasets/experiments/grounded_5x_kimi_workflow_20260604.pid`
- log: `/mnt/nvme2/peft/datasets/experiments/grounded_5x_kimi_workflow_20260604.log`

The deterministic replacement should be completed with the same post-generation steps after each Data Designer job reaches a terminal successful state:

1. collect native Data Designer results
2. result repo collection
3. grounded result normalization to `accepted_samples.jsonl`
4. merge into the experiment dataset directory

## New/Updated Local Tooling

- `scripts/pipeline/run_resumable_kimi_seed_generation.py`
  - appends each completed Kimi seed record durably
  - resumes by stable seed id
  - logs failed Kimi records and can fall back to heuristic seed briefs after retries

- `scripts/pipeline/collect_data_designer_grounded_results.py`
  - reads Data Designer result repos, including parquet via DuckDB/fastparquet fallback
  - strips think/reasoning text from trainable rows
  - excludes exact duplicates against grounded/eval source artifacts and synthetic output
  - writes `generated_raw.jsonl`, `generated_samples.jsonl`, `accepted_samples.jsonl`, `provenance/synthetic_samples.jsonl`, and `result_manifest.json`

- `scripts/pipeline/data_designer_grounded_augmentation.py`
  - now accepts experiment label/description so job metadata is not mislabeled as the earlier 1B run

## Verification Completed

- Kimi direct seed smoke: passed with one NIM seed.
- Resumable seed runner smoke: passed with three NIM seeds and zero fallbacks.
- Native Data Designer preview: passed against provider `kimi-k2` and model `nvidia/moonshotai/kimi-k2.6`.
- Collector smoke against prior 1B result repo: reproduced 577 accepted / 2 rejected / 7 omitted.
- Focused tests: `8 passed` for seed builder, collector, and merge contracts.

## Status Check Commands

```bash
for d in   /mnt/nvme2/peft/datasets/experiments/nim_curated_dd_kimi_5x_20260604   /mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_kimi_5x_20260604; do
  echo "=== $(basename "$d")"
  wc -l "$d/data_designer/kimi_seed_requests.jsonl"
  cat "$d/data_designer/kimi_seed_progress.json"
  test -f "$d/data_designer/kimi_seed_failures.jsonl" && wc -l "$d/data_designer/kimi_seed_failures.jsonl" || true
done

tail -40 /mnt/nvme2/peft/datasets/experiments/grounded_5x_kimi_workflow_20260604.log
```

## NeMo-USVCS Timeout Repair Runner

A post-primary repair runner was launched after observing provider timeout omissions in the NeMo-USVCS deterministic Data Designer job. It waits for the primary job to complete, collects and normalizes the primary results, builds a repair seed dataset from final `omitted_seed_records.jsonl`, submits a lower-concurrency repair job, and then builds a combined augmented dataset.

- runner PID file: `/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_deterministic_5x_20260604/data_designer/repair_after_primary.pid`
- runner log: `/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_deterministic_5x_20260604/data_designer/repair_after_primary.runner.log`
- repair experiment dir: `/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_deterministic_5x_repair_20260605`
- combined dataset dir: `/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_deterministic_5x_combined_20260605`
- repair settings: `max_parallel_requests=2`, `timeout_s=1800`, same `nvidia/moonshotai/kimi-k2.6` Data Designer provider.

Current log-derived failure snapshot before the runner was launched:

- NeMo-USVCS timeout omissions: 91
- internal server omissions: 0
- last observed batch: 4 of 8

The final repair seed count should be taken from the collector output after the primary job finishes, not from this interim log snapshot.

### Completion Update - 2026-06-05

The primary NeMo-USVCS Data Designer job completed successfully, but the repair runner exited before final collection because the repair-specific `submission_plan.json` did not contain the same `data_designer` object as the original deterministic plan. The helper has been fixed to tolerate both schemas.

A second issue was found in result normalization: reading all Data Designer parquet batches at once through DuckDB nulled the nested `qa_pairs_json` column. The collector now reads parquet batches one file at a time, preserving the structured generation output.

Final NeMo-USVCS deterministic 5x outputs:

- primary job: `job-cqubk2j9fw3xcz7ztpr6zv`
- primary returned seeds: 3,466 of 3,557
- primary accepted synthetic pairs: 15,357
- repair job: `job-fpad47un7qtebohirxppxu`
- repair returned seeds: 91 of 91
- repair accepted synthetic pairs: 453
- combined unique accepted synthetic pairs: 15,788
- combined dataset dir: `/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_deterministic_5x_combined_20260605`
- combined estimated Kimi tokens: 5,902,821 total; 4,764,592 prompt; 1,138,229 completion

The combined dataset appends synthetic rows only to `training.jsonl` and `adapter_train.jsonl`; validation and test artifacts remain unchanged.

## NIM Auto-Resume After NeMo

A separate watcher was launched to resume the suspended NIM deterministic Data Designer job after the NeMo-USVCS workflow completes successfully. This waits for the NeMo repair runner to exit with a success marker, so NIM concurrency 8 does not overlap the lower-concurrency NeMo repair job.

- watcher PID file: `/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_deterministic_5x_20260604/data_designer/resume_nim_after_nemo.pid`
- watcher log: `/mnt/nvme2/peft/datasets/experiments/nemo_usvcs_curated_dd_deterministic_5x_20260604/data_designer/resume_nim_after_nemo.log`
- NIM Kubernetes job to resume: `nemo-peft/jobstep-upcymm4yu2wznzchskrhtp`
- NIM Data Designer settings retained from the original submitted job: `max_parallel_requests=8`, `timeout_s=600`.
- 2026-06-05 update: the watcher stopped after the repair runner exited without a success marker, so the NIM job was manually resumed after the NeMo primary and repair results were collected and merged. Kubernetes reported `suspend=false`, `active=1`, and pod `jobstep-upcymm4yu2wznzchskrhtp-vgkzr` running `2/2`.

## Follow-Up Design Questions

The current run keeps the earlier NIM 1B augmentation pattern: each eligible
grounded training row is first converted into a Kimi-authored Data Designer seed
brief, and native Data Designer then uses Kimi to generate the requested
synthetic QA pairs.

This seed-brief phase is not a replacement for the grounded datasets. It is a
handoff layer that adds per-row generation intent, coverage axis, constraints,
and Data Designer prompt fields while preserving source provenance.

Two future improvements should be evaluated:

1. Compare Kimi-authored seed briefs against deterministic seed briefs generated
   directly from the grounded rows. This would quantify whether the extra Kimi
   pass improves synthetic data quality enough to justify its token cost and
   wall-clock time.
2. Refactor the grounded dataset schema so Data Designer-ready fields are
   emitted during dataset creation. If the grounded datasets already carry
   `coverage_axis`, `generation_brief`, `pairs_count`, source context, and
   constraints, the seed-brief phase can become optional or be skipped for
   large augmentation runs.

The 2x and 3x augmentation multipliers remain useful follow-up ablations. If the
5x run improves downstream dense-model performance, smaller multipliers should
be tested to find the most efficient synthetic-data scale before making this a
default training recipe.
