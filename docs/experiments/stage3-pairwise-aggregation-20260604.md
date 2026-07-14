# Stage 3 Pairwise Aggregation and Augmentation Hypothesis

Generated from persisted Kimi artifacts on 2026-06-04. Stage 3 uses eval run `20260602T161342Z` and compares `49b-base` against each target-matched LoRA response set.

## Stage 3 Ranking Against 49B

Primary ranking metric: LoRA win rate against `49b-base`; non-loss rate includes ties. Lower failed-row counts are better but are small here.

### NIM
| Dataset | LoRA | Rows | Failed | LoRA wins | 49B wins | Ties | LoRA win % | Non-loss % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nim_curated | nano-nim-r16 | 540 | 0 | 129 | 326 | 85 | 23.89 | 39.63 |
| nim_curated | 8b-r32 | 539 | 1 | 99 | 374 | 66 | 18.37 | 30.61 |
| nim_curated | 8b-r16 | 539 | 1 | 90 | 381 | 68 | 16.7 | 29.31 |
| nim_curated | 3b-r16 | 540 | 0 | 83 | 392 | 65 | 15.37 | 27.41 |
| nim_curated | 3b-r32 | 540 | 0 | 82 | 411 | 47 | 15.19 | 23.89 |
| nim_curated | 1b-aug-r32 | 540 | 0 | 58 | 448 | 34 | 10.74 | 17.04 |
| nim_curated | 1b-aug-r16 | 539 | 1 | 53 | 458 | 28 | 9.83 | 15.03 |
| nim_curated | 1b-r16 | 538 | 2 | 49 | 446 | 43 | 9.11 | 17.1 |
| nim_curated | 1b-r32 | 539 | 1 | 44 | 458 | 37 | 8.16 | 15.03 |

### NeMo-USVCS
| Dataset | LoRA | Rows | Failed | LoRA wins | 49B wins | Ties | LoRA win % | Non-loss % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nemo_usvcs_curated | 8b-r16 | 462 | 0 | 132 | 270 | 60 | 28.57 | 41.56 |
| nemo_usvcs_curated | 8b-r32 | 462 | 0 | 125 | 281 | 56 | 27.06 | 39.18 |
| nemo_usvcs_curated | 3b-r32 | 462 | 0 | 107 | 306 | 49 | 23.16 | 33.77 |
| nemo_usvcs_curated | 3b-r16 | 462 | 0 | 107 | 310 | 45 | 23.16 | 32.9 |
| nemo_usvcs_curated | 1b-r32 | 462 | 0 | 82 | 346 | 34 | 17.75 | 25.11 |
| nemo_usvcs_curated | nano-nemo-usvcs-r16-retrain | 462 | 0 | 79 | 349 | 34 | 17.1 | 24.46 |
| nemo_usvcs_curated | 1b-r16 | 462 | 0 | 68 | 367 | 27 | 14.72 | 20.56 |

## Stage 2 Within-Family Pairwise

Correction noted after review: the saved Phase 2 matrix is incomplete for dense models. It includes LoRA rank/variant comparisons and Nano base-vs-LoRA, but it does not include 3B-base-vs-3B-LoRA or 8B-base-vs-8B-LoRA comparisons. The matrix builder has been updated so future Phase 2 runs include base-vs-adapter comparisons for every base family before rank/variant comparisons. The missing non-augmented dense comparisons should be backfilled when Kimi endpoint pressure allows.

| Dataset | Pair | Rows | Failed | Left wins | Right wins | Ties | Left % | Right % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nemo_usvcs_curated | 1b-r16__vs__1b-r32 | 462 | 0 | 118 | 151 | 193 | 25.54 | 32.68 |
| nemo_usvcs_curated | 3b-r16__vs__3b-r32 | 461 | 1 | 102 | 131 | 228 | 22.13 | 28.42 |
| nemo_usvcs_curated | 8b-r16__vs__8b-r32 | 462 | 0 | 106 | 112 | 244 | 22.94 | 24.24 |
| nemo_usvcs_curated | nano-base__vs__nano-nemo-usvcs-r16-retrain | 462 | 0 | 327 | 91 | 44 | 70.78 | 19.7 |
| nim_curated | 1b-aug-r16__vs__1b-aug-r32 | 538 | 2 | 143 | 164 | 231 | 26.58 | 30.48 |
| nim_curated | 1b-r16__vs__1b-aug-r16 | 535 | 5 | 157 | 134 | 244 | 29.35 | 25.05 |
| nim_curated | 1b-r16__vs__1b-r32 | 537 | 3 | 150 | 134 | 253 | 27.93 | 24.95 |
| nim_curated | 1b-r32__vs__1b-aug-r32 | 538 | 2 | 137 | 171 | 230 | 25.46 | 31.78 |
| nim_curated | 3b-r16__vs__3b-r32 | 536 | 4 | 142 | 140 | 254 | 26.49 | 26.12 |
| nim_curated | 8b-r16__vs__8b-r32 | 538 | 2 | 123 | 156 | 259 | 22.86 | 29.0 |
| nim_curated | nano-base__vs__nano-nim-r16 | 537 | 3 | 186 | 183 | 168 | 34.64 | 34.08 |

## Single-Axis Context

Dense single-axis rows are from `20260530T195331Z`; Nano single-axis rows are from `20260602T161342Z`, so use this as context, not as a strict apples-to-apples endpoint comparison.

| Dataset | Run | Model | Rows | Failed | Faith | Acc | Comp | Clarity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nemo_usvcs_curated | 20260530T195331Z | 8b-r16 | 458 | 4 | 4.421 | 4.303 | 3.376 | 4.865 |
| nemo_usvcs_curated | 20260530T195331Z | 8b-r32 | 454 | 8 | 4.419 | 4.291 | 3.412 | 4.813 |
| nemo_usvcs_curated | 20260530T195331Z | llama-3.3-nemotron-super-49b-v1.5-base | 462 | 0 | 4.279 | 4.177 | 3.952 | 4.814 |
| nemo_usvcs_curated | 20260530T195331Z | llama-3.1-8b-base | 458 | 4 | 4.19 | 3.974 | 3.445 | 4.59 |
| nemo_usvcs_curated | 20260602T161342Z | nemotron-3-nano-30b-a3b-base | 462 | 0 | 4.128 | 4.117 | 3.788 | 4.814 |
| nemo_usvcs_curated | 20260530T195331Z | 3b-r32 | 453 | 9 | 4.124 | 4.049 | 3.247 | 4.656 |
| nemo_usvcs_curated | 20260530T195331Z | 3b-r16 | 459 | 3 | 4.115 | 3.95 | 3.148 | 4.691 |
| nemo_usvcs_curated | 20260530T195331Z | llama-3.2-3b-base | 458 | 4 | 3.95 | 3.731 | 3.251 | 4.459 |
| nemo_usvcs_curated | 20260530T195331Z | 1b-r32 | 459 | 3 | 3.706 | 3.49 | 2.819 | 4.342 |
| nemo_usvcs_curated | 20260530T195331Z | 1b-r16 | 461 | 1 | 3.601 | 3.369 | 2.675 | 4.295 |
| nemo_usvcs_curated | 20260602T161342Z | nano-nemo-usvcs-r16-retrain | 462 | 0 | 3.463 | 3.169 | 2.922 | 4.002 |
| nemo_usvcs_curated | 20260530T195331Z | llama-3.2-1b-base | 457 | 5 | 3.133 | 3.018 | 2.615 | 3.81 |
| nim_curated | 20260530T195331Z | llama-3.3-nemotron-super-49b-v1.5-base | 539 | 1 | 4.471 | 4.34 | 4.063 | 4.859 |
| nim_curated | 20260530T195331Z | 8b-r32 | 531 | 9 | 4.392 | 4.081 | 3.326 | 4.723 |
| nim_curated | 20260530T195331Z | 8b-r16 | 524 | 16 | 4.347 | 4.09 | 3.282 | 4.744 |
| nim_curated | 20260530T195331Z | llama-3.1-8b-base | 513 | 27 | 4.263 | 4.084 | 3.526 | 4.585 |
| nim_curated | 20260602T161342Z | nano-nim-r16 | 539 | 1 | 4.182 | 4.247 | 3.868 | 4.801 |
| nim_curated | 20260602T161342Z | nemotron-3-nano-30b-a3b-base | 538 | 2 | 4.145 | 4.229 | 3.862 | 4.818 |
| nim_curated | 20260530T195331Z | 3b-r16 | 527 | 13 | 4.11 | 3.852 | 3.144 | 4.594 |
| nim_curated | 20260530T195331Z | 3b-r32 | 527 | 13 | 4.055 | 3.791 | 3.099 | 4.598 |
| nim_curated | 20260530T195331Z | llama-3.2-3b-base | 515 | 25 | 3.866 | 3.722 | 3.264 | 4.35 |
| nim_curated | 20260530T195331Z | 1b-r16 | 536 | 4 | 3.703 | 3.246 | 2.599 | 4.196 |
| nim_curated | 20260530T195331Z | 1b-aug-r16 | 535 | 5 | 3.664 | 3.206 | 2.54 | 4.189 |
| nim_curated | 20260530T195331Z | 1b-aug-r32 | 536 | 4 | 3.647 | 3.302 | 2.664 | 4.174 |
| nim_curated | 20260530T195331Z | 1b-r32 | 537 | 3 | 3.575 | 3.188 | 2.596 | 4.162 |
| nim_curated | 20260530T195331Z | llama-3.2-1b-base | 532 | 8 | 3.096 | 2.981 | 2.62 | 3.726 |

## Initial Interpretation

- The best Stage 3 LoRA by win rate against 49B is `nemo_usvcs_curated/8b-r16`: 132 LoRA wins, 270 49B wins, 60 ties, 28.57% win rate, 41.56% non-loss rate.
- `nemo_usvcs_curated/8b-r32` is close behind at 27.06% win rate and 39.18% non-loss rate.
- On NIM, the strongest Stage 3 LoRA is `nano-nim-r16` at 23.89% win rate and 39.63% non-loss rate against 49B, followed by `8b-r32` at 18.37%.
- 1B augmented adapters did not close the 49B gap in Stage 3. The better augmented 1B variant, `1b-aug-r32`, reached 10.74% win rate against 49B, versus 9.09% for original `1b-r16` and 8.16% for original `1b-r32`. This suggests augmentation can help, but not enough at the 1B scale to change the model-class outcome.
- NeMo-USVCS 8B is the first candidate for the next augmentation experiment because it is already closest to 49B on the Stage 3 metric.


## Grounded-Seed Multiplier Sizing

Use training rows as augmentation seeds; keep validation/test rows held out to avoid leakage.

| Corpus | Grounded training rows | 2x synthetic + grounded | 3x synthetic + grounded | 5x synthetic + grounded | 8x synthetic + grounded |
| --- | ---: | ---: | ---: | ---: | ---: |
| NIM | 4,870 | 14,610 | 19,480 | 29,220 | 43,830 |
| NeMo-USVCS | 4,162 | 12,486 | 16,648 | 24,972 | 37,458 |

The completed 1B augmentation experiment was intentionally smaller: 200 seed records yielded 577 accepted synthetic rows, producing 5,447 total NIM training rows. That was enough to test the mechanics, but it was not a full-corpus augmentation strategy.

## Augmentation Experiment Hypothesis

If augmentation benefit scales with target model capacity, the next test should not reuse the 1B synthetic size blindly. Use the full ground-truth dataset as seeds and produce a larger synthetic set for the best target family, with multipliers bounded by data quality and review cost rather than parameter count alone.

Recommended next experiment:

- Target: NeMo-USVCS 8B, both `r16` and `r32`, because Stage 3 shows the 8B adapters have the strongest non-loss rate against 49B.
- Dataset: NeMo-USVCS ground-truth training set plus Kimi/Data Designer synthetic augmentation generated from all eligible ground-truth training rows.
- Starting multiplier: 2x to 3x synthetic per grounded seed, not 8x. A literal parameter-count multiplier from 1B to 8B would likely create a token-expensive dataset with high redundancy. Stage the experiment so we can stop if quality saturates.
- Escalation path: if 2x/3x improves both single-axis and Stage 3 pairwise without increasing factual errors, then try 5x. Reserve 8x for a later saturation test.
- Evaluation: rerun completions for 8B augmented `r16/r32`, then run single-axis and Stage 3 49B pairwise only for the augmented 8B variants first. Full matrix expansion can wait until the targeted test proves value.

Open question: run the same augmentation on NIM 8B or Nano NIM after NeMo-USVCS 8B. Nano NIM has a strong non-loss rate, but its NeMo-USVCS LoRA regressed badly against base, so it is less clean as the first augmentation candidate.
