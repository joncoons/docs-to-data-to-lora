# 8B Augmented e5 Learning Rate Analysis

Date: 2026-06-09

Purpose: decide whether 8B augmented LoRA SFT should be extended from the completed e2 runs to e5.

## Current State

Evidence found:

- 8B augmented dd5x e2 jobs completed for NIM and NeMo Microservices, both `r16` and `r32`.
- 8B grounded non-augmented e5 jobs completed for NIM and NeMo Microservices, both `r16` and `r32`.
- 3B NIM dd5x e2 and e5 results exist and provide the closest evidence for the augmented e5 question.
- No live Customizer Volcano jobs are currently running or queued in `nemo-peft`.
- No local evidence was found that 8B augmented dd5x e5 jobs were already queued.

## Scheduler

The Customizer dense LoRA jobs use:

- Peak learning rate: `1e-4`
- Warmup steps: `30`
- Scheduler: cosine decay after warmup
- Batch size: `16`
- Sequence packing: disabled

The meaningful difference between e2 and e5 is not peak LR. It is how long the cosine schedule stays above a useful update rate.

| Run | Steps/Epoch | Epochs | Total Steps | LR End e1 | LR End e2 | LR End e3 | LR End e4 | LR End e5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NIM non-aug e2 | 290 | 2 | 580 | 5.428e-05 | 0.000e+00 |  |  |  |
| NIM non-aug e5 | 290 | 5 | 1450 | 9.195e-05 | 6.733e-05 | 3.582e-05 | 9.943e-06 | 0.000e+00 |
| NIM dd5x e2 | 1560 | 2 | 3120 | 5.076e-05 | 0.000e+00 |  |  |  |
| NIM dd5x e5 | 1560 | 5 | 7800 | 9.073e-05 | 6.580e-05 | 3.478e-05 | 9.621e-06 | 0.000e+00 |
| NeMo non-aug e2 | 248 | 2 | 496 | 5.505e-05 | 0.000e+00 |  |  |  |
| NeMo non-aug e5 | 248 | 5 | 1240 | 9.220e-05 | 6.766e-05 | 3.604e-05 | 1.001e-05 | 0.000e+00 |
| NeMo dd5x e2 | 1247 | 2 | 2494 | 5.096e-05 | 0.000e+00 |  |  |  |
| NeMo dd5x e5 | 1247 | 5 | 6235 | 9.080e-05 | 6.588e-05 | 3.484e-05 | 9.639e-06 | 0.000e+00 |

Interpretation:

- e2 runs end at zero LR after two passes.
- e5 runs are still at about `6.6e-5` at the end of epoch 2, then use epochs 3-5 as lower-LR refinement.
- Augmented e5 is approximately 2.5x the training step cost of augmented e2.
- Augmented e5 is also much larger than grounded e5: NIM is `7800` steps versus `1450`, and NeMo Microservices is `6235` steps versus `1240`.

## Observed Evidence

3B NIM, Claude Sonnet judge:

| Variant | Composite | vs Base | vs 49B Wins | vs 49B Ties | vs 49B Non-Loss |
|---|---:|---|---:|---:|---:|
| dd5x r16 e2 | 3.743 | L 166-200-174 | 68 | 67 | 25.00% |
| dd5x r16 e5 | 3.753 | L 180-203-157 | 75 | 56 | 24.26% |
| dd5x r32 e2 | 3.809 | W 203-183-154 | 82 | 78 | 29.63% |
| dd5x r32 e5 | 3.781 | W 204-184-152 | 89 | 62 | 27.96% |
| non-aug r16 e5 | 3.766 | W 184-179-177 | 80 | 70 | 27.78% |
| non-aug r32 e5 | 3.763 | L 168-185-187 | 65 | 71 | 25.19% |

Direct e2 versus e5 pairwise:

| Pair | e2 Wins | e5 Wins | Ties | Read |
|---|---:|---:|---:|---|
| dd5x r16 e2 vs dd5x r16 e5 | 119 | 153 | 268 | e5 wins direct comparison, but remains weak overall |
| dd5x r32 e2 vs dd5x r32 e5 | 141 | 160 | 239 | e5 wins direct comparison, but e2 has higher non-loss versus 49B |
| non-aug r16 e2 vs non-aug r16 e5 | 97 | 180 | 263 | e5 materially improves grounded r16 |
| non-aug r32 e2 vs non-aug r32 e5 | 104 | 145 | 291 | e5 improves grounded r32 |

8B NIM, Nemotron Ultra judge:

| Family | Pair | Target Wins | 49B Wins | Ties | Target Non-Loss |
|---|---|---:|---:|---:|---:|
| non-aug e2-era | 49B vs r16 | 75 | 392 | 73 | 27.41% |
| non-aug e2-era | 49B vs r32 | 79 | 377 | 83 | 30.06% |
| dd5x e2 | 49B vs r16 | 77 | 381 | 82 | 29.44% |
| dd5x e2 | 49B vs r32 | 85 | 366 | 89 | 32.22% |

## Assessment

The LR schedule supports an e5 test because e2 is a hard two-pass run that reaches zero LR quickly. e5 is not simply a higher-risk larger-step run; it gives more epochs while keeping the same peak LR and tapering updates across epochs 3-5.

The result evidence is mixed:

- Extra epochs clearly helped the grounded non-augmented 3B adapters.
- For augmented 3B, e5 won direct e2-vs-e5 pairwise comparisons for both ranks.
- Against 49B, augmented e5 increased raw wins but did not improve non-loss over augmented e2, because ties dropped.
- The best augmented pattern is consistently `r32`; `r16` has not justified more augmented training by itself.
- 8B dd5x e2 already improved 49B non-loss over the 8B non-aug e2-era comparison, especially `r32`.

## Recommendation

Do not queue all four 8B augmented e5 jobs automatically yet.

Practical next step:

1. Finish/aggregate the 8B non-aug e5 evaluation already created.
2. If 8B non-aug e5 improves over the e2-era 8B adapters, run a targeted 8B augmented e5 test for NIM `r32` first.
3. If NIM `r32` augmented e5 beats augmented e2 in both direct pairwise and 49B non-loss, then train NeMo Microservices `r32`.
4. Only run augmented `r16` e5 if there is a specific reason to test rank sensitivity; current evidence favors `r32`.

The strongest hypothesis is not "augmented e5 for every adapter." It is "augmented e5 may be worthwhile for `r32`, but should be gated by the completed 8B non-aug e5 eval and one targeted 8B dd5x r32 run."
