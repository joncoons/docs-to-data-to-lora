# NIM Judge Delta Tables - 2026-06-08

Scope: NIM datasets only. This report separates model-family coverage so prior e2 results are visible without implying false equivalence across adapter families.

Primary 49B metric: **target non-loss vs 49B** = target wins + ties. This treats ties as useful because the practical question is whether the smaller model avoids losing to 49B.

## Coverage Notes

- Kimi K2.6 has prior 3B e2 non-aug results: `base`, `r16`, `r32`, plus 49B comparisons for `r16/r32`.
- Claude Sonnet 4.6 has 3B dd5x e2, dd5x e5, and non-aug e5 results. It does not have a direct Claude single-axis score for the prior non-aug e2 adapters.
- Nemotron Ultra has NIM 8B results, not NIM 3B results in the saved artifacts.
- Kimi older single-axis runs retain unresolved failed rows; pairwise rows for the repaired Claude 3B set are clean.

## 3B Single-Axis By Judge

| Judge | Family | Variant | Rows | Failed | Accuracy | Completeness | Faithfulness | Clarity | Composite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi K2.6 | 3B prior e2 non-aug | base | 515 | 25 | 3.722 | 3.264 | 3.866 | 4.349 | 3.800 |
| Kimi K2.6 | 3B prior e2 non-aug | r16 | 527 | 13 | 3.852 | 3.144 | 4.110 | 4.594 | 3.925 |
| Kimi K2.6 | 3B prior e2 non-aug | r32 | 527 | 13 | 3.791 | 3.099 | 4.055 | 4.598 | 3.886 |
| Claude Sonnet 4.6 | 3B dd5x e2 | base | 540 | 0 | 3.761 | 3.070 | 3.954 | 4.167 | 3.738 |
| Claude Sonnet 4.6 | 3B dd5x e2 | r16 | 540 | 0 | 3.833 | 2.854 | 4.046 | 4.239 | 3.743 |
| Claude Sonnet 4.6 | 3B dd5x e2 | r32 | 540 | 0 | 3.907 | 2.969 | 4.089 | 4.272 | 3.809 |
| Claude Sonnet 4.6 | 3B dd5x e5 | r16 | 540 | 0 | 3.865 | 2.878 | 4.026 | 4.243 | 3.753 |
| Claude Sonnet 4.6 | 3B dd5x e5 | r32 | 540 | 0 | 3.843 | 3.013 | 4.033 | 4.233 | 3.781 |
| Claude Sonnet 4.6 | 3B non-aug e5 | r16 | 540 | 0 | 3.820 | 2.970 | 4.011 | 4.263 | 3.766 |
| Claude Sonnet 4.6 | 3B non-aug e5 | r32 | 540 | 0 | 3.850 | 2.859 | 4.072 | 4.270 | 3.763 |

## 3B vs 49B Non-Loss

| Judge | Family | Pair | Rows | Failed | 3B Wins | 49B Wins | Ties | 3B Non-Loss | 3B Non-Loss % | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi K2.6 | 3B prior e2 non-aug | 49B vs r16 | 540 | 0 | 83 | 392 | 65 | 148 | 27.41% | prior e2-era non-aug adapter |
| Kimi K2.6 | 3B prior e2 non-aug | 49B vs r32 | 540 | 0 | 82 | 411 | 47 | 129 | 23.89% | prior e2-era non-aug adapter |
| Claude Sonnet 4.6 | 3B base | 49B vs base | 540 | 0 | 56 | 398 | 86 | 142 | 26.30% | later base completion run |
| Claude Sonnet 4.6 | 3B dd5x e2 | 49B vs r16 | 540 | 0 | 68 | 405 | 67 | 135 | 25.00% | augmented dd5x e2 |
| Claude Sonnet 4.6 | 3B dd5x e2 | 49B vs r32 | 540 | 0 | 82 | 380 | 78 | 160 | 29.63% | augmented dd5x e2 |
| Claude Sonnet 4.6 | 3B dd5x e5 | 49B vs r16 | 540 | 0 | 75 | 409 | 56 | 131 | 24.26% | augmented dd5x e5 |
| Claude Sonnet 4.6 | 3B dd5x e5 | 49B vs r32 | 540 | 0 | 89 | 389 | 62 | 151 | 27.96% | augmented dd5x e5 |
| Claude Sonnet 4.6 | 3B non-aug e5 | 49B vs r16 | 540 | 0 | 80 | 390 | 70 | 150 | 27.78% | grounded non-aug e5 |
| Claude Sonnet 4.6 | 3B non-aug e5 | 49B vs r32 | 540 | 0 | 65 | 404 | 71 | 136 | 25.19% | grounded non-aug e5 |

## 3B e2 Approximate Judge Delta

This table includes the prior e2 Kimi results, but it is **not an exact adapter match**: Kimi is prior non-aug e2; Claude is dd5x e2.

| Pair | Kimi Prior e2 Non-Aug Non-Loss % | Claude dd5x e2 Non-Loss % | Delta |
| --- | --- | --- | --- |
| 49B vs r16 | 27.41% | 25.00% | -2.41% |
| 49B vs r32 | 23.89% | 29.63% | 5.74% |

## 3B Rank Pair Preference

| Judge | Family | Pair | Rows | Failed | Left Wins | Right Wins | Ties | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi K2.6 | 3B prior e2 non-aug | r16 vs r32 | 536 | 4 | 142 | 140 | 254 | prior e2-era non-aug adapters; unresolved failed rows remain |
| Claude Sonnet 4.6 | 3B dd5x e2 | r16 vs r32 | 540 | 0 | 114 | 152 | 274 | augmented dd5x e2 |
| Claude Sonnet 4.6 | 3B dd5x e5 | r16 vs r32 | 540 | 0 | 131 | 183 | 226 | augmented dd5x e5 |
| Claude Sonnet 4.6 | 3B non-aug e5 | r16 vs r32 | 540 | 0 | 146 | 103 | 291 | grounded non-aug e5 |

## 8B Exact Single-Axis Judge Delta

These are exact family overlaps between Kimi and Nemotron Ultra for 8B non-aug NIM runs.

| Variant | Kimi Composite | Ultra Composite | Ultra - Kimi | Kimi Failed | Ultra Failed |
| --- | --- | --- | --- | --- | --- |
| base | 4.114 | 4.247 | 0.133 | 27 | 0 |
| r16 | 4.116 | 4.206 | 0.090 | 16 | 0 |
| r32 | 4.130 | 4.241 | 0.111 | 9 | 1 |

## 8B vs 49B Non-Loss By Judge

| Judge | Family | Pair | Rows | Failed | 8B Wins | 49B Wins | Ties | 8B Non-Loss | 8B Non-Loss % | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi K2.6 | 8B non-aug | 49B vs r16 | 539 | 1 | 90 | 381 | 68 | 158 | 29.31% | exact family overlap with Ultra; unresolved failed row |
| Kimi K2.6 | 8B non-aug | 49B vs r32 | 539 | 1 | 99 | 374 | 66 | 165 | 30.61% | exact family overlap with Ultra; unresolved failed row |
| Nemotron Ultra | 8B non-aug | 49B vs r16 | 540 | 0 | 75 | 392 | 73 | 148 | 27.41% | exact family overlap with Kimi |
| Nemotron Ultra | 8B non-aug | 49B vs r32 | 539 | 1 | 79 | 377 | 83 | 162 | 30.06% | exact family overlap with Kimi; unresolved failed row |
| Nemotron Ultra | 8B dd5x | 49B vs r16 | 540 | 0 | 77 | 381 | 82 | 159 | 29.44% | augmented dd5x |
| Nemotron Ultra | 8B dd5x | 49B vs r32 | 540 | 0 | 85 | 366 | 89 | 174 | 32.22% | augmented dd5x |

## 8B Exact 49B Non-Loss Judge Delta

| Pair | Kimi 8B Non-Loss % | Ultra 8B Non-Loss % | Ultra - Kimi | Kimi Failed | Ultra Failed |
| --- | --- | --- | --- | --- | --- |
| 49B vs r16 | 29.31% | 27.41% | -1.90% | 1 | 0 |
| 49B vs r32 | 30.61% | 30.06% | -0.55% | 1 | 1 |

## 8B Rank Pair Preference

| Judge | Family | Pair | Rows | Failed | Left Wins | Right Wins | Ties | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi K2.6 | 8B non-aug | r16 vs r32 | 538 | 2 | 123 | 156 | 259 | exact family overlap with Ultra; unresolved failed rows |
| Nemotron Ultra | 8B non-aug | r16 vs r32 | 539 | 1 | 114 | 134 | 291 | exact family overlap with Kimi; unresolved failed row |
| Nemotron Ultra | 8B dd5x | r16 vs r32 | 540 | 0 | 141 | 171 | 228 | augmented dd5x |

## Artifacts

- JSON: `<EVAL_ROOT>/aggregates/nim-judge-delta-20260608/judge_delta_tables.json`
- Markdown: `docs/experiments/nim-judge-delta-tables-20260608.md`
