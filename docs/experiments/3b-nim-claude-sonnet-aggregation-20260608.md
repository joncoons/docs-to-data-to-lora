# 3B NIM Claude Sonnet Evaluation Aggregate - 2026-06-08

Scope: `nim_curated` test set, `llama-3.2-3b` base and LoRA variants, judged by `azure/anthropic/claude-sonnet-4-6`. Pairwise rows were repaired with the parser update that accepts the first valid JSON object when the judge appends extra text.

## Repair Status

- Deduplicated pairwise comparisons: 23
- Pairwise rows compared: 12,420
- Unresolved pairwise failures: 0
- Omitted duplicate pair summaries: 1

## Phase 1: Single-Axis

| Rank | Model | Rows | Accuracy | Completeness | Faithfulness | Clarity | Composite | Target Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3B dd5x r32 e2 | 540 | 3.907 | 2.969 | 4.089 | 4.272 | 3.809 | 1,297,199 |
| 2 | 3B dd5x r32 e5 | 540 | 3.843 | 3.013 | 4.033 | 4.233 | 3.781 | 1,301,062 |
| 3 | 3B non-aug r16 e5 | 540 | 3.820 | 2.970 | 4.011 | 4.263 | 3.766 | 1,307,659 |
| 4 | 3B non-aug r32 e5 | 540 | 3.850 | 2.859 | 4.072 | 4.270 | 3.763 | 1,309,857 |
| 5 | 3B dd5x r16 e5 | 540 | 3.865 | 2.878 | 4.026 | 4.243 | 3.753 | 1,323,436 |
| 6 | 3B dd5x r16 e2 | 540 | 3.833 | 2.854 | 4.046 | 4.239 | 3.743 | 1,303,551 |
| 7 | 3B base | 540 | 3.761 | 3.070 | 3.954 | 4.167 | 3.738 | 1,354,178 |

## Phase 2: 3B Pairwise

| Pair | Rows | Left Wins | Right Wins | Ties | Left Win % | Right Win % |
| --- | --- | --- | --- | --- | --- | --- |
| 3B base vs 3B dd5x r16 e5 | 540 | 203 | 180 | 157 | 37.59 | 33.33 |
| 3B base vs 3B dd5x r32 e5 | 540 | 184 | 204 | 152 | 34.07 | 37.78 |
| 3B base vs 3B dd5x r16 e2 | 540 | 200 | 166 | 174 | 37.04 | 30.74 |
| 3B base vs 3B dd5x r32 e2 | 540 | 183 | 203 | 154 | 33.89 | 37.59 |
| 3B base vs 3B non-aug r16 e5 | 540 | 179 | 184 | 177 | 33.15 | 34.07 |
| 3B base vs 3B non-aug r32 e5 | 540 | 185 | 168 | 187 | 34.26 | 31.11 |
| 3B dd5x r16 e5 vs 3B dd5x r32 e5 | 540 | 131 | 183 | 226 | 24.26 | 33.89 |
| 3B dd5x r16 e2 vs 3B dd5x r16 e5 | 540 | 119 | 153 | 268 | 22.04 | 28.33 |
| 3B dd5x r16 e2 vs 3B dd5x r32 e2 | 540 | 114 | 152 | 274 | 21.11 | 28.15 |
| 3B dd5x r32 e2 vs 3B dd5x r16 e5 | 540 | 158 | 117 | 265 | 29.26 | 21.67 |
| 3B dd5x r32 e2 vs 3B dd5x r32 e5 | 540 | 141 | 160 | 239 | 26.11 | 29.63 |
| 3B non-aug r16 e5 vs 3B dd5x r16 e5 | 540 | 176 | 154 | 210 | 32.59 | 28.52 |
| 3B non-aug r16 e5 vs 3B non-aug r32 e5 | 540 | 146 | 103 | 291 | 27.04 | 19.07 |
| 3B non-aug r32 e5 vs 3B dd5x r32 e5 | 540 | 136 | 198 | 206 | 25.19 | 36.67 |
| 3B non-aug r16 e2 vs 3B non-aug r16 e5 | 540 | 97 | 180 | 263 | 17.96 | 33.33 |
| 3B non-aug r32 e2 vs 3B non-aug r32 e5 | 540 | 104 | 145 | 291 | 19.26 | 26.85 |

## Phase 3: 49B Comparator

| Pair | Rows | 49B Wins | 3B Wins | Ties | 49B Win % | 3B Win % |
| --- | --- | --- | --- | --- | --- | --- |
| 49B base vs 3B base | 540 | 398 | 56 | 86 | 73.70 | 10.37 |
| 49B base vs 3B dd5x r16 e5 | 540 | 409 | 75 | 56 | 75.74 | 13.89 |
| 49B base vs 3B dd5x r32 e5 | 540 | 389 | 89 | 62 | 72.04 | 16.48 |
| 49B base vs 3B dd5x r16 e2 | 540 | 405 | 68 | 67 | 75.00 | 12.59 |
| 49B base vs 3B dd5x r32 e2 | 540 | 380 | 82 | 78 | 70.37 | 15.19 |
| 49B base vs 3B non-aug r16 e5 | 540 | 390 | 80 | 70 | 72.22 | 14.81 |
| 49B base vs 3B non-aug r32 e5 | 540 | 404 | 65 | 71 | 74.81 | 12.04 |

## Pairwise Scoreboard

| Model | Pairs | Pair Wins | Pair Losses | Pair Ties | Net Wins |
| --- | --- | --- | --- | --- | --- |
| 3B non-aug r16 e5 | 5 | 4 | 1 | 0 | 3 |
| 3B dd5x r32 e5 | 5 | 4 | 1 | 0 | 3 |
| 3B dd5x r32 e2 | 5 | 3 | 2 | 0 | 1 |
| 3B base | 7 | 3 | 4 | 0 | -1 |
| 3B non-aug r32 e2 | 1 | 0 | 1 | 0 | -1 |
| 3B non-aug r16 e2 | 1 | 0 | 1 | 0 | -1 |
| 3B non-aug r32 e5 | 5 | 1 | 4 | 0 | -3 |
| 3B dd5x r16 e5 | 6 | 1 | 5 | 0 | -4 |
| 3B dd5x r16 e2 | 4 | 0 | 4 | 0 | -4 |

## Artifacts

- Aggregate JSON: `/mnt/nvme2/peft/evals/aggregates/3b-nim-claude-sonnet-20260608/aggregate.json`
- Source roots: `/mnt/nvme2/peft/evals/singleaxis-claude-sonnet`, `/mnt/nvme2/peft/evals/pairwise-claude-sonnet`
