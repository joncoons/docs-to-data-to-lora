# 3B NIM Claude Sonnet Comparison Matrix - 2026-06-08

Scope: `nim_curated`, `llama-3.2-3b` base and LoRA variants, judged by `azure/anthropic/claude-sonnet-4-6`. Pairwise summaries are repaired and have zero unresolved failures.

Cell format: `Outcome row_wins-column_wins-ties (row win %)`, from the row model perspective. Blank cells were not run. `W` means the row model won more rows than the column model; `L` means the row model lost more rows than it won.

## Scorecard

| Model | Single Composite | Accuracy | Completeness | Faithfulness | Pair Net | vs Base | vs 49B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3B base | 3.738 | 3.761 | 3.070 | 3.954 | -1 | — | L 56-398-86 (10.4%) |
| dd5x r16 e2 | 3.743 | 3.833 | 2.854 | 4.046 | -4 | L 166-200-174 (30.7%) | L 68-405-67 (12.6%) |
| dd5x r32 e2 | 3.809 | 3.907 | 2.969 | 4.089 | 1 | W 203-183-154 (37.6%) | L 82-380-78 (15.2%) |
| dd5x r16 e5 | 3.753 | 3.865 | 2.878 | 4.026 | -4 | L 180-203-157 (33.3%) | L 75-409-56 (13.9%) |
| dd5x r32 e5 | 3.781 | 3.843 | 3.013 | 4.033 | 3 | W 204-184-152 (37.8%) | L 89-389-62 (16.5%) |
| non-aug r16 e2 |  |  |  |  | -1 |  |  |
| non-aug r32 e2 |  |  |  |  | -1 |  |  |
| non-aug r16 e5 | 3.766 | 3.820 | 2.970 | 4.011 | 3 | W 184-179-177 (34.1%) | L 80-390-70 (14.8%) |
| non-aug r32 e5 | 3.763 | 3.850 | 2.859 | 4.072 | -3 | L 168-185-187 (31.1%) | L 65-404-71 (12.0%) |
| 49B base |  |  |  |  | 0 | W 398-56-86 (73.7%) | — |

## Outcome Matrix

Compact outcome only, from the row model perspective.

| row \ col | base | dd5x-r16-e2 | dd5x-r32-e2 | dd5x-r16-e5 | dd5x-r32-e5 | nonaug-r16-e2 | nonaug-r32-e2 | nonaug-r16-e5 | nonaug-r32-e5 | 49b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | — | W | L | W | L |  |  | L | W | L |
| dd5x-r16-e2 | L | — | L | L |  |  |  |  |  | L |
| dd5x-r32-e2 | W | W | — | W | L |  |  |  |  | L |
| dd5x-r16-e5 | L | W | L | — | L |  |  | L |  | L |
| dd5x-r32-e5 | W |  | W | W | — |  |  |  | W | L |
| nonaug-r16-e2 |  |  |  |  |  | — |  | L |  |  |
| nonaug-r32-e2 |  |  |  |  |  |  | — |  | L |  |
| nonaug-r16-e5 | W |  |  | W |  | W |  | — | W | L |
| nonaug-r32-e5 | L |  |  |  | L |  | W | L | — | L |
| 49b | W | W | W | W | W |  |  | W | W | — |

## Detailed Matrix

| row \ col | base | dd5x-r16-e2 | dd5x-r32-e2 | dd5x-r16-e5 | dd5x-r32-e5 | nonaug-r16-e2 | nonaug-r32-e2 | nonaug-r16-e5 | nonaug-r32-e5 | 49b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | — | W 200-166-174 (37.0%) | L 183-203-154 (33.9%) | W 203-180-157 (37.6%) | L 184-204-152 (34.1%) |  |  | L 179-184-177 (33.1%) | W 185-168-187 (34.3%) | L 56-398-86 (10.4%) |
| dd5x-r16-e2 | L 166-200-174 (30.7%) | — | L 114-152-274 (21.1%) | L 119-153-268 (22.0%) |  |  |  |  |  | L 68-405-67 (12.6%) |
| dd5x-r32-e2 | W 203-183-154 (37.6%) | W 152-114-274 (28.1%) | — | W 158-117-265 (29.3%) | L 141-160-239 (26.1%) |  |  |  |  | L 82-380-78 (15.2%) |
| dd5x-r16-e5 | L 180-203-157 (33.3%) | W 153-119-268 (28.3%) | L 117-158-265 (21.7%) | — | L 131-183-226 (24.3%) |  |  | L 154-176-210 (28.5%) |  | L 75-409-56 (13.9%) |
| dd5x-r32-e5 | W 204-184-152 (37.8%) |  | W 160-141-239 (29.6%) | W 183-131-226 (33.9%) | — |  |  |  | W 198-136-206 (36.7%) | L 89-389-62 (16.5%) |
| nonaug-r16-e2 |  |  |  |  |  | — |  | L 97-180-263 (18.0%) |  |  |
| nonaug-r32-e2 |  |  |  |  |  |  | — |  | L 104-145-291 (19.3%) |  |
| nonaug-r16-e5 | W 184-179-177 (34.1%) |  |  | W 176-154-210 (32.6%) |  | W 180-97-263 (33.3%) |  | — | W 146-103-291 (27.0%) | L 80-390-70 (14.8%) |
| nonaug-r32-e5 | L 168-185-187 (31.1%) |  |  |  | L 136-198-206 (25.2%) |  | W 145-104-291 (26.9%) | L 103-146-291 (19.1%) | — | L 65-404-71 (12.0%) |
| 49b | W 398-56-86 (73.7%) | W 405-68-67 (75.0%) | W 380-82-78 (70.4%) | W 409-75-56 (75.7%) | W 389-89-62 (72.0%) |  |  | W 390-80-70 (72.2%) | W 404-65-71 (74.8%) | — |

## Artifacts

- JSON: `/mnt/nvme2/peft/evals/aggregates/3b-nim-claude-sonnet-20260608/comparison_matrix.json`
- Win-rate CSV: `/mnt/nvme2/peft/evals/aggregates/3b-nim-claude-sonnet-20260608/comparison_matrix_win_rates.csv`
- Detail CSV: `/mnt/nvme2/peft/evals/aggregates/3b-nim-claude-sonnet-20260608/comparison_matrix_details.csv`
- Source aggregate: `/mnt/nvme2/peft/evals/aggregates/3b-nim-claude-sonnet-20260608/aggregate.json`
