# LE Micro Claude Judge

Created: 2026-07-09T20:04:57.915209Z

Judge model: `azure/anthropic/claude-sonnet-4-6` via `https://llm.example.com/v1`.

Rows judged: 16 independent LE QA examples.

## Average Scores

| Cell | Rows | Pass | Borderline | Fail | Overall | Groundedness | Correctness | Specificity | Usefulness | Leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nemo_usvcs_curated/super | 4 | 4 | 0 | 0 | 4.75 | 5.0 | 5.0 | 4.75 | 4.75 | 5.0 |
| nemo_usvcs_curated/ultra | 4 | 3 | 1 | 0 | 4.75 | 5.0 | 5.0 | 4.75 | 4.75 | 4.5 |
| nim_curated/super | 4 | 4 | 0 | 0 | 4.5 | 5.0 | 5.0 | 4.5 | 4.5 | 5.0 |
| nim_curated/ultra | 4 | 3 | 1 | 0 | 4.5 | 5.0 | 5.0 | 4.5 | 4.25 | 4.75 |

## Ultra Minus Super Deltas

| Corpus | Overall | Groundedness | Correctness | Specificity | Usefulness | Leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nemo_usvcs_curated | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -0.5 |
| nim_curated | 0.0 | 0.0 | 0.0 | 0.0 | -0.25 | -0.25 |

## Interpretation

Claude did not find an aggregate quality regression for Ultra on this sample, so the larger Ultra row yield is worth a small Customizer r16 proxy training edge.
