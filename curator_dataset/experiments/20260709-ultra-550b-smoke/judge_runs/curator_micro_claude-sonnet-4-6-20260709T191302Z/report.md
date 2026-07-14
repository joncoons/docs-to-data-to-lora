# Curator Micro Claude Judge

Created: 2026-07-09T19:14:10.902333Z

Judge model: `azure/anthropic/claude-sonnet-4-6` via `https://llm.example.com/v1`.

Rows judged: 8 paired Super-vs-Ultra examples with position swap=True.

## Wins

| Side | Wins |
| --- | ---: |
| super | 6 |
| ultra | 1 |
| tie | 1 |

## Wins By Corpus

| Corpus | Super | Ultra | Tie |
| --- | ---: | ---: | ---: |
| nemo_usvcs_curated | 3 | 1 | 0 |
| nim_curated | 3 | 0 | 1 |

## Average Scores

| Metric | Super | Ultra | Delta |
| --- | ---: | ---: | ---: |
| overall | 3.5 | 2.938 | -0.562 |
| groundedness | 4.812 | 4.75 | -0.062 |
| specificity | 3.438 | 3.125 | -0.313 |
| usefulness | 3.438 | 2.688 | -0.75 |

## Interpretation

Claude preferred Super on this small paired sample, so Ultra does not currently justify proxy LoRA training without a larger or more targeted sample.
