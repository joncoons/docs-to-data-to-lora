# Curator Micro Super vs Ultra Smoke Comparison

Created: 2026-07-09T19:07:47.632060Z

Scope: Curator DiverseQA generation on the deterministic 8-document-per-corpus micro slice. This does not include LE regeneration, Claude judge scoring, or LoRA training yet.

## Result

The deterministic Curator-only smoke does not show a clear Ultra 550B win. Output validity was identical, yield was effectively tied, and grounding proxies were mixed. Ultra reduced duplicate answers in both corpora and reduced very short answers for NeMo Microservices, but its answer/context overlap did not improve and NeMo exact answer containment fell.

## Cell Metrics

| Corpus | Model | Raw segments | QA pairs | Parse failures | Duplicate answers | Short answers <4w | Exact answer in context | Avg answer/context token overlap | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NIM | Super | 31 | 249 | 0 | 21 | 90 | 32.93% | 72.17% | 94.2s |
| NIM | Ultra | 31 | 248 | 0 | 18 | 91 | 33.06% | 69.49% | 113.4s |
| NeMo Microservices | Super | 35 | 279 | 0 | 26 | 98 | 34.05% | 72.22% | 144.5s |
| NeMo Microservices | Ultra | 35 | 279 | 0 | 23 | 87 | 30.47% | 71.74% | 138.5s |

## Pairwise Deltas

| Corpus | QA pair delta | Runtime delta | Duplicate answer delta | Short answer delta | Exact containment delta | Token-overlap delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NIM | -1 (-0.40%) | +19.2s (+20.39%) | -3 | +1 | +0.13% | -2.68% |
| NeMo Microservices | 0 (+0.00%) | -6.0s (-4.18%) | -3 | -11 | -3.58% | -0.48% |

## Interpretation

- Both Super and Ultra parsed cleanly: zero parse failures and zero exact duplicate pair drops in all four cells.
- Ultra does not improve row yield: NIM produced one fewer QA pair and NeMo Microservices tied exactly.
- Ultra improves a few surface quality metrics: duplicate answers dropped by 3 in both corpora, and NeMo short answers dropped by 11.
- Grounding proxies are not improved: NIM token overlap declined by 2.68 percentage points, while NeMo exact answer containment declined by 3.58 percentage points.
- This is too small to reject Ultra, but it is not strong enough to justify full-corpus regeneration or LoRA training without a paired judge sample.

## Next Edge

Run a small paired Claude Sonnet 4.6 judge pass over matched Super/Ultra rows from the same source segments. Only proceed to Customizer r16 proxy training if the judge finds a meaningful Ultra quality gain that the deterministic metrics did not expose.
