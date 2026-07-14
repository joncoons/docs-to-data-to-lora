# Logical Entailment Micro Super vs Ultra Smoke Comparison

Created: 2026-07-09T20:01:37.908232Z

Scope: Stage 1A logical-entailment generation on the deterministic 8-document-per-corpus micro slice. This does not include Claude judge scoring or LoRA training yet.

## Result

Ultra 550B strongly increases LE row yield on the micro slice. NIM output nearly doubles, and NeMo Microservices output more than triples. Deterministic grounding proxies are broadly similar or better for Ultra, although higher yield needs judge review because the extra rows may include lower-value examples.

## Cell Metrics

| Corpus | Model | Rows | Entailments | Contributing passages | Duplicate answers | Short answers <4w | Exact answer in context | Avg answer/context token overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NIM | Super | 46 | 17 | 5/8 | 1 | 0 | 0.00% | 71.22% |
| NIM | Ultra | 90 | 36 | 5/8 | 1 | 0 | 0.00% | 69.82% |
| NeMo Microservices | Super | 30 | 11 | 3/8 | 0 | 0 | 0.00% | 71.79% |
| NeMo Microservices | Ultra | 103 | 37 | 5/8 | 0 | 0 | 0.00% | 69.42% |

## Pairwise Deltas

| Corpus | Row delta | Entailment delta | Passage delta | Duplicate answer delta | Short answer delta | Exact containment delta | Token-overlap delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NIM | 44 (+95.65%) | 19 (+111.76%) | +0 | +0 | +0 | +0.00% | -1.40% |
| NeMo Microservices | 73 (+243.33%) | 26 (+236.36%) | +2 | +0 | +0 | +0.00% | -2.37% |

## Interpretation

- Ultra generated 90 NIM rows versus 46 for Super, with 36 entailments versus 17.
- Ultra generated 103 NeMo Microservices rows versus 30 for Super, with 37 entailments versus 11.
- Passage coverage improved in both corpora: NIM stayed at 5 contributing passages, while NeMo increased from 3 to 5.
- Ultra has more duplicate answers in absolute terms because it emits many more rows; duplicate rate should be judged alongside usefulness, not by raw count alone.
- Deterministic metrics are strong enough to justify a Claude single-axis quality sample before deciding whether to run Customizer r16 proxy training.
