# Logical Entailment Micro 550B Hypothesis Result

Created: 2026-07-09T20:05:31.505900Z

Scope: Stage 1A logical-entailment generation on the deterministic 8-document-per-corpus micro slice. This result does not include Customizer training or validation loss yet.

## Evidence

- NIM Super generated 46 KVP rows from 17 entailments across 5 contributing passages.
- NIM Ultra generated 90 KVP rows from 36 entailments across 5 contributing passages.
- NeMo Microservices Super generated 30 KVP rows from 11 entailments across 3 contributing passages.
- NeMo Microservices Ultra generated 103 KVP rows from 37 entailments across 5 contributing passages.
- Deterministic answer/context token overlap was slightly lower for Ultra: -1.40 percentage points for NIM and -2.37 percentage points for NeMo Microservices.
- Claude Sonnet 4.6 judged 16 independent LE QA examples through `https://inference-api.nvidia.com/v1` using the Kubernetes secret for credentials.
- Judge overall score was tied by corpus: NIM Super 4.50 vs Ultra 4.50; NeMo Super 4.75 vs Ultra 4.75.
- Judge groundedness and correctness were tied at 5.00 for both Super and Ultra in both corpora.
- Ultra had one borderline row in each corpus and slightly lower leakage/usefulness scores, but no judged failures.

## Decision

Proceed to a small Customizer r16 proxy training edge for the LE Ultra micro outputs if the next step is to test validation-loss impact. Unlike the Curator micro test, this LE micro test shows materially higher Ultra row yield without an aggregate judge-quality regression.

Do not interpret this as approval for full-corpus Ultra regeneration yet. The next evidence edge should be r16 proxy training or a somewhat larger LE judge sample if training resources are not immediately available.
