# Curator Micro 550B Hypothesis Result

Created: 2026-07-09T19:14:52.030588Z

Scope: Curator DiverseQA only, using the deterministic 8-document-per-corpus micro slice. Logical-entailment Ultra generation and LoRA training are not included in this result.

## Evidence

- Generation completed for NIM and NeMo Microservices with both Super and Ultra.
- Normalization parsed all raw segments with zero parse failures.
- Deterministic metrics were mixed: Ultra did not improve row yield, reduced duplicate answers slightly, improved NeMo short-answer count, but did not improve answer/context overlap.
- Claude Sonnet 4.6 judged 8 paired examples with position swapping through `https://llm.example.com/v1` using the Kubernetes secret for credentials.
- Judge result: Super 6 wins, Ultra 1 win, 1 tie.
- Average judge score delta, Ultra minus Super: overall -0.562, groundedness -0.062, specificity -0.313, usefulness -0.750.

## Decision

Do not proceed to Curator Ultra 550B full-corpus regeneration or Customizer r16 proxy training from this evidence. The current Curator micro smoke points toward Super being the better generator for this slice.

The next useful experiment edge is a separate logical-entailment Ultra micro test on the same `inputs_micro/*.passages.jsonl` files, because this Curator-only result does not answer whether Ultra improves the LE pipeline.
