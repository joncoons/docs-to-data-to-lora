# Stage 3 Curator Reduction Summary

Stage 3 applies structural curation after Stage 2 QA admission. The reductions below explain row loss before the train/validation split.

## Parameters

- train_ratio: `0.9`
- minhash_threshold: `0.85`
- min_question_tokens: `12`
- min_answer_tokens: `8`
- tokenizer_name_or_path: `/data/nim-cache/ngc/hub/models--nim--meta--llama-3.1-8b-instruct/snapshots/fp8-tool-calling`
- tokenizer_class: `TokenizersBackend`

## Reduction Steps

| Step | Cause | Before | After | Removed | Removed % of Step | Removed % of Input |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exact_question_dedup | Duplicate questions after case/whitespace normalization | 10712 | 9405 | 1307 | 12.20% | 12.20% |
| minhash_fuzzy_dedup | Near-duplicate question+answer pairs by MinHash LSH | 9405 | 8730 | 675 | 7.18% | 6.30% |
| length_filter | Question tokens < 12 or answer tokens < 8 | 8730 | 7970 | 760 | 8.71% | 7.09% |
| answer_subset_filter | Answer text normalizes to a substring of the question | 7970 | 7970 | 0 | 0.00% | 0.00% |

## Final Counts

- Stage 3 input rows: `10712`
- Final curated rows before split: `7970`
- Total removed before split: `2742` (`25.60%` of input)
- Retained before split: `74.40%`
- Training rows: `7171`
- Validation rows: `799`

The train/validation split is recorded for completeness but is not a row-loss cause.
