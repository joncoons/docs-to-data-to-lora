# Stage 3 Curator Reduction Summary

Stage 3 applies structural curation after Stage 2 QA admission. The reductions below explain row loss before the train/validation split.

## Parameters

- train_ratio: `0.9`
- minhash_threshold: `0.85`
- min_question_tokens: `12`
- min_answer_tokens: `8`
- tokenizer_name_or_path: `/mnt/nvme4/nim_cache/nim/ngc/hub/models--nim--meta--llama-3.1-8b-instruct/snapshots/fp8-tool-calling`
- tokenizer_class: `TokenizersBackend`

## Reduction Steps

| Step | Cause | Before | After | Removed | Removed % of Step | Removed % of Input |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exact_question_dedup | Duplicate questions after case/whitespace normalization | 9404 | 9334 | 70 | 0.74% | 0.74% |
| minhash_fuzzy_dedup | Near-duplicate question+answer pairs by MinHash LSH | 9334 | 9098 | 236 | 2.53% | 2.51% |
| length_filter | Question tokens < 12 or answer tokens < 8 | 9098 | 8787 | 311 | 3.42% | 3.31% |
| answer_subset_filter | Answer text normalizes to a substring of the question | 8787 | 8785 | 2 | 0.02% | 0.02% |

## Final Counts

- Stage 3 input rows: `9404`
- Final curated rows before split: `8785`
- Total removed before split: `619` (`6.58%` of input)
- Retained before split: `93.42%`
- Training rows: `7905`
- Validation rows: `880`

The train/validation split is recorded for completeness but is not a row-loss cause.
