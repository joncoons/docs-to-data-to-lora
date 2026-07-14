# Stage 3 Curator Reductions

This report captures the Stage 3 row reductions for the LE-derived NIM and NeMo Microservices datasets and records the policy correction made after inspecting the initial reductions.

The initial Stage 3 pass used `tiktoken` `cl100k_base` with `question >= 8` and `answer >= 25`. That was the wrong tokenizer for Llama LoRA training and over-penalized concise technical answers. The revised pass uses the production Llama 3.1 8B tokenizer resolved from `$LOCAL_NIM_CACHE`; this run used `/data/nim-cache/ngc/hub/models--nim--meta--llama-3.1-8b-instruct/snapshots/fp8-tool-calling`, with `question >= 12` and `answer >= 8`.

The train/validation split is shown for traceability but is not treated as a reduction cause.

## Policy Comparison

| Policy | Tokenizer | Min Question Tokens | Min Answer Tokens |
| --- | --- | ---: | ---: |
| Initial | `tiktoken cl100k_base` | 8 | 25 |
| Revised | `/data/nim-cache/ngc/hub/models--nim--meta--llama-3.1-8b-instruct/snapshots/fp8-tool-calling` | 12 | 8 |

## Outcome Summary

| Corpus | Policy | Stage 3 Input | Final Curated Rows | Removed | Removed % | Train | Val | Delta Final Rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NIM | Initial | 9404 | 6481 | 2923 | 31.08% | 5831 | 650 | - |
| NIM | Revised | 9404 | 8785 | 619 | 6.58% | 7905 | 880 | +2304 |
| NeMo Microservices | Initial | 10712 | 5115 | 5597 | 52.25% | 4602 | 513 | - |
| NeMo Microservices | Revised | 10712 | 7970 | 2742 | 25.60% | 7171 | 799 | +2855 |

## Revised Reduction Causes

### NIM

Run directory: `curator_dataset/experiments/20260709-le-uncapped-batch/runs/nim_curated/super-v3`

| Step | Cause | Before | After | Removed | Removed % of Step | Removed % of Input |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exact_question_dedup | Duplicate questions after case/whitespace normalization | 9404 | 9334 | 70 | 0.74% | 0.74% |
| minhash_fuzzy_dedup | Near-duplicate question+answer pairs by MinHash LSH | 9334 | 9098 | 236 | 2.53% | 2.51% |
| length_filter | Question tokens < 12 or answer tokens < 8 | 9098 | 8787 | 311 | 3.42% | 3.31% |
| answer_subset_filter | Answer text normalizes to a substring of the question | 8787 | 8785 | 2 | 0.02% | 0.02% |

Sidecars: `curator_dataset/experiments/20260709-le-uncapped-batch/runs/nim_curated/super-v3/stage3_curator_summary.md` and `curator_dataset/experiments/20260709-le-uncapped-batch/runs/nim_curated/super-v3/stage3_curator_summary.json`

### NeMo Microservices

Run directory: `curator_dataset/experiments/20260709-le-uncapped-batch/runs/nemo_usvcs_curated/super-v3`

| Step | Cause | Before | After | Removed | Removed % of Step | Removed % of Input |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exact_question_dedup | Duplicate questions after case/whitespace normalization | 10712 | 9405 | 1307 | 12.20% | 12.20% |
| minhash_fuzzy_dedup | Near-duplicate question+answer pairs by MinHash LSH | 9405 | 8730 | 675 | 7.18% | 6.30% |
| length_filter | Question tokens < 12 or answer tokens < 8 | 8730 | 7970 | 760 | 8.71% | 7.09% |
| answer_subset_filter | Answer text normalizes to a substring of the question | 7970 | 7970 | 0 | 0.00% | 0.00% |

Sidecars: `curator_dataset/experiments/20260709-le-uncapped-batch/runs/nemo_usvcs_curated/super-v3/stage3_curator_summary.md` and `curator_dataset/experiments/20260709-le-uncapped-batch/runs/nemo_usvcs_curated/super-v3/stage3_curator_summary.json`
