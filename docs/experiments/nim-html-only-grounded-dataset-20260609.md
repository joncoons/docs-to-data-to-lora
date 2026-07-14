# NIM HTML-Only Grounded Dataset

Date: 2026-06-09

Purpose: remove PDF-derived NIM rows from the grounded dataset to test whether the NIM LoRA adaptation gap is caused by source heterogeneity rather than model capacity alone.

## Dataset

- Source dataset: `<DATASET_ROOT>/nim_curated`
- New dataset: `<DATASET_ROOT>/nim_curated_html_only`
- Dataset version ID: `dsv_5694e359929202aff451042d`
- Filter: `doc_kind == "html"`
- Builder: `scripts/pipeline/build_html_only_grounded_dataset.py`
- Context baker: `scripts/eval/bake_context_into_testset_filtered.py`

The rebuild uses `stage2_eval.jsonl` and `passages.jsonl` lineage, not the simplified `training.jsonl`, so each QA KVP remains tied to its original `passage_id` and `source_url`.

## Source Filter

| Metric | Original NIM | HTML-only NIM |
|---|---:|---:|
| Passages | 498 | 431 |
| Stage 2 QA rows | 5,752 | 5,126 |
| PDF QA rows removed | 626 | 0 |
| HTML QA rows retained | 5,126 | 5,126 |

The removed PDF-derived rows were all from `product_family=unknown`. The HTML-only dataset still has a small `unknown` bucket from HTML sources.

## Curator Output

| Step | Rows |
|---|---:|
| HTML Stage 2 input | 5,126 |
| After exact dedup | 5,071 |
| After MinHash dedup | 5,069 |
| After length filter | 4,835 |
| After subset filter | 4,834 |

Stage distribution after filtering:

| Stage | Rows |
|---|---:|
| `1a` | 3,839 |
| `1b` | 741 |
| `1c` | 254 |

Final split:

| Split | Rows |
|---|---:|
| `training.jsonl` | 4,349 |
| `validation.jsonl` | 485 |
| `adapter_train.jsonl` | 4,136 |
| `adapter_val.jsonl` | 216 |
| `test_set.jsonl` | 482 |

## Context Bake

The context bake reuses ES index `nim_curated` but filters selected context chunks to the HTML-only passage URL allow-list.

| Metric | Value |
|---|---:|
| Test rows | 482 |
| Context URL headers | 1,871 |
| Unique context URLs | 373 |
| Bad context URLs outside allow-list | 0 |
| PDF context URLs | 0 |
| Average chunks kept | 3.8817 |
| Rows with zero chunks | 18 |
| Zero-chunk rate | 3.73% |

`test_set_with_context.jsonl` is included in the refreshed dataset manifest hash set.

## Next Steps

1. Register `nim_curated_html_only` as a new dataset entity for Customizer rather than replacing `nim_curated`.
2. Train a targeted 8B `r32` LoRA first, using the same BF16 Llama 3.1 8B template and the same training settings as prior grounded NIM runs.
3. Compare against the existing NIM grounded 8B `r32` result on the same evaluation phases.
4. If HTML-only materially improves NIM adaptation, consider a second pass that keeps only strongly classified NIM HTML rows and removes the remaining HTML `unknown` bucket.
