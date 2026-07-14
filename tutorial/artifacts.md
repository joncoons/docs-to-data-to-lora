# Artifact Map

This page maps tutorial stages to the files they produce in a full run.

## Stage 1 - Curated Crawl

| Artifact | Purpose |
|---|---|
| `inventory.csv` | Sitemap-derived URL inventory with product, version, page, depth, and lastmod columns |
| crawler collection | Chunked HTML, markdown, text, and parsed binary content in the vector store |
| binary manifest | Linked PDF, DOCX, and PPTX assets queued for document parsing |

The tutorial notebook builds a small inventory in memory. A live crawler run should persist the collection in Elasticsearch or an equivalent vector store.

## Stage 2 - Dataset Creation

Default full-run location:

```text
/mnt/nvme2/peft/datasets/v2/<collection>/
```

Key files:

| Artifact | Producer | Purpose |
|---|---|---|
| `passages.jsonl` | Stage 0 | grouped and filtered source passages |
| `stage1a_le.jsonl` | Stage 1A | entailment-derived QA pairs |
| `stage1b_synthesis.jsonl` | Stage 1B | semantic-neighborhood synthesis pairs |
| `stage1c_instruction.jsonl` | Stage 1C | summary, listicle, and procedural examples |
| `stage1_5_gapfill.jsonl` | Stage 1.5 | synthetic gap-fill rows, when generated directly |
| `bias_report.json` | Stage 1.5 | product-family coverage report |
| `stage2_eval.jsonl` | Stage 2 | refined or accepted rows after self-eval |
| `training.jsonl` | Stage 3 curator | Customizer training split |
| `validation.jsonl` | Stage 3 curator | Customizer validation split |
| `stage4_validation.json` | Stage 4 | external grounding gate result |
| `manifests/dataset_version_manifest.json` | finalization | dataset lineage and version metadata |

The tutorial writes local smoke artifacts under `tutorial/_outputs/`.

## Stage 3 - LoRA Training

Customizer jobs use registered dataset entity names such as:

```text
default/stage3-nim-curated
default/stage3-nemo-usvcs-curated
```

The adapter output is a NeMo Entity Store model ref, usually:

```text
default/lora-<corpus>-<base>-r<rank>
```

The dry-run notebook prints the Customizer payload without submitting it.

## Evaluation

Completion collection writes durable model outputs to:

```text
/mnt/nvme2/peft/evals/completions/<dataset>/<base>/<target>/<rank>/<run_id>/
```

Each run directory should contain:

| Artifact | Purpose |
|---|---|
| `responses.jsonl` | one saved target completion per test row |
| `errors.jsonl` | rows that failed target generation |
| `manifest.json` | target, dataset, and run metadata |
| `token_summary.json` | prompt, completion, reasoning, and total token counts |

Direct judge evals write `summary.json` files with separate `target_generation` and `judge_scoring` sections. Keep those separate in reports so judge cost is not attributed to target serving cost.
