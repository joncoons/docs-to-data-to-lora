# 8B NIM HTML-Only LoRA Training

Date: 2026-06-10

Purpose: retrain NIM Llama 3.1 8B LoRA adapters on the non-augmented HTML-only grounded dataset to measure whether removing PDF-derived rows improves NIM adaptation.

## Dataset

- Training dataset entity: `default/stage3-nim-curated-html-only`
- Context-baked evaluation dataset entity: `default/stage3-nim-curated-html-only-test-with-context`
- Local dataset path: `/mnt/nvme2/peft/datasets/v2/nim_curated_html_only`
- Dataset version ID: `dsv_5694e359929202aff451042d`
- Training rows: `4,349`
- Validation rows: `485`
- Test rows: `482`
- Context filtering: HTML URL allow-list, zero PDF context URLs.

## Training Configuration

- Base model: `meta/llama-3.1-8b-instruct`
- Customizer template: `meta/llama-3.1-8b-instruct@v1.0.0+80GB`
- Target precision: `bf16-mixed`
- Epochs: `5`
- Batch size: `16`
- Learning rate: `1e-4`
- Warmup steps: `30`
- Optimizer: `adamw_with_cosine_annealing`
- Sequence packing: disabled
- Placement: Blackwell node `ubuntu-local-dev`

## Jobs

| Rank | Job ID | Output model |
|---:|---|---|
| 16 | `cust-YMKPmhoy1wJUbSfT9vfebi` | `default/lora-nim-html-only-e5-llama-3.1-8b-r16@cust-YMKPmhoy1wJUbSfT9vfebi` |
| 32 | `cust-5WYibbNE6EVdrwTXLcncF5` | `default/lora-nim-html-only-e5-llama-3.1-8b-r32@cust-5WYibbNE6EVdrwTXLcncF5` |

Final Customizer status: both jobs completed and both output model entity-handlers completed.

Observed checkpoint behavior:

| Rank | First validation | Best observed validation | Final validation | Notes |
|---:|---:|---:|---:|---|
| 16 | `1.33638` | `1.20846` | `1.418` | Final checkpoint was not top-1. |
| 32 | `1.27332` | `1.15298` | `1.609` | Final checkpoint was not top-1. |

Training completed at 5 epochs / 1,360 steps for both ranks. The late validation
loss increase suggests the 5-epoch terminal weights overfit this dataset, but
Customizer checkpointing retained a top-1 validation checkpoint for export.

## Adapter Sync

- Synced PEFT source: `/mnt/nvme4/nim_cache/nim/lora-adapters-8b-html-only-flat`
- Adapter IDs:
  - `lora-nim-html-only-e5-llama-3.1-8b-r16`
  - `lora-nim-html-only-e5-llama-3.1-8b-r32`
- Live 8B NIMService was temporarily pointed at
  `/model-store/lora-adapters-8b-html-only-flat` for completion collection.
- Smoke test passed for base, `r16`, and `r32`.

## Completion Collection

- Kubernetes job:
  `deploy/evaluation-matrix/collect-completions-8b-nim-html-only-20260610t032644z.yaml`
- Run ID: `20260610t032644z`
- Dataset slug: `nim_curated_html_only`
- Target API: direct 8B LoRA NIM service,
  `http://nim-llm-8b-bw-lora.runai-rag:8000`
- Rows: `482` for each target
- Row errors: `0`

| Target | Responses path | Total tokens |
|---|---|---:|
| 8B base | `/mnt/nvme2/peft/evals/completions/nim_curated_html_only/llama-3.1-8b/base/base/20260610t032644z/responses.jsonl` | `1,226,169` |
| 8B HTML-only `r16` | `/mnt/nvme2/peft/evals/completions/nim_curated_html_only/llama-3.1-8b/lora-nim-html-only-e5/r16/20260610t032644z/responses.jsonl` | `1,189,421` |
| 8B HTML-only `r32` | `/mnt/nvme2/peft/evals/completions/nim_curated_html_only/llama-3.1-8b/lora-nim-html-only-e5/r32/20260610t032644z/responses.jsonl` | `1,193,038` |

## Planned Evaluation

Active / queued evaluation:

1. Nemotron Ultra Phase 1 + Phase 2:
   `deploy/evaluation-matrix/direct-nemotron-ultra-8b-nim-html-only-20260610t032644z.yaml`
2. Matching 49B HTML-only completion prerequisite:
   `deploy/evaluation-matrix/collect-completions-49b-nim-html-only-20260610t032644z.yaml`
3. Phase 3 should compare the matching 49B response set against 8B base, `r16`,
   and `r32` after the 49B completion job finishes.
