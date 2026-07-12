# Golden Evaluation Plan - LE vs Curator vs Llama 3.3 70B

Date captured: 2026-07-12

## Purpose

This directory freezes the evaluation dataset and target plan for comparing the current Logical Entailment (LE) and Curator DiverseQA LoRA adapters, then comparing the best LoRA winner against a dense Llama 3.3 70B reference target.

Llama 3.3 70B is a comparison target, not the judge. The judge remains independent.

## Golden Dataset

`golden-v1` is built from the prior held-out test sets, not from the current validation splits. The builder removes rows that exactly overlap current LE or Curator train/validation prompts and enforces HTML-only provenance.

### Construction Methodology

1. Start from the prior deterministic held-out test sets under `/mnt/nvme2/peft/datasets/experiments/*_dd_deterministic_5x_combined_20260605/`.
2. Pair each plain QA row with its context-baked companion row by source row index; mismatched row counts fail the build.
3. Build a leakage guard from all current LE and Curator train/validation prompts, normalized by lowercasing and whitespace collapse.
4. Exclude any candidate whose normalized prompt exactly matches the current train/validation prompt set.
5. Enforce HTML provenance by retaining only rows whose context includes `https://docs.nvidia.com/`; rows with PDF references, no relevant context, or non-docs context are excluded.
6. Assign each retained row a deterministic `golden_id` from corpus, source row index, prompt, and completion.
7. Emit both question-only and context-baked variants, plus `excluded_rows.jsonl`, per-corpus manifests, and a top-level checksum file.

Primary model-answer collection uses `golden-v1/*/test_set.jsonl` without RAG or injected context. This isolates model and adapter behavior. `test_set_with_context.jsonl` is retained as a diagnostic artifact for context-baked follow-up runs, not as the primary LE-vs-Curator comparison input.

Rows retained:

| Corpus | Source rows | Kept | Excluded | Exclusion notes |
| --- | ---: | ---: | ---: | --- |
| NIM | 540 | 424 | 116 | 3 train/val overlaps, 72 PDF contexts, 30 no-context rows, 11 non-docs contexts |
| NeMo Microservices | 462 | 431 | 31 | 5 train/val overlaps, 23 no-context rows, 3 non-docs contexts |

Key artifacts:

- `golden-v1/manifest.json` records source files, output hashes, and the current train/validation prompt-set fingerprint.
- `golden-v1/checksums.sha256` records checksums for all frozen files.
- `golden-v1/*/test_set.jsonl` is the primary no-RAG, question-only evaluation input.
- `golden-v1/*/test_set_with_context.jsonl` is the context-baked diagnostic input.
- `golden-v1/*/excluded_rows.jsonl` records excluded source row indexes and reasons.

Rebuild command:

```bash
/home/joncoons/anaconda3/bin/python scripts/eval/build_golden_testset.py
```

Treat rebuilt output as a new artifact version unless the checksums are unchanged.

## Evaluation Matrix

Single-axis evaluation measures standalone model efficacy on the frozen context-baked rows. Use the direct Kimi single-axis scorer over saved completion files, with axes already implemented as accuracy, completeness, faithfulness, and clarity.

Pairwise evaluation has two layers:

1. Matched LE vs Curator comparisons by corpus, base size, and LoRA rank.
2. The best LoRA winner from that context against the dense Llama 3.3 70B reference target.

Use position-swapped pairwise judging for all pairwise runs. Select the winner primarily by pairwise win rate; use single-axis composite as a tie breaker.

## Judge And Reference Models

Primary judge:

- Endpoint: `https://inference-api.nvidia.com/v1`
- Model: `azure/moonshotai/kimi-k2.6`
- Availability: verified with HTTP 200 on 2026-07-12 using `runai-rag/nvidia-inference-key`.

Fallback judge:

- Claude Sonnet 4.6 through NeMo Evaluator if Kimi K2.6 is unavailable or unstable.
- Preferred NVIDIA-hosted model ID when available: `azure/anthropic/claude-sonnet-4-6`.

Dense reference target:

- Endpoint: `https://inference-api.nvidia.com/v1`
- Model: `nvidia/meta/llama-3.3-70b-instruct`
- Alternate endpoint/model pair when using the integrate endpoint: `https://integrate.api.nvidia.com/v1`, `meta/llama-3.3-70b-instruct`.

Do not persist API key values. Load them from Kubernetes Secrets or process environment only.

## Completion Collection

Primary completions should be collected from `test_set.jsonl` without RAG or injected context. This is the evaluation mode used for the 2026-07-12 LoRA and local 70B answer sets.

Hosted or local 70B completions can be collected directly with the updated collector:

```bash
export NVIDIA_API_KEY="$(kubectl get secret -n runai-rag nvidia-inference-key -o jsonpath='{.data.api-key}' | base64 -d)"
/home/joncoons/anaconda3/bin/python scripts/eval/collect_completions.py \
  --dataset curator_dataset/experiments/20260709-curator-vs-le/golden_eval/golden-v1/nim_curated/test_set.jsonl \
  --dataset-slug nim_curated_golden_v1_question_only \
  --target-api-url https://inference-api.nvidia.com/v1 \
  --target-api-key-env NVIDIA_API_KEY \
  --model nvidia/meta/llama-3.3-70b-instruct \
  --run-id golden-v1-70b-20260712 \
  --max-tokens 4096 \
  --temperature 0.0001 \
  --concurrency 1 \
  --max-attempts 3 \
  --retry-backoff-s 10
```

Repeat for `nemo_usvcs_curated/test_set.jsonl`.

Local LoRA completions require serving the matching base NIM and syncing the target adapters into that NIM's `NIM_PEFT_SOURCE`. The 2026-07-12 answer collection used direct ClusterIP calls to the NIMServices, not `rag-oai-proxy`, so route staleness in the proxy cannot affect the saved answer sets.

## Kimi Scoring

Single-axis scoring over saved completions:

```bash
/home/joncoons/anaconda3/bin/python scripts/eval/run_direct_kimi_singleaxis.py \
  --responses /mnt/nvme2/peft/evals/completions/<dataset>/<base>/<target>/<rank>/<run>/responses.jsonl \
  --judge-api-url https://inference-api.nvidia.com/v1 \
  --judge-model azure/moonshotai/kimi-k2.6 \
  --judge-api-key-env NVIDIA_API_KEY \
  --eval-run-id golden-v1-kimi-20260712 \
  --resume \
  --concurrency 1
```

Pairwise scoring over saved completions:

```bash
/home/joncoons/anaconda3/bin/python scripts/eval/run_direct_kimi_pairwise.py \
  --pair le-r32 /path/to/le/responses.jsonl curator-r32 /path/to/curator/responses.jsonl \
  --judge-api-url https://inference-api.nvidia.com/v1 \
  --judge-model azure/moonshotai/kimi-k2.6 \
  --judge-api-key-env NVIDIA_API_KEY \
  --eval-run-id golden-v1-kimi-pairwise-20260712 \
  --position-swap \
  --resume \
  --concurrency 1
```


## Hosted Smoke Result

A constrained hosted smoke was run on 2026-07-12 with two rows from each corpus to verify endpoint wiring. This is not a formal winner evaluation.

| Dataset slug | Target | Rows completed | Kimi rows scored | Mean accuracy | Mean completeness | Mean faithfulness | Mean clarity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nim_curated_golden_v1` | `nvidia/meta/llama-3.3-70b-instruct` | 2 | 2 | 5.0 | 4.0 | 5.0 | 5.0 |
| `nemo_usvcs_curated_golden_v1` | `nvidia/meta/llama-3.3-70b-instruct` | 2 | 2 | 3.5 | 3.5 | 3.5 | 4.0 |

The smoke summary is captured in `hosted_70b_kimi_smoke_20260712.json`; raw completion and score artifacts are under `/mnt/nvme2/peft/evals/`. An initial 1024-token Kimi judge budget was superseded because Kimi could spend the entire budget on reasoning text and hit `finish_reason=length` before returning parseable JSON. Keep the default 8192-token judge budget for Kimi unless a later prompt or endpoint setting reliably forces compact JSON.

## Current Status

The golden dataset, target catalog, hosted 70B reference model ID, Kimi judge ID, LoRA answer-set materialization, and no-RAG completion outputs are captured. Next, run Kimi single-axis and position-swapped pairwise scoring from the saved `responses.jsonl` files.
