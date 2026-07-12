# Golden Evaluation Plan - LE vs Curator vs Llama 3.3 70B

Date captured: 2026-07-12

## Purpose

This directory freezes the evaluation dataset and target plan for comparing the current Logical Entailment (LE) and Curator DiverseQA LoRA adapters, then comparing the best LoRA winner against a dense Llama 3.3 70B reference target.

Llama 3.3 70B is a comparison target, not the judge. The judge remains independent.

## Golden Dataset

`golden-v1` is built from the prior held-out test sets, not from the current validation splits. The builder removes rows that exactly overlap current LE or Curator train/validation prompts and enforces HTML-only provenance.

Rows retained:

| Corpus | Source rows | Kept | Excluded | Exclusion notes |
| --- | ---: | ---: | ---: | --- |
| NIM | 540 | 424 | 116 | 3 train/val overlaps, 72 PDF contexts, 30 no-context rows, 11 non-docs contexts |
| NeMo Microservices | 462 | 431 | 31 | 5 train/val overlaps, 23 no-context rows, 3 non-docs contexts |

Key artifacts:

- `golden-v1/manifest.json` records source files, output hashes, and the current train/validation prompt-set fingerprint.
- `golden-v1/checksums.sha256` records checksums for all frozen files.
- `golden-v1/*/test_set_with_context.jsonl` is the context-baked evaluation input for all model completions.
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

Hosted 70B completions can be collected directly with the updated collector:

```bash
export NVIDIA_API_KEY="$(kubectl get secret -n runai-rag nvidia-inference-key -o jsonpath='{.data.api-key}' | base64 -d)"
/home/joncoons/anaconda3/bin/python scripts/eval/collect_completions.py \
  --dataset curator_dataset/experiments/20260709-curator-vs-le/golden_eval/golden-v1/nim_curated/test_set_with_context.jsonl \
  --dataset-slug nim_curated_golden_v1 \
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

Repeat for `nemo_usvcs_curated/test_set_with_context.jsonl`.

Local LoRA completions require serving the matching base NIM and syncing the target adapters into that NIM's `NIM_PEFT_SOURCE`. As of this capture, `runai-rag` LoRA NIM deployments and `rag-oai-proxy` are scaled to zero, and the proxy routes point at older generic adapter names. Do not interpret the absence of local completions as an evaluation result.

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

The golden dataset, target catalog, hosted 70B reference model ID, and Kimi judge ID are captured. Full formal evaluation still requires local serving for the selected LE and Curator adapter revisions. Once local adapters are served, run completion collection first, then Kimi single-axis and pairwise scoring from the saved `responses.jsonl` files.
