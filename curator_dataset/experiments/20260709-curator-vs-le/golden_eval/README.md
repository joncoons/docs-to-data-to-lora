# Golden Evaluation Plan - LE vs Curator vs Llama 3.3 70B

Date captured: 2026-07-12

## Purpose

This directory freezes the evaluation dataset and target plan for comparing the current Logical Entailment (LE) and Curator DiverseQA LoRA adapters, then comparing the best LoRA winner against a dense Llama 3.3 70B reference target.

Llama 3.3 70B is a comparison target, not the judge. The judge remains independent.


## Applying This To Your Own Corpora

The same process is intended to be reusable for any corpus or group of corpora made from unstructured source material. The experiment-specific names in this directory are NIM and NeMo Microservices, but the durable workflow is corpus-agnostic:

1. Register each corpus with a stable corpus slug and retain source provenance for every generated row.
2. Generate training candidates through one or both dataset paths: LE extraction/generation and Curator DiverseQA.
3. Configure the generation model and endpoint per stage. The scripts support local NIM endpoints, hosted NVIDIA inference endpoints, and multiple endpoint entries for load sharing or fallback.
4. Run the downstream dataset preparation stages with the same reliability controls: resumable JSONL writes, per-row provenance, failure capture, retryable batches, deterministic splits, and leak checks.
5. Register finalized datasets with Entity/Data Store or the target dataset store expected by the training environment.
6. Train the comparison adapters with fixed hyperparameter grids, then collect no-RAG answer sets from an immutable golden QA set to isolate model and adapter behavior.
7. Optionally collect RAG answer sets against corpus-scoped retrieval collections. Scope retrieval to the active corpus collection so a NIM question cannot retrieve from the NeMo collection, or vice versa.
8. Score no-RAG and RAG answer sets with the same independent judge rubric, export repo-local artifacts and MLflow telemetry, then use optional RAGAS as a retrieval diagnostic rather than the primary LoRA winner criterion.

For a new corpus, the minimum project-specific inputs are the source document set, corpus slug, generation model endpoint configuration, target base models/adapters, and a frozen golden QA set for evaluation. The evaluation graphics under `graphics/` are generated from summary files, so the same rendering path can be reused once a new corpus writes compatible single-axis and RAGAS summaries.

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

The formal winner evaluation is no-RAG. Model answers are collected from `test_set.jsonl` question-only prompts, then judged against the immutable golden reference answer. The judge must not receive retrieved context or context-baked prompts for the primary LE-vs-Curator comparison.

Single-axis evaluation measures standalone model efficacy with a direct saved-response judge over completion files. The active judge for the formal path is Claude Sonnet 4.6 via `https://inference-api.nvidia.com/v1`; Kimi is no longer part of the active evaluation plan. The active rubric axes are accuracy, completeness, reference-grounded faithfulness, and clarity. Here, faithfulness means the model avoids contradictions or unsupported additions relative to the golden reference answer; it is not a RAG/source-context metric.

Pairwise evaluation is intentionally reduced after single-axis completes:

1. For each corpus and dense size, select the top-scoring LoRA from single-axis across LE vs Curator and r16 vs r32.
2. Compare only those selected 1B, 3B, and 8B LoRA winners against the dense Llama 3.3 70B reference target.

This avoids rebuilding the full pairwise matrix after the single-axis run has already identified the strongest candidate in each size class. Use position-swapped pairwise judging for all pairwise runs. Select the winner primarily by pairwise win rate; use single-axis composite as a tie breaker. Export every scoring run both to repo-local artifacts and to MLflow.

## Judge And Reference Models

Primary judge:

- Endpoint: `https://inference-api.nvidia.com/v1`
- Model: `azure/anthropic/claude-sonnet-4-6`
- Scope: single-axis and pairwise saved-response judging for the active no-RAG and RAG reduced evaluation paths.

Historical note:

- Kimi K2.6 smoke artifacts from 2026-07-12 are retained only as provenance for prior endpoint testing. Kimi is not part of the active evaluation path.

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

## No-RAG Claude Scoring

Single-axis scoring over saved question-only completions uses an independent Claude Sonnet 4.6 judge. Use the generic direct LLM scorer wrappers for the active path; they default to Claude Sonnet 4.6 and still allow explicit endpoint/model overrides.

```bash
export NVIDIA_API_KEY="$(kubectl get secret -n runai-rag nvidia-inference-key -o jsonpath='{.data.api-key}' | base64 -d)"
/home/joncoons/anaconda3/bin/python scripts/eval/run_direct_llm_singleaxis.py \
  --responses /mnt/nvme2/peft/evals/completions-question-only/<dataset>/<base>/<target>/<rank>/<run>/responses.jsonl \
  --completions-root /mnt/nvme2/peft/evals/completions-question-only \
  --output-root /mnt/nvme2/peft/evals/singleaxis-claude-sonnet-4-6-norag \
  --repo-summary-dir curator_dataset/experiments/20260709-curator-vs-le/golden_eval/claude_norag_20260713 \
  --judge-api-url https://inference-api.nvidia.com/v1 \
  --judge-model azure/anthropic/claude-sonnet-4-6 \
  --judge-api-key-env NVIDIA_API_KEY \
  --eval-run-id golden-v1-claude-sonnet-4-6-norag-20260713 \
  --resume \
  --concurrency 1
```

Pairwise scoring over saved question-only completions should use the same Claude judge and position swapping:

```bash
/home/joncoons/anaconda3/bin/python scripts/eval/run_direct_llm_pairwise.py \
  --pair le-r32 /path/to/le/responses.jsonl curator-r32 /path/to/curator/responses.jsonl \
  --output-root /mnt/nvme2/peft/evals/pairwise-claude-sonnet-4-6-norag \
  --repo-summary-dir curator_dataset/experiments/20260709-curator-vs-le/golden_eval/claude_pairwise_norag_20260714 \
  --judge-api-url https://inference-api.nvidia.com/v1 \
  --judge-model azure/anthropic/claude-sonnet-4-6 \
  --judge-api-key-env NVIDIA_API_KEY \
  --eval-run-id golden-v1-claude-sonnet-4-6-norag-pairwise-20260714 \
  --position-swap \
  --resume \
  --concurrency 1
```

Both direct saved-response scorers write compact repo-local summaries and export the full local run directory to MLflow by default:

- Repo summaries: `golden_eval/claude_norag_20260713/<eval_run_id>/*.json` and `golden_eval/claude_pairwise_norag_20260714/<eval_run_id>/*.json`
- Full local artifacts: `/mnt/nvme2/peft/evals/singleaxis-claude-sonnet-4-6-norag/` and `/mnt/nvme2/peft/evals/pairwise-claude-sonnet-4-6-norag/`
- MLflow tracking URI: `http://10.43.102.80:5000`
- MLflow experiment: `docs-to-data-to-lora-golden-eval`
- MLflow artifact location: `file:///mnt/nvme2/peft/mlflow-artifacts/golden-eval`

Use `--no-mlflow` only for local parser/debug runs that should not publish telemetry.

## Historical Kimi Smoke Result

A one-row direct Kimi smoke was run on 2026-07-12 to validate early endpoint wiring, repo summary capture, and MLflow artifact export. This is retained as provenance only and is not part of the active evaluation path.

| Scope | Dataset | Comparison target | Rows | Failures | Result | MLflow run |
| --- | --- | --- | ---: | ---: | --- | --- |
| Single-axis | `nim_curated_golden_v1_question_only` | `llama-3.2-1b` / `lora-nim-le-super-v3-e5` / `r16` | 1 | 0 | accuracy 1.0, completeness 2.0, faithfulness 1.0, clarity 5.0 | `47c7fc926c864a5186ce0a43f20e5eb6` |
| Pairwise | `nim_curated_golden_v1_question_only` | `le-r16` vs `curator-r16` | 1 | 0 | left/LE won; position-swap agreement `agree` | `5067ae704f5e420cab8e47148cf2cef5` |

Historical repo summaries remain under `kimi_norag_20260712/golden-v1-kimi-norag-smoke-20260712/` and `kimi_norag_20260712/golden-v1-kimi-norag-pairwise-smoke-20260712/`.

## Optional RAGAS Follow-Up

NeMo Evaluator RAGAS remains useful as a later diagnostic, but it is not part of the formal LE-vs-Curator winner selection because RAGAS is context/retrieval-oriented. Running RAGAS requires context-backed rows and, for `response_relevancy`, an embedding judge endpoint such as `nemoretriever-embedding-ms` with `input_type=query` support.

The optional runner is `scripts/eval/run_nemo_evaluator_saved_responses.py`. For the active path, keep outputs under `/mnt/nvme2/peft/evals/nemo-evaluator-claude-ragas-reduced/` and repo summaries under a Claude-specific RAGAS directory. A historical one-row 2026-07-12 Kimi smoke reached Evaluator and MLflow but was intentionally stopped for formal evaluation because it would move the experiment back into RAGAS semantics.

## Reduced RAG Follow-Up

After the Claude single-axis run completes, run the RAG answer-capture and RAG-aware scoring on the same reduced population used for pairwise: the top 1B, 3B, and 8B LoRA winners per corpus plus the Llama 3.3 70B base reference. For the 2026-07-14 RAG capture, the 70B reference is hosted at `https://inference-api.nvidia.com/v1` as `nvidia/meta/llama-3.3-70b-instruct`; local Blackwell GPUs are reserved for the LoRA-capable 1B and 8B NIMs.

RAG deployment must be scoped sequentially by corpus. Do not use the broad `nvidia` collection for this evaluation.

| Step | Corpus | Required `COLLECTION_NAME` | Input questions |
| --- | --- | --- | --- |
| 1 | NIM | `nim_curated` | `golden-v1/nim_curated/test_set.jsonl` |
| 2 | NeMo Microservices | `nemo_usvcs_curated` | `golden-v1/nemo_usvcs_curated/test_set.jsonl` |

Keep retrieval sizing fixed unless a smoke run proves it is too noisy:

- Candidate retrieval: `VECTOR_DB_TOPK=100`
- Reranker: `ENABLE_RERANKER=True`
- Final returned context count: `APP_RETRIEVER_TOPK=10`
- Retriever score threshold: `APP_RETRIEVER_SCORETHRESHOLD=0.25`
- Reranker confidence threshold: `RERANKER_CONFIDENCE_THRESHOLD=0.0`

For each corpus deployment, patch the RAG server to the active corpus collection and then run target models sequentially by patching the RAG server LLM backend to the selected target's NIM service and model id. Store RAG answer sets separately from question-only artifacts, for example under `/mnt/nvme2/peft/evals/completions-rag-reduced`.

The RAG single-axis scorer should run after RAG answer capture completes and before pairwise. It compares saved RAG answers to the immutable golden reference answer, does not send retrieved context to the judge, and tags MLflow/repo artifacts as `rag_reduced` RAG-answer mode. Pairwise should compare each reduced LoRA winner against the same-corpus 70B RAG answer set only after RAG single-axis completes with zero unresolved scoring failures. RAGAS should use this same reduced population first; expand only if the reduced result is ambiguous or surprising.

Operational note: the hosted 70B path requires `APP_LLM_APIKEY` from Kubernetes secret `runai-rag/nvidia-inference-key`, key `api-key`, and `rag-server` must use a combined system-plus-ECK CA bundle so both external NVIDIA HTTPS and internal Elasticsearch TLS work. The live deployment creates `/tmp/combined-ca.crt` at startup and points `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` at it.

Operational note: this RAG pass should restore the prior `ubuntu2` GPU time-slicing profile before deploying retrieval services. The pre-training profile used `timeSlicing.replicas: 5` for `ubuntu2`, advertising 10 logical GPU slots across the two Ada GPUs. Restore that profile, restart the `ubuntu2` NVIDIA device-plugin and GPU Feature Discovery pods, and verify `nvidia.com/gpu.replicas=5` before starting the RAG pass. This restores schedulability for the NIMService-based 3B, embedding, and reranker deployments. It does not guarantee physical GPU isolation; if physical isolation is required, verify actual device assignment out of band or convert the services to a Run:ai-native workload shape that supports `gpuMemory`.

## Hosted Smoke Result

A constrained hosted smoke was run on 2026-07-12 with two rows from each corpus to verify endpoint wiring. This is not a formal winner evaluation.

| Dataset slug | Target | Rows completed | Judge rows scored | Mean accuracy | Mean completeness | Mean faithfulness | Mean clarity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nim_curated_golden_v1` | `nvidia/meta/llama-3.3-70b-instruct` | 2 | 2 | 5.0 | 4.0 | 5.0 | 5.0 |
| `nemo_usvcs_curated_golden_v1` | `nvidia/meta/llama-3.3-70b-instruct` | 2 | 2 | 3.5 | 3.5 | 3.5 | 4.0 |

The historical smoke summary is captured in `hosted_70b_kimi_smoke_20260712.json`; raw completion and score artifacts are under `/mnt/nvme2/peft/evals/`. This artifact is retained only as provenance for prior endpoint testing.


## Result Graphics

Documentation-ready SVG graphics are captured under `graphics/` and can be regenerated with:

```bash
/home/joncoons/anaconda3/bin/python scripts/eval/render_golden_eval_graphics.py
```

Artifacts:

- `graphics/rag_vs_norag_composite.svg` shows the reduced RAG lift over the comparable no-RAG population.
- `graphics/rag_vs_norag_axis_heatmap.svg` shows axis-level scores for accuracy, completeness, reference-grounded faithfulness, and clarity.
- `graphics/golden_eval_score_table.svg` provides a numeric SVG table for the same reduced population.
- `graphics/ragas_coverage_status.svg` records RAGAS coverage/status.
- `graphics/golden_eval_graphics_data.json` contains the exact source data used to render the SVGs.

The no-RAG and RAG charts use the same reduced comparison set: 1B, 3B, and 8B LE r32 adapters plus the dense Llama 3.3 70B reference for NIM and NeMo Microservices. The RAGAS graphic is currently a status artifact only: full reduced-population RAGAS has not been run, and the existing one-row smoke failed before scoring because `params.judge_embeddings.model` was not configured.

## Current Status

The golden dataset, target catalog, hosted 70B reference model ID, judge endpoint IDs, LoRA answer-set materialization, no-RAG completion outputs, reduced RAG completion outputs, reduced RAG single-axis scores, and documentation SVG graphics are captured. NeMo Evaluator RAGAS is retained as an optional later diagnostic and still needs a configured judge embedding model before it can produce formal metrics.
