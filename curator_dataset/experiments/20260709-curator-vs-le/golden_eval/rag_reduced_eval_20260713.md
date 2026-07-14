# Reduced RAG Evaluation Queue - Golden v1

Date captured: 2026-07-13

## Status

Answer capture and reduced RAG single-axis scoring completed on 2026-07-14. The reduced population was derived from completed Claude no-RAG single-axis results. Local Blackwell GPUs hosted the LoRA-capable 1B and 8B NIMs, the Ada GPUs hosted the 3B path, and the dense 70B reference used the NVIDIA hosted inference endpoint.

## Scope

Use the same reduced population for RAG answer capture, direct single-axis scoring, pairwise scoring, and the first RAGAS diagnostic pass:

- top 1B LoRA winner per corpus from completed single-axis results
- top 3B LoRA winner per corpus from completed single-axis results
- top 8B LoRA winner per corpus from completed single-axis results
- Llama 3.3 70B base reference per corpus

Do not run the full pairwise ocean for the RAG pass.

## Sequential Corpus Deployments

RAG retrieval must be scoped to one corpus collection at a time.

1. NIM pass:
   - `COLLECTION_NAME=nim_curated`
   - input: `golden-v1/nim_curated/test_set.jsonl`
   - output slug: `nim_curated_golden_v1_rag_reduced`

2. NeMo Microservices pass:
   - `COLLECTION_NAME=nemo_usvcs_curated`
   - input: `golden-v1/nemo_usvcs_curated/test_set.jsonl`
   - output slug: `nemo_usvcs_curated_golden_v1_rag_reduced`

The broad `COLLECTION_NAME=nvidia` deployment must not be used for this evaluation because it permits cross-corpus retrieval.

## Retrieval Sizing

Start with the current RAG sizing:

- `VECTOR_DB_TOPK=100`
- `ENABLE_RERANKER=True`
- `APP_RETRIEVER_TOPK=10`
- `APP_RETRIEVER_SCORETHRESHOLD=0.25`
- `RERANKER_CONFIDENCE_THRESHOLD=0.0`

This is right-sized for the reduced smoke/full pass: broad enough for recall, small enough after reranking to avoid excessive context dilution. If the smoke shows noisy context, lower only `APP_RETRIEVER_TOPK` to `5`; keep the candidate pool fixed for comparability.

## Target Deployment Order

Within each corpus deployment, run target models sequentially. Patch the RAG server LLM backend for the active target before collecting answers:

- 3B winner: `APP_LLM_SERVERURL=nim-llm-3b-ada-lora:8000`, selected LE r32 model id, subject to the `ubuntu2` GPU note below.
- 8B winner: `APP_LLM_SERVERURL=nim-llm-8b-bw-lora:8000`, selected LE r32 model id.
- 70B base reference: `APP_LLM_SERVERURL=https://inference-api.nvidia.com/v1`, `APP_LLM_MODELNAME=nvidia/meta/llama-3.3-70b-instruct`.
- 1B winner: `APP_LLM_SERVERURL=nim-llm-1b-bw-lora:8000`, selected LE r32 model id.

Current live layout: local `nim-llm` 70B is scaled to zero, `nim-llm-1b-bw-lora` and `nim-llm-8b-bw-lora` run on `ubuntu-local-dev`, and the 70B comparison is hosted. The capture worker resumes partial JSONL outputs under run id `golden-v1-rag-reduced-20260714`.

## Hosted 70B Runtime Notes

The hosted 70B path uses `APP_LLM_APIKEY` from Kubernetes secret `runai-rag/nvidia-inference-key`, key `api-key`. Do not persist the key value. The older global `NVIDIA_API_KEY` from `ngc-api` is not sufficient for `https://inference-api.nvidia.com/v1`; it is an NGC/NVIDIA key and the endpoint expects the inference virtual key.

`rag-server` must validate both external NVIDIA TLS and the internal Elasticsearch TLS endpoint. The live deployment generates `/tmp/combined-ca.crt` at startup by concatenating `/etc/ssl/certs/cacert.pem` with `/etc/ssl/eck/ca.crt`, then sets `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` to the combined path. A smoke test on 2026-07-14 verified both hosted 70B generation and local NIM RAG retrieval after this change.

## GPU Scheduling Note

This RAG pass should reinstate the prior `ubuntu2` GPU time-slicing profile that was temporarily removed for 3B LoRA training. The prior profile used `timeSlicing.replicas: 5`, so the two Ada GPUs advertise 10 logical `nvidia.com/gpu` slots.

RAG preflight sequence:

1. Restore the `ubuntu2` entry in `gpu-operator/time-slicing-config` to `timeSlicing.replicas: 5`.
2. Restart the `ubuntu2` `nvidia-device-plugin-daemonset` pod and `gpu-feature-discovery` pod.
3. Verify `ubuntu2` reports `nvidia.com/gpu.replicas=5` and `nvidia.com/gpu.sharing-strategy=time-slicing`.
4. Start `nim-llm-3b-ada-lora`, `nemoretriever-embedding-ms`, and `nemoretriever-ranking-ms`.
5. Verify all three pods are Ready and record the `ubuntu2` logical GPU state before 3B RAG answer capture.

The `gpu-operator` time-slicing profile restores schedulability for NIMService-based deployments, but it advertises logical GPU slots and does not by itself guarantee physical GPU isolation. If physical isolation is required, verify actual device assignment out of band or convert the services to a Run:ai-native workload shape that supports `gpuMemory`; do not disable reranking only for 3B, because that would change the retrieval semantics for one size class.

## Post-Capture Scoring Gate

After RAG answer capture completes, run reduced RAG single-axis before any pairwise scoring. The single-axis pass scores the saved RAG answers against the immutable golden reference answer using Claude Sonnet 4.6 as the judge; retrieved context is not sent to the judge. This preserves the same standalone answer-quality rubric while labeling the artifacts as `rag_reduced` answer mode.

The live post-capture launcher is `rag_singleaxis_after_capture_20260714.py`. It waits for `rag_capture_status_20260714.json` to become `complete`, verifies all eight reduced answer sets are complete with zero unresolved failures, then runs `scripts/eval/run_direct_kimi_singleaxis.py` with:

- output root: `/mnt/nvme2/peft/evals/singleaxis-claude-sonnet-4-6-rag-reduced`
- repo summary dir: `golden_eval/claude_rag_reduced_20260714`
- eval run id: `golden-v1-claude-sonnet-4-6-rag-reduced-20260714`
- judge endpoint/model: `https://inference-api.nvidia.com/v1`, `azure/anthropic/claude-sonnet-4-6`
- answer labels: `--answer-mode rag_reduced --uses-rag-answer`

Pairwise remains blocked until the RAG single-axis status file reports `complete` and the summaries show zero unresolved scoring failures.

## Output Roots

Recommended durable output roots:

- RAG answers: `/mnt/nvme2/peft/evals/completions-rag-reduced`
- RAG single-axis: `/mnt/nvme2/peft/evals/singleaxis-claude-sonnet-4-6-rag-reduced`
- RAG pairwise: `/mnt/nvme2/peft/evals/pairwise-claude-sonnet-4-6-rag-reduced` (gated until RAG single-axis completes)
- Optional RAGAS: `/mnt/nvme2/peft/evals/nemo-evaluator-claude-ragas-reduced`

Repo-local summaries live under `golden_eval/claude_rag_reduced_20260714/` and MLflow export should remain enabled. Documentation graphics for the completed no-RAG/RAG comparison live under `golden_eval/graphics/`.


## Completed Scoring And Graphics

The completed reduced RAG single-axis run writes full artifacts to `/mnt/nvme2/peft/evals/singleaxis-claude-sonnet-4-6-rag-reduced` and repo summaries to `claude_rag_reduced_20260714/`. Generated SVG graphics and their source JSON are under `graphics/`:

- `graphics/rag_vs_norag_composite.svg`
- `graphics/rag_vs_norag_axis_heatmap.svg`
- `graphics/golden_eval_score_table.svg`
- `graphics/ragas_coverage_status.svg`
- `graphics/golden_eval_graphics_data.json`

RAGAS remains pending as a formal reduced-population diagnostic. The current RAGAS SVG records coverage/status only because the existing one-row smoke failed before scoring without `params.judge_embeddings.model`.

## Execution Gate

Do not deploy the RAG pass until all gates are true:

- Claude no-RAG single-axis has completed for all 26 saved answer sets.
- Reduced winners have been selected from complete single-axis results.
- The reduced winner manifest has been written locally.
- RAG server collection scoping has been patched to the active corpus before launch.
- A one-row smoke confirms the active corpus does not retrieve from the other corpus.
- Hosted 70B smoke succeeds through `rag-server` using `nvidia/meta/llama-3.3-70b-instruct` and `APP_LLM_APIKEY` from `nvidia-inference-key`.
