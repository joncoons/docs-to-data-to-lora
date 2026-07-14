# NeMo Microservices LE Stage 1A First Pass Summary

Generated locally after the first durable, load-balanced Nemotron 3 Super pass.

## Run

- Input passages: 486
- Started at: 2026-07-10T07:30:18.147321Z
- Finished at: 2026-07-10T10:39:18.213346Z
- Temperature: 0.95
- LE max tokens: 16384
- Batched KVP max tokens: 16384
- Max workers: 6
- Min request interval seconds: 3.0
- Max premises per batch: 64

## Targets

- `http://10.43.114.25:8000/v1` using `nvidia/nemotron-3-super-120b-a12b` max_context=32768
- `https://inference-api.nvidia.com/v1` using `nvidia/nvidia/nemotron-3-super-v3` max_context=None

## First-Pass Results

- complete: 461
- exception: 1
- le_parse_failed: 13
- partial: 11
- Stage 1A KVP rows: 9166
- Entailment provenance rows: 2918
- Raw failure payloads: 210

## Endpoint Health

- local_calls: 1016
- external_calls: 1019
- http_400: 0
- http_429: 0
- warnings: 0

## Failure Payload Reasons

- batched_kvp/missing_pairs: 28
- batched_kvp/parse_failed_or_empty_pairs: 154
- fallback_kvp/parse_failed: 15
- le/parse_failed: 13

## Failure Payload Targets

- `http://10.43.114.25:8000/v1`: 50
- `https://inference-api.nvidia.com/v1`: 160

## Recovery Set

Total retryable passages after first pass: 25

| status | rows | missing | passage |
| --- | ---: | ---: | --- |
| exception | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/example-applications/tool-calling.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/audit/configs/create-config.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/auth/authorization/managing-access.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/customizer/tutorials/dpo-customization-job.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/customizer/tutorials/embedding-customization-job.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/evaluator/metrics/agentic.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/evaluator/metrics/remote.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/evaluator/tutorials/run-llm-judge-evaluation.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/guardrails/observability.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/guardrails/tutorials/index.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/run-inference/tutorials/deploy-models.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/run-inference/tutorials/run-inference.html#p0` |
| le_parse_failed | 0 | 0 | `https://docs.nvidia.com/nemo/microservices/latest/troubleshooting/evaluator.html#p0` |
| partial | 43 | 2 | `https://docs.nvidia.com/nemo/microservices/latest/auth/authorization/api-scopes.html#p0` |
| partial | 18 | 1 | `https://docs.nvidia.com/nemo/microservices/latest/auth/index.html#p0` |
| partial | 70 | 1 | `https://docs.nvidia.com/nemo/microservices/latest/data-designer/migration.html#p0` |
| partial | 16 | 3 | `https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/define-your-data-columns/column-types/index.html#p0` |
| partial | 5 | 1 | `https://docs.nvidia.com/nemo/microservices/latest/evaluate/tutorials/index.html#p0` |
| partial | 44 | 1 | `https://docs.nvidia.com/nemo/microservices/latest/evaluator/metrics/llm-as-a-judge.html#p0` |
| partial | 47 | 2 | `https://docs.nvidia.com/nemo/microservices/latest/get-started/concepts/manage-secrets.html#p0` |
| partial | 11 | 1 | `https://docs.nvidia.com/nemo/microservices/latest/guardrails/manage-guardrail-configs/list-configs.html#p0` |
| partial | 34 | 1 | `https://docs.nvidia.com/nemo/microservices/latest/guardrails/tutorials/deploy-nemoguard-nims.html#p0` |
| partial | 13 | 1 | `https://docs.nvidia.com/nemo/microservices/latest/guardrails/tutorials/integrate-nemoguard-nims.html#p0` |
| partial | 21 | 1 | `https://docs.nvidia.com/nemo/microservices/latest/index.html#p0` |

## Recovery Note

Partial passage rows are provisional in `stage1a_le.jsonl`. Before a resume pass, archive the first-pass row/provenance files and filter rows for non-complete latest passage statuses so retries can replace them without duplicates.
