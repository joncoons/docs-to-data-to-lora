# NeMo Microservices LE Stage 1A Recovery Prep

Prepared at: 2026-07-10T10:41:59.927373+00:00

## Source First Pass

- First-pass checkpoint commit: `3ee9b73`
- First-pass summary: `stage1a_first_pass_summary.md`
- Local archive: `.local-archive/20260710-nemo-usvcs-super-stage1a-first-pass`

## Latest First-Pass Status Counts

- complete: 461
- exception: 1
- le_parse_failed: 13
- partial: 11

## Filtering

- Retryable passage IDs: 25
- Partial passage IDs with provisional rows removed: 11
- KVP rows before filtering: 9166
- KVP rows removed: 322
- KVP rows after filtering: 8844
- Entailment rows after regeneration: 2828

| removed_rows | passage |
| ---: | --- |
| 43 | `https://docs.nvidia.com/nemo/microservices/latest/auth/authorization/api-scopes.html#p0` |
| 18 | `https://docs.nvidia.com/nemo/microservices/latest/auth/index.html#p0` |
| 70 | `https://docs.nvidia.com/nemo/microservices/latest/data-designer/migration.html#p0` |
| 16 | `https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/define-your-data-columns/column-types/index.html#p0` |
| 5 | `https://docs.nvidia.com/nemo/microservices/latest/evaluate/tutorials/index.html#p0` |
| 44 | `https://docs.nvidia.com/nemo/microservices/latest/evaluator/metrics/llm-as-a-judge.html#p0` |
| 47 | `https://docs.nvidia.com/nemo/microservices/latest/get-started/concepts/manage-secrets.html#p0` |
| 11 | `https://docs.nvidia.com/nemo/microservices/latest/guardrails/manage-guardrail-configs/list-configs.html#p0` |
| 34 | `https://docs.nvidia.com/nemo/microservices/latest/guardrails/tutorials/deploy-nemoguard-nims.html#p0` |
| 13 | `https://docs.nvidia.com/nemo/microservices/latest/guardrails/tutorials/integrate-nemoguard-nims.html#p0` |
| 21 | `https://docs.nvidia.com/nemo/microservices/latest/index.html#p0` |

## Resume Behavior

The durable runner will skip first-pass complete/no-entailment passages and retry statuses `exception`, `le_no_response`, `le_parse_failed`, and `partial`. Because provisional rows for incomplete passages were removed before resume, successful retries can append replacement rows without duplicating first-pass partial rows.
