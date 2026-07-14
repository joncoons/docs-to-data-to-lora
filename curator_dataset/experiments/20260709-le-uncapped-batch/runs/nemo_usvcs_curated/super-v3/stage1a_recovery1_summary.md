# NeMo Microservices LE Stage 1A Recovery Pass 1 Summary

Generated at: 2026-07-10T10:56:53.778584+00:00

## Inputs

- Started from recovery prep checkpoint: `85905e0`
- Selected retryable passages: 25
- Completed before start: 461

## Latest Status After Recovery 1

- complete: 485
- partial: 1
- Stage 1A KVP rows: 9612
- Entailment provenance rows: 3049
- Raw failure payloads total: 221

## Recovery 1 Log Counts

- complete_lines: 24
- partial_lines: 1
- le_parse_failed_lines: 0
- exception_lines: 0
- http_400: 0
- http_429: 0
- warnings: 0
- complete_msg: 1

## Remaining Retryable Passages

Total remaining: 1

| status | rows | missing | passage |
| --- | ---: | ---: | --- |
| partial | 7 | 1 | `https://docs.nvidia.com/nemo/microservices/latest/evaluate/tutorials/index.html#p0` |

## Failure Payload Reasons Total

- batched_kvp/missing_pairs: 29
- batched_kvp/parse_exception: 1
- batched_kvp/parse_failed_or_empty_pairs: 162
- fallback_kvp/parse_failed: 16
- le/parse_failed: 13
