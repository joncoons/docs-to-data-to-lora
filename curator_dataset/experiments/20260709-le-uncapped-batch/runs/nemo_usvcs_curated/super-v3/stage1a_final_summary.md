# NeMo Microservices LE Stage 1A Final Summary

Generated at: 2026-07-10T11:01:44.651931+00:00

## Checkpoints

- First pass: `3ee9b73`
- Recovery prep: `85905e0`
- Recovery pass 1: `dd641a6`
- Recovery 2 prep: `8f1f89a`

## Final Status

- complete: 486
- Unique passages: 486
- Passage result records including retries: 512
- Stage 1A KVP rows: 9611
- Latest complete row-count sum: 9611
- Entailment provenance rows: 3051
- Raw failure payloads retained: 222

## Validation

- Incomplete latest statuses: 0
- Complete passages with zero live rows: 0
- Latest row-count mismatches: 0

## Final Manifest

- selected_passages: 0
- completed_before_start: 486
- temperature: 0.95
- le_max_tokens: 16384
- batched_kvp_max_tokens: 16384

## Targets

- `http://10.43.114.25:8000/v1` using `nvidia/nemotron-3-super-120b-a12b` max_context=32768
- `https://llm.example.com/v1` using `nvidia/nvidia/nemotron-3-super-v3` max_context=None

## Log Health

### stage1a_durable_super_lb_hardened.log
- local_calls: 1016
- external_calls: 1019
- http_400: 0
- http_429: 0
- warnings: 0
- complete_msg: 1

### stage1a_durable_super_lb_recovery1.log
- local_calls: 30
- external_calls: 33
- http_400: 0
- http_429: 0
- warnings: 0
- complete_msg: 1

### stage1a_durable_super_lb_recovery2.log
- local_calls: 2
- external_calls: 1
- http_400: 0
- http_429: 0
- warnings: 0
- complete_msg: 1

## Failure Payload Reasons Total

- batched_kvp/missing_pairs: 29
- batched_kvp/parse_exception: 1
- batched_kvp/parse_failed_or_empty_pairs: 163
- fallback_kvp/parse_failed: 16
- le/parse_failed: 13

## Failure Payload Targets Total

- `http://10.43.114.25:8000/v1`: 54
- `https://llm.example.com/v1`: 168

