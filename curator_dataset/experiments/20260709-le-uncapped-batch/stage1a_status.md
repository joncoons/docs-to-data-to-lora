# Stage 1A Uncapped Batch Status

Updated: 2026-07-10T01:47:13.610186Z

## NIM / Super

- Model: `nvidia/nvidia/nemotron-3-super-v3`
- Endpoint: `https://inference-api.nvidia.com/v1`
- Temperature: `0.95`
- LE max tokens: `16384`
- Batched KVP max tokens: `16384`
- Passage inputs: `498`
- Current QA rows: `1607`
- Passage result records: `277`
- Status counts: `{'le_parse_failed': 9, 'complete': 91, 'partial': 15, 'le_no_response': 162}`

The first naive ThreadPool launch held five initial HTTPS requests open for more than 15 minutes and was interrupted before writing rows. The durable runner then completed useful work but hit NVIDIA endpoint 429 throttling at about 55% of the passage future set. `le_no_response` records are treated as transient and retried by the durable runner after the resume-logic fix.

Next resume setting: single worker, longer cooldown, `--min-request-interval-s 10`, `--retry-attempts 8`, `--retry-base-delay-s 30`.
