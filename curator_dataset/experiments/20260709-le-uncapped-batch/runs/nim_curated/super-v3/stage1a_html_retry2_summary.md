# NIM Stage 1A HTML-Only Retry 2 Summary

- Created: `2026-07-10T12:36:47.235840Z`
- Scope: second retry only HTML Stage 1A passages after PDF exclusion policy.
- Command log: `stage1a_durable_html_retry2.log`
- PID file: `stage1a_durable_html_retry2.pid`

## Counts
- Stage 1A rows: 9312
- Entailment provenance rows: 3049
- Raw failure payloads: 61

## Status By Doc Kind
- html: {'complete': 430, 'le_parse_failed': 1}; incomplete=1
- pdf: {'complete': 65, 'le_parse_failed': 1, 'partial': 1}; incomplete=2

## Remaining HTML Retryable Passages
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/boltz2/latest/deploy-helm.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/bionemo/boltz2/latest/deploy-helm.html
