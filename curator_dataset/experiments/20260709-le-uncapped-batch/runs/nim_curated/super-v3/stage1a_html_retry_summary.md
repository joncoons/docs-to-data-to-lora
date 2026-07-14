# NIM Stage 1A HTML-Only Retry Summary

- Created: `2026-07-10T12:27:13.250651Z`
- Scope: retry only HTML Stage 1A passages after PDF exclusion policy.
- Command log: `stage1a_durable_html_retry.log`
- PID file: `stage1a_durable_html_retry.pid`

## Counts
- Stage 1A rows: 9178
- Entailment provenance rows: 3022
- Raw failure payloads: 58

## Status By Doc Kind
- html: {'complete': 425, 'le_parse_failed': 5, 'partial': 1}; incomplete=6
- pdf: {'complete': 65, 'le_parse_failed': 1, 'partial': 1}; incomplete=2

## Remaining HTML Retryable Passages
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/alphafold2-multimer/latest/quickstart-guide.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/bionemo/alphafold2-multimer/latest/quickstart-guide.html
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/boltz2/latest/deploy-helm.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/bionemo/boltz2/latest/deploy-helm.html
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/genmol/latest/endpoints.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/bionemo/genmol/latest/endpoints.html
- `le_parse_failed` `https://docs.nvidia.com/nim/ingestion/object-detection/latest/getting-started.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/ingestion/object-detection/latest/getting-started.html
- `le_parse_failed` `https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/prompt-embeds.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/prompt-embeds.html
- `partial` `https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/prompt-template.html#p0` rows=7 missing=2 url=https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/prompt-template.html
