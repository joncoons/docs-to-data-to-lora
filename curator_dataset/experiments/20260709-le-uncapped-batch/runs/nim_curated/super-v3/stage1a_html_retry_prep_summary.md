# NIM Stage 1A HTML-Only Retry Prep

- Created: `2026-07-10T12:19:58.136184Z`
- Scope: remove provisional rows only for retryable HTML passages; keep PDF extraction artifacts preserved but out of dataset scope.
- Local archive: `.local-archive/20260710-nim-super-stage1a-html-retry-prep`
- Retry input: `passages_html_retry.jsonl`
- Retry passages: 13
- Retry statuses: `{'le_parse_failed': 9, 'partial': 4}`
- Removed provisional Stage 1A rows: 44
- Remaining Stage 1A rows: 9031
- Remaining entailment provenance rows: 2962

## Retry Passage IDs
- `le_parse_failed` `https://docs.nvidia.com/nim/alchemi/alchemi-bmd/latest/api-reference.html#p0` rows=0 missing=0
- `le_parse_failed` `https://docs.nvidia.com/nim/alchemi/alchemi-bmd/latest/quickstart-guide.html#p0` rows=0 missing=0
- `le_parse_failed` `https://docs.nvidia.com/nim/benchmarking/llm/latest/step-by-step.html#p0` rows=0 missing=0
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/alphafold2-multimer/latest/quickstart-guide.html#p0` rows=0 missing=0
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/boltz2/latest/deploy-helm.html#p0` rows=0 missing=0
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/genmol/latest/endpoints.html#p0` rows=0 missing=0
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/msa-search/latest/release-notes.html#p0` rows=0 missing=0
- `le_parse_failed` `https://docs.nvidia.com/nim/ingestion/object-detection/latest/getting-started.html#p0` rows=0 missing=0
- `partial` `https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/prompt-embeds.html#p0` rows=33 missing=1
- `partial` `https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/prompt-template.html#p0` rows=3 missing=4
- `partial` `https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest/prompt-template.html#p0` rows=7 missing=1
- `le_parse_failed` `https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/performance.html#p0` rows=0 missing=0
- `partial` `https://docs.nvidia.com/nim/nvclip/latest/EULA.html#p0` rows=1 missing=1
