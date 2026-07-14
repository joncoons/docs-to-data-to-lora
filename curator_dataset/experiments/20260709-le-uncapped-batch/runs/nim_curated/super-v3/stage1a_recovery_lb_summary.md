# NIM Stage 1A Load-Balanced Recovery Summary

- Created: `2026-07-10T12:18:47.357434Z`
- Scope: NIM Super Stage 1A recovery pass using local Super plus a configured OpenAI-compatible Super alias.
- Dataset policy update: downstream datasets are HTML-only; PDF/document-capture passages remain preserved in extraction artifacts but are excluded from dataset creation.
- Command log: `stage1a_durable_recovery_lb.log`
- PID file: `stage1a_durable_recovery_lb.pid`

## Counts
- Passage result records: 744
- Unique passages: 498 / 498
- Stage 1A rows: 9075
- Entailment provenance rows: 2977
- Raw failure payloads: 47

## Status By Doc Kind
- html: {'complete': 418, 'partial': 4, 'le_parse_failed': 9}; incomplete=13
- pdf: {'complete': 65, 'le_parse_failed': 1, 'partial': 1}; incomplete=2

## HTML Retryable Passages
- `le_parse_failed` `https://docs.nvidia.com/nim/alchemi/alchemi-bmd/latest/api-reference.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/alchemi/alchemi-bmd/latest/api-reference.html
- `le_parse_failed` `https://docs.nvidia.com/nim/alchemi/alchemi-bmd/latest/quickstart-guide.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/alchemi/alchemi-bmd/latest/quickstart-guide.html
- `le_parse_failed` `https://docs.nvidia.com/nim/benchmarking/llm/latest/step-by-step.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/benchmarking/llm/latest/step-by-step.html
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/alphafold2-multimer/latest/quickstart-guide.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/bionemo/alphafold2-multimer/latest/quickstart-guide.html
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/boltz2/latest/deploy-helm.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/bionemo/boltz2/latest/deploy-helm.html
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/genmol/latest/endpoints.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/bionemo/genmol/latest/endpoints.html
- `le_parse_failed` `https://docs.nvidia.com/nim/bionemo/msa-search/latest/release-notes.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/bionemo/msa-search/latest/release-notes.html
- `le_parse_failed` `https://docs.nvidia.com/nim/ingestion/object-detection/latest/getting-started.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/ingestion/object-detection/latest/getting-started.html
- `partial` `https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/prompt-embeds.html#p0` rows=33 missing=1 url=https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/prompt-embeds.html
- `partial` `https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/prompt-template.html#p0` rows=3 missing=4 url=https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/prompt-template.html
- `partial` `https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest/prompt-template.html#p0` rows=7 missing=1 url=https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest/prompt-template.html
- `le_parse_failed` `https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/performance.html#p0` rows=0 missing=0 url=https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/performance.html
- `partial` `https://docs.nvidia.com/nim/nvclip/latest/EULA.html#p0` rows=1 missing=1 url=https://docs.nvidia.com/nim/nvclip/latest/EULA.html

## Next Step
- Archive this pass, filter live rows for the 13 HTML retryable passages, regenerate entailment provenance, and rerun Stage 1A only for those HTML passages.
