# LE Uncapped Batched Rerun - 2026-07-09

Purpose: rerun logical-entailment dataset generation after removing the inherited Stage 1A entailment/premise caps and switching to the batched KVP path.

## Stage 0 Extraction

| Corpus | ES index | Passages | Output |
| --- | --- | ---: | --- |
| NIM | `nim_curated` | 498 | `stage0/nim_curated/passages.jsonl` |
| NeMo Microservices | `nemo_usvcs_curated` | 486 | `stage0/nemo_usvcs_curated/passages.jsonl` |

Stage 0 used `PIPELINE_ES_HOST=https://10.43.233.46:9200` and the in-cluster `rag-eck-elasticsearch-es-elastic-user` secret.

## Stage 1A Rerun

Run both corpora with Nemotron 3 Super for logical-entailment extraction and
KVP expansion. Treat `nvidia/nvidia/nemotron-3-super-v3` and
`nvidia/nemotron-3-super-120b-a12b` as the same Super 120B model for this work.

Ultra 550B is parked for Stage 1A. It remains the preferred frontier-level
model for downstream synthesis and instruction-diversity stages such as 1B and
1C, where cross-passage synthesis and controlled rewriting matter more than raw
LE extraction throughput.

Required generation settings:

- Stage 1A mode: batched KVP expansion
- Temperature: `0.95`
- LE max tokens: `16384`
- Batched KVP max tokens: `16384`
- No entailment count cap
- No premise count cap

After extraction, continue through the remainder of the LE dataset pipeline, including Curator/Data Designer-compatible downstream steps, before preparing Entity/Data Store registration artifacts.

## Downstream Runner Defaults

`run_le_downstream.py` is intended to stay runnable with either one inference endpoint or several.
Use repeated `--target` arguments to add targets:

```bash
# Single frontier endpoint
python run_le_downstream.py   --collection nim_curated   --output-dir runs/nim_curated/super-v3   --stage 1b   --resume   --target https://inference-api.nvidia.com/v1=nvidia/nvidia/nemotron-3-ultra

# Two OpenAI-compatible frontier endpoints, load-balanced
python run_le_downstream.py   --collection nim_curated   --output-dir runs/nim_curated/super-v3   --stage 1b   --resume   --target https://inference-api.nvidia.com/v1=nvidia/nvidia/nemotron-3-ultra   --target https://frontier.example.com/v1=vendor/frontier-model@32768   --api-key "$FRONTIER_API_KEY"
```

Target format is `ENDPOINT=MODEL` or `ENDPOINT=MODEL@MAX_CONTEXT_TOKENS`.
The endpoint must implement OpenAI-compatible chat completions. Hosted NVIDIA
endpoints use the configured external API key; local OpenAI-compatible endpoints
use a placeholder `local` key unless `--api-key` is supplied. The code does not
require Nemotron specifically: any model ID accepted by the target endpoint can
be used, and repeated `--target` values are load-balanced. For this experiment,
Stage 1B and Stage 1C should use a frontier-level model such as Nemotron 3
Ultra. Stage 1A extraction should stay on Super unless an experiment explicitly
overrides it.

The runner defaults to all source document kinds for turnkey use. Pass `--source-doc-kind html` only when an experiment intentionally excludes parsed PDFs. When a source filter is active, the runner writes selected KVP and lineage sidecars so Data Store publication can use only the selected provenance.

If `--allow-incomplete-stage1a` is supplied, selected passages whose latest Stage 1A status is incomplete are excluded from downstream stages and recorded in `provenance/source_filter_excluded_passages.jsonl` with a `stage1a_*` reason. This keeps downstream augmentation grounded in completed Stage 1A rows without deleting the raw extraction artifacts.

Stage 1B is durable: `stage1b_synthesis.jsonl` is appended as each passage finishes, and `stage1b_passage_results.jsonl` records per-passage status. Resume runs skip passages that already have persisted rows or terminal no-work statuses.

Stage 1C is durable the same way: `stage1c_instruction.jsonl` is appended as passages finish, and `stage1c_passage_results.jsonl` records per-passage status. The default selection mode is `stratified`; pass `--stage1c-selection-mode top_density` for legacy high-density-only behavior or `--stage1c-selection-mode all` for full instruction augmentation coverage.

Stage 1B retrieves kNN neighbors from the target corpus Elasticsearch index selected by `--collection`; for example, `nim_curated` neighbors come from `nim_curated`, not from the NeMo Microservices corpus.
