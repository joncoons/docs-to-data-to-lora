# LE Uncapped Batched Rerun - 2026-07-09

Purpose: rerun logical-entailment dataset generation after removing the inherited Stage 1A entailment/premise caps and switching to the batched KVP path.

## Stage 0 Extraction

| Corpus | ES index | Passages | Output |
| --- | --- | ---: | --- |
| NIM | `nim_curated` | 498 | `stage0/nim_curated/passages.jsonl` |
| NeMo Microservices | `nemo_usvcs_curated` | 486 | `stage0/nemo_usvcs_curated/passages.jsonl` |

Stage 0 used `PIPELINE_ES_HOST=https://10.43.233.46:9200` and the in-cluster `rag-eck-elasticsearch-es-elastic-user` secret.

## Planned Stage 1A Rerun

Run both corpora with both generator models:

- Super: `nvidia/nvidia/nemotron-3-super-v3`
- Ultra: `nvidia/nvidia/nemotron-3-ultra`

Required generation settings:

- Stage 1A mode: batched KVP expansion
- Temperature: `0.95`
- LE max tokens: `16384`
- Batched KVP max tokens: `16384`
- No entailment count cap
- No premise count cap

After extraction, continue through the remainder of the LE dataset pipeline, including Curator/Data Designer-compatible downstream steps, before preparing Entity/Data Store registration artifacts.
