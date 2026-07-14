# Stage 1A First-Corpus Pause

Run: NIM curated corpus, Nemotron 3 Super, uncapped LE batch path.

Completed at: 2026-07-10 02:03:18 UTC-equivalent log time.

## Batch Path Verification

- Durable runner: `curator_dataset/experiments/20260709-le-uncapped-batch/run_stage1a_batched_durable.py`
- Batched implementation imported by the runner: `scripts/pipeline/stage1a_batched_kvp.py`
- Configured limits:
  - `temperature`: `0.95`
  - `le_max_tokens`: `16384`
  - `batched_kvp_max_tokens`: `16384`
  - `max_passages`: none
  - `max_entailments`: none
  - `max_premises`: none
  - `max_premises_per_batch`: `64`, used only as an operational chunk size

## Load-Balanced Targets

Canonical model recorded on KVP rows:

- `nvidia/nvidia/nemotron-3-super-v3`

Endpoint-specific targets used:

- Local NIM: `http://10.43.114.25:8000/v1` with `nvidia/nemotron-3-super-120b-a12b@32768`
- remote endpoint: `https://llm.example.com/v1` with `nvidia/nvidia/nemotron-3-super-v3`

The local and external Super model IDs are treated as exact-equivalent targets, with the local context cap used for routing.

Resume-6 log request counts:

- Local target calls: 237
- External target calls: 239
- HTTP 429 responses: 0
- Local context-length HTTP 400 responses: 7
- Retry warnings: 7

The context-aware router recovered from the local context-length 400s and completed the corpus. Before using the load-balanced path for the next corpus, tighten the prompt-token estimator or route any local 400 retry immediately to the uncapped external target.

## Final First-Corpus Counts

- Input passages: 498
- Terminal passage statuses: 498
- Complete passages: 414
- Partial passages: 40
- LE parse failed passages: 44
- `stage1a_le.jsonl` rows: 8169
- `provenance/entailments.jsonl` rows: 2644

## Pause Notes

- No NeMo, Ultra, Stage 1B, Curator conversion, Entity Store, or Data Store work was started after this corpus.
- `le_parse_failed` and `partial` are treated as terminal by the current durable runner. Before moving into the remaining LE pipeline, decide whether to retry those statuses or carry them forward as missing coverage.
