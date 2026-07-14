# Stage 1A RCA: Partial Passages and LE Parse Failures

Scope: NIM curated corpus, Nemotron 3 Super, uncapped batched Stage 1A run.

Committed baseline: `94927f3 Complete NIM Super uncapped Stage 1A batch run`

## Decision

The Ultra 550B full extraction should be deferred for now. The NIM/Super LE dataset expanded substantially once the old entailment/premise caps were removed, so the next useful step is to harden and run the NeMo Microservices corpus through the same Super path rather than paying for another extraction pass.

## Final Status Counts

- Input passages: 498
- Complete passages: 414
- Partial passages: 40
- LE parse failed passages: 44
- Stage 1A KVP rows: 8169
- Provenance entailment rows: 2644

## Partial Passage RCA

The 40 partial passages are KVP-expansion failures after successful LE extraction. They are not LE extraction failures.

Observed counts:

- Missing premise-level KVP rows: 85 total
- Partial passages with exactly 1 missing premise: 19
- Partial passages with 2 missing premises: 9
- Partial passages with 3 missing premises: 4
- Partial passages with 4 missing premises: 4
- Partial passages with 5 missing premises: 4
- Rows still produced from partial passages: 520
- Partial passages where batched KVP produced zero rows: 34
- Partial passages where fallback KVP produced at least one row: 37
- Partial passages where fallback KVP produced zero rows: 3

Likely cause:

- The batched KVP response was invalid, incomplete, duplicate-keyed, or omitted one or more requested `(entailment_index, premise_index)` pairs.
- The fallback per-premise KVP call then failed to recover every missing pair.
- The durable runner currently records `partial` as terminal, so these passages were not retried after the run stabilized.

The current artifact does not persist raw failed batched/fallback KVP responses, so we cannot prove whether each missed pair was caused by invalid JSON, schema mismatch, empty response, or semantic omission. That is an observability gap in the runner.

## LE Parse Failure RCA

The 44 `le_parse_failed` passages received non-empty model responses, but `parse_le_response` could not turn them into `LogEntailmentList`.

Response-size distribution for failed LE responses:

- `<1k` chars: 1
- `1-5k` chars: 12
- `5-10k` chars: 21
- `10-15k` chars: 5
- `15-20k` chars: 1
- `20k+` chars: 4
- Min chars: 23
- Max chars: 65932

Likely cause:

- Model output did not match the expected JSON object schema, despite being non-empty.
- Failure modes likely include malformed JSON, multiple JSON objects, a root array instead of `{"entailments": [...]}`, schema drift such as non-list `premises`, or output containing extra text that defeated the greedy JSON extraction.
- Higher temperature (`0.95`) and uncapped extraction increased output diversity and response length, which likely increased schema drift.

The current artifact stores `le_response_chars` but not the raw LE response, so exact payload-level failure classification is not possible after the fact. That is the primary RCA limitation.

## What Did Not Cause The Final Failures

- The final context-aware load-balanced run had `0` HTTP 429 responses.
- The final run had `7` local context-length HTTP 400 responses, but all were retried/recovered and did not directly create final `le_parse_failed` or `partial` statuses.
- The old entailment/premise caps were not active in this run.
- `max_premises_per_batch=64` was an operational chunk size, not a drop cap.

## Mitigations Before NeMo Microservices LE Run

1. Keep load balancing enabled:
   - Local service: `http://10.43.114.25:8000/v1`
   - External endpoint: `https://llm.example.com/v1`
   - Canonical row model: `nvidia/nvidia/nemotron-3-super-v3`
   - Local target model: `nvidia/nemotron-3-super-120b-a12b@32768`

2. Persist raw failure payloads:
   - Save raw LE parse-failed responses.
   - Save raw batched KVP parse-failed responses.
   - Save raw fallback KVP parse-failed responses.
   - Include passage ID, target endpoint/model, prompt type, attempt, and response hash.

3. Treat these statuses as retryable before declaring a corpus complete:
   - `le_no_response`
   - `le_parse_failed`
   - `partial`
   - `exception`

4. Add a repair path before final failure:
   - Try tolerant JSON extraction for root arrays and single entailment objects.
   - Optionally run a deterministic JSON repair prompt with `temperature=0` for parse failures.
   - Re-parse repaired output before marking `le_parse_failed`.

5. Improve local context routing:
   - Use a more conservative prompt-token estimator for local 32k NIM.
   - If local returns a context-length 400, retry that same call on the external endpoint immediately instead of allowing another local attempt.

6. Recover partials after the main pass:
   - Build a missing-pair retry list from partial passage result records.
   - Retry only missing `(passage_id, entailment_index, premise_index)` pairs.
   - Append recovered rows and update the result record before downstream Stage 1B/Curator work.

## Super Replica Preflight

The local Nemotron 3 Super deployment was scaled from 1 to 2 replicas before the NeMo Microservices run.

- Deployment: `runai-rag/nim-llm-super-120b-bw`
- Status after scale-up: `2/2` ready
- Service ClusterIP: `10.43.114.25`
- Ready pod endpoints:
  - `10.42.0.15:8000`
  - `10.42.0.16:8000`
- Both pod `/v1/models` endpoints report `nvidia/nemotron-3-super-120b-a12b` with `max_model_len=32768`.

Cleanup performed to make room:

- Removed stale failed `rag-nv-ingest-blackwell` pod objects from a deployment already scaled to `0/0`.
- Scaled noncritical `nrl265-durable-fused` application deployments to `0`.
- Removed stale failed `nrl265-durable-isolated` caption pods.
- Removed completed `nvidia-cuda-validator` pod objects.

No NeMo Microservices LE run was started as part of this RCA/preflight step.
