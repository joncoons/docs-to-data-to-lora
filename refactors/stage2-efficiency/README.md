# Stage 2 Efficiency Refactor

This directory is an isolated workspace for testing Stage 1A efficiency changes
without changing the working `scripts/build_v2_dataset.py` example or deployed
K8s jobs.

## Decision

Use the conservative batch approach first:

```text
passage -> logical entailments -> batched QA rows
```

The aggressive one-call approach is parked. It collapses the audit edge between
entailment extraction and QA generation, so it is not the right first production
candidate for this pipeline.

## Goal

Reduce repeated LLM calls in Stage 1A while preserving the qualitative value and
provenance of the current logical-entailment QA generation path.

Compare two active arms:

1. **Baseline**: current `LE -> one KVP call per premise`.
2. **Conservative batch**: current LE extraction, followed by one batched KVP
   call per passage or capped premise group.

## Prototype Files

```text
refactors/stage2-efficiency/
  README.md
  .gitignore
  stage1a_batched_kvp.py
```

The prototype writes filenames that cannot overwrite production Stage 1A output:

```text
stage1a_le.batched_kvp.jsonl
provenance/entailments.batched_kvp.jsonl
```

Run outputs should stay under ignored `runs/` directories.

## Conservative Batch Shape

Batched KVP response:

```json
{
  "pairs": [
    {
      "entailment_index": 0,
      "premise_index": 1,
      "question": "...",
      "answer": "..."
    }
  ]
}
```

Validation rules:

- Drop pairs whose indexes do not map to an extracted entailment and premise.
- Drop pairs with empty question or answer.
- Dedupe normalized questions within the passage.
- Retry invalid batched responses according to `--batch-parse-attempts`.
- Fall back to the current per-premise KVP prompt for missing or invalid pairs.

## Minimal Experiment

Use a fixed, stratified sample from existing Stage 0 `passages.jsonl` files.

Sample shape:

- 30 passages from `nim_curated`.
- 30 passages from `nemo_usvcs_curated`.
- Include a mix of short HTML, long HTML, binary-derived/document passages, and
  several product families.
- Save selected passage IDs in this directory so every run uses the same input.

Suggested artifact layout:

```text
refactors/stage2-efficiency/runs/<run-id>/
  input_passage_ids.txt
  baseline/stage1a_le.jsonl
  conservative_batch/stage1a_le.batched_kvp.jsonl
  conservative_batch/provenance/entailments.batched_kvp.jsonl
  metrics.json
  judge_sample.jsonl
```

Do not overwrite the canonical dataset outputs.

## Metrics

Collect mechanical metrics first:

- LLM request count.
- Prompt tokens, completion tokens, and total tokens when available.
- Wall-clock runtime.
- Rows emitted per passage.
- Valid JSON parse rate.
- Fallback rate.
- Missing or invalid `entailment_index` / `premise_index` references.
- Duplicate question rate within a passage.

Collect quality metrics second:

- Grounding pass rate from the existing Stage 4 judge prompt.
- Answer specificity: answer includes source identifiers, commands, versions,
  parameters, config values, or product-specific names when present.
- Question usefulness: avoids yes/no and generic "what is X" patterns.
- Entailment alignment: question is traceable to the premise and answer is
  traceable to the conclusion/source text.
- Provenance completeness: every emitted row preserves the same source revision,
  chunk, modality, entailment, and premise fields expected by current Stage 1A.

## Acceptance Gates

The conservative batch can replace the current Stage 1A path only if:

- Grounding pass rate is no worse than baseline by more than 2 percentage points.
- Row yield is at least 90 percent of baseline on the fixed sample.
- Invalid alignment rate is below 2 percent.
- Duplicate question rate is no higher than baseline.
- Request count or total token count improves by at least 15 percent.
- All emitted rows remain valid `KVPRow` records.

## Implementation Boundary

Allowed imports from the working pipeline:

- `scripts.pipeline.models`
- `scripts.pipeline.prompts`
- `scripts.pipeline.provenance`
- `scripts.pipeline.provenance_io`
- `scripts.pipeline.llm_client`, only inside the CLI entrypoint

Do not change these files during the experiment:

- `scripts/build_v2_dataset.py`
- `scripts/pipeline/stage1a_le_kvp.py`
- `deploy/stage1a-entailment-shards/*`

If the variant passes the gates, promote it later as a new explicit Stage 1A
mode or v2 entrypoint rather than silently changing baseline behavior.

## Local Test

`pytest` is not installed in the current environment, so the tests can be run
through a direct plain-assert harness until the dev extras are available:

```bash
python3 - <<'PY'
from tests import test_stage2_efficiency_batched_kvp as t
for name in sorted(dir(t)):
    if name.startswith('test_'):
        getattr(t, name)()
        print(f'{name}: ok')
PY
```
