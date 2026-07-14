# Execution Modes

The tutorial supports two operating modes.

## Local And Dry Run

Use this mode first. It exercises import paths, schema builders, curation logic, JSONL formatting, adapter payload construction, and evaluation matrix planning without calling cluster services.

Typical settings in notebooks:

```python
RUN_LIVE = False
SUBMIT_CUSTOMIZER_JOB = False
SUBMIT_EVALUATOR_JOBS = False
```

Typical CLI options:

```bash
python scripts/build_v2_dataset.py --collection nim_curated --output /tmp/nim --dry-run
python scripts/stage3/train_adapter.py --collection nim_curated --base-model meta/llama-3.2-3b-instruct --rank 16 --dry-run
python scripts/eval/run_evaluation_matrix.py --wave A --limit-jobs 1 --submit-only
```

## Live Infrastructure

Live mode requires service reachability and credentials. The defaults in the repo assume in-cluster service DNS. Override these when running notebooks from a workstation.

| Area | Environment variables or CLI options |
|---|---|
| Elasticsearch corpus | `PIPELINE_ES_HOST`, `PIPELINE_ES_USER`, `PIPELINE_ES_PASSWORD` |
| Generation NIM for Stage 2 | `PIPELINE_NIM_ENDPOINTS` |
| External validation judge | `PIPELINE_EXTERNAL_JUDGE_BASE`, `PIPELINE_EXTERNAL_JUDGE_MODEL` |
| Customizer | `--customizer-url` |
| Data Store / Entity Store | `DATA_STORE_URL`, `DATA_STORE_GIT_BASE`, `ENTITY_STORE_URL` |
| Evaluator | `EVALUATOR_URL`, `EVALUATOR_API_KEY` |
| Completion collection | `TARGET_API_URL`, `COMPLETIONS_OUTPUT_ROOT` |
| Direct judge fallback | `JUDGE_API_URL`, `EVALUATOR_JUDGE_MODEL`, `JUDGE_API_KEY_ENV` |

Before switching a notebook flag to live mode, run the corresponding dry-run cell and inspect the command or payload. This keeps expensive actions such as Customizer training and judge sweeps explicit.

## Suggested Live Sequence

1. Build a curated corpus with the crawler.
2. Run `scripts/build_v2_dataset.py` for one collection with `--max-passages` for a smoke test.
3. Register the resulting train, validation, and test datasets.
4. Submit one LoRA adapter job with `--wait` only after the dry-run payload is correct.
5. Collect completions for base and adapter targets.
6. Run direct single-axis or pairwise evals on saved completions.
7. Roll up token usage and compare target serving cost separately from judge cost.
