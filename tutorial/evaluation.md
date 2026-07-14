# Evaluation Tutorial Notes

The evaluation tutorial covers two compatible paths.

## NeMo Evaluator Matrix

The matrix has three waves:

| Wave | Comparison | Purpose |
|---|---|---|
| A | single-axis rubric over each LoRA and base target | absolute quality scores |
| B | base-vs-adapter and adapter-vs-adapter within a corpus | prove adapter lift and rank variants |
| C | 49B comparator vs each LoRA | compare smaller adapted targets to a stronger reference |

The notebook uses the pure builder functions in `scripts/eval/run_evaluation_matrix.py` to inspect job counts and payload shape without submitting jobs.

## Durable Completions First

For direct judge evaluation, collect model completions once and score those saved artifacts. This avoids regenerating answers every time a judge prompt, rubric, or aggregation changes.

Collection command shape:

```bash
python scripts/eval/collect_completions.py   --dataset <DATASET_ROOT>/nim_curated/test_set_with_context.jsonl   --model default/lora-nim-llama-3.2-3b-r16   --run-id <run-id>   --max-tokens 8192
```

## Direct Judge Fallback

Use direct single-axis and pairwise scripts when the Evaluator service cannot reach the external judge endpoint or when you need a repeatable offline scoring pass over saved completions.

Single-axis command shape:

```bash
python scripts/eval/run_direct_kimi_singleaxis.py   --responses <EVAL_ROOT>/completions/.../responses.jsonl   --eval-run-id <run-id>   --limit 10
```

Pairwise command shape:

```bash
python scripts/eval/run_direct_kimi_pairwise.py   --pair base /path/to/base/responses.jsonl lora-r16 /path/to/lora/responses.jsonl   --eval-run-id <run-id>   --limit 10
```

## Token Accounting

The project tracks target generation and judge scoring independently:

- `target_generation`: serving cost of the base or adapter being evaluated.
- `judge_scoring`: cost of the external judge used for analysis.
- `combined_total_tokens_raw`: full experiment cost.

Use target-generation totals for deployment ROI decisions. Use judge-scoring totals for evaluation budget planning.
