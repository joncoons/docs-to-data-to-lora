# LE Downstream Dataset Summary

- Collection: `nim_curated`
- Output dir: `curator_dataset/experiments/20260709-le-uncapped-batch/runs/nim_curated/super-v3`
- Completed stages: `1b, 1c, 2, 3, 4, finalize`
- Dataset version: `dsv_178223936b9918e346f3c4c5`
- LLM temperature: `0.2`
- Stage 2 execution surface: `curator_llm_quality`
- Stage 2 max tokens: `2048`
- Source filter: `html`

## Synthesis Targets
- `https://llm.example.com/v1` -> `nvidia/nvidia/nemotron-3-ultra` (`uncapped`)

## Stage 2 QA Targets
- `https://llm.example.com/v1` -> `nvidia/nvidia/nemotron-3-super-v3` (`uncapped`)

## Source Filter
- Passages selected: 430 / 498
- Stage 1A rows selected: 8438 / 9312

## Counts
- passages: 498
- stage1a_rows: 9312
- stage1b_rows: 858
- stage1c_rows: 252
- stage1_5_rows: 0
- stage2_rows: 9404
- stage2_dropped_rows: 149
- training_rows: 7905
- validation_rows: 880
- stage4_sample_size: 100
- stage4_passed: True
- stage4_pass_rate: 0.99
