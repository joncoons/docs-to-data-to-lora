# LE Downstream Dataset Summary

- Collection: `nemo_usvcs_curated`
- Output dir: `curator_dataset/experiments/20260709-le-uncapped-batch/runs/nemo_usvcs_curated/super-v3`
- Completed stages: `1b, 1c, 2, 3, 4`
- Dataset version: `not finalized`
- LLM temperature: `0.2`
- Stage 2 execution surface: `curator_llm_quality`
- Stage 2 max tokens: `2048`
- Source filter: `html`

## Synthesis Targets
- `https://inference-api.nvidia.com/v1` -> `nvidia/nvidia/nemotron-3-ultra` (`uncapped`)

## Stage 2 QA Targets
- `https://inference-api.nvidia.com/v1` -> `nvidia/nvidia/nemotron-3-super-v3` (`uncapped`)

## Source Filter
- Passages selected: 486 / 486
- Stage 1A rows selected: 9611 / 9611

## Counts
- passages: 486
- stage1a_rows: 9611
- stage1b_rows: 972
- stage1c_rows: 262
- stage1_5_rows: 0
- stage2_rows: 10712
- stage2_dropped_rows: 142
- training_rows: 7171
- validation_rows: 799
- stage4_sample_size: 100
- stage4_passed: True
- stage4_pass_rate: 0.99
