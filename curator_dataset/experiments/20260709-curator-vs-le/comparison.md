# Curator vs Logical Entailment Comparison - 2026-07-09

Scope: Llama 3.1 8B LoRA SFT on the same two documentation corpora: NIM and NeMo Microservices. This comparison captures the current state before running any Nemotron 3 Ultra 550B smoke test.

## Bottom line

The available validation-loss evidence favors the logical entailment (LE) dataset path across both corpora and both LoRA ranks. Curator's best checkpoints are higher loss in every matched corpus/rank slice, with the gap especially large on the NeMo Microservices corpus.

Important caveat: the strict five-epoch LE jobs did complete, but retained local artifacts and Customizer API status no longer expose their validation-loss values. The loss-bearing LE baseline in the repo is the earlier two-epoch table from `evals/training_session.log`. The Curator values below are five-epoch validation-best checkpoints, so this is a practical state comparison, not a perfectly epoch-matched result.

## Loss-bearing comparison

Lower validation loss is better.

| Corpus | Rank | LE val loss, 2 epochs | Curator best val loss, 5 epochs | Curator minus LE | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| NIM | r16 | 1.311 | 1.42401 | +0.11301 | +8.6% |
| NIM | r32 | 1.269 | 1.42204 | +0.15304 | +12.1% |
| NeMo Microservices | r16 | 0.991 | 1.45658 | +0.46558 | +47.0% |
| NeMo Microservices | r32 | 0.947 | 1.41646 | +0.46946 | +49.6% |

## Current Curator results

| Corpus | Rank | Job ID | Dataset rows | Best val loss | Best step | Final/late val loss | Output model revision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| NIM | r16 | `cust-6cpWwCz38TzoDCmZcMJK6U` | 6845 train / 815 val | 1.42401 | 853 | n/a | `default/lora-nim-curator-e5-llama31-8b-r16-20260629@cust-6cpWwCz38TzoDCmZcMJK6U` |
| NIM | r32 | `cust-EqsxmYZskEV95oRQzqjZRA` | 6845 train / 815 val | 1.42204 | 426 | n/a | `default/lora-nim-curator-e5-llama31-8b-r32-20260629@cust-EqsxmYZskEV95oRQzqjZRA` |
| NeMo Microservices | r16 | `cust-CiQrMdjTEBqr73zvLXzM7o` | 3865 train / 480 val | 1.45658 | 241 | 2.199 | `default/lora-nemo-ms-curator-e5-llama31-8b-r16-bw-20260709@cust-CiQrMdjTEBqr73zvLXzM7o` |
| NeMo Microservices | r32 | `cust-XJ7f4T6DF8oRQzcukph5Kb` | 3865 train / 480 val | 1.41646 | 241 | 2.215 | `default/lora-nemo-ms-curator-e5-llama31-8b-r32-bw-20260709@cust-XJ7f4T6DF8oRQzcukph5Kb` |

The NeMo Curator jobs completed sequentially on the Blackwell node and used the HF-style dataset entity backed by the Data Store Git/LFS compatibility shim. Both jobs reached their best validation loss at the first validation checkpoint and were not top-1 by the final checkpoint, indicating overfit under the five-epoch schedule.

## LE state for the same corpora

The repo contains two LE result layers:

1. Loss-bearing two-epoch runs from `evals/training_session.log`:
   - NIM r16/r32: 1.311 / 1.269 validation loss on 4631 train / 242 validation rows.
   - NeMo Microservices r16/r32: 0.991 / 0.947 validation loss on 3957 train / 208 validation rows.
2. Five-epoch 8B LE jobs from `docs/experiments/8b-nonaug-5epoch-training-20260609.md` and the current Customizer API:
   - NIM r16/r32 and NeMo Microservices r16/r32 all completed.
   - API status retains completion, step, and best-epoch metadata, but `best_val_loss` is null for all four and old worker pods/logs are not available.

## Interpretation

Curator is not beating LE on the available validation-loss evidence. The gap is modest on NIM (+8.6% to +12.1% higher loss) and severe on NeMo Microservices (+47.0% to +49.6% higher loss). Rank 32 is slightly better than rank 16 for Curator, but the rank change does not close the LE gap.

Generation provenance note: actual Curator manifests record `nvidia/nvidia/nemotron-3-super-v3`; the catalog shorthand `nvidia/nemotron-3-super-v3` is treated as exact-equivalent to `nvidia/nemotron-3-super-120b-a12b` for this project.

This supports testing the Nemotron 3 Ultra 550B hypothesis as a dataset-quality experiment, but the smoke test should be source-fixed and small. Running the whole corpus first would hide whether improvements come from generation quality, row mix, epoch schedule, or validation split differences.

## Artifacts

- Data: `curator_dataset/experiments/20260709-curator-vs-le/metrics.json`
- Chart: `curator_dataset/experiments/20260709-curator-vs-le/comparison.svg`
- NeMo Curator manifest with worker-log best losses: `curator_dataset/data/customizer/runs/nemo-ms-curator-seq-e5-8b-20260709/training_manifest.json`

## Sources

- `evals/training_session.log`
- `docs/experiments/8b-nonaug-5epoch-training-20260609.md`
- `curator_dataset/data/customizer/runs/curator-parent-matched-e5-8b-20260629/best_checkpoint_promotion.json`
- `curator_dataset/data/customizer/runs/nemo-ms-curator-seq-e5-8b-20260709/training_manifest.json`
- `curator_dataset/data/customizer/registration/nim/metrics.json`
- `curator_dataset/data/customizer/registration/nemo_usvcs/metrics.json`
