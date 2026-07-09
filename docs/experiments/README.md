# Experiments

Focused ablation and extension plans for the docs-to-data-to-lora pipeline.

| Plan | Purpose |
|---|---|
| [Local findings](local-findings.md) | Local-only observations from live experiments that should not be promoted without review. |
| [NIM 1B Data Designer augmentation](nim-1b-data-designer-augmentation.md) | Test whether NeMo Data Designer synthetic rows improve 1B LoRA adapters trained from the grounded NIM dataset. |
| [Deferred full-seed dense augmentation](nim-1b-data-designer-augmentation.md#deferred-follow-up-full-seed-dense-augmentation) | Later-phase plan for canonical full-seed Data Designer pools and model-sized synthetic mixes after the current eval matrix finishes. |

| Experiment | Purpose |
|---|---|
| [Full-seed Kimi 5x augmentation](full-seed-kimi-5x-augmentation-20260604.md) | Active 5x NeMo Data Designer augmentation run for both NIM and NeMo Microservices grounded datasets using Kimi K2.6. |
| [8B non-augmented 5-epoch LoRA](8b-nonaug-5epoch-training-20260609.md) | Blackwell-only 5-epoch 8B LoRA SFT run on grounded non-augmented NIM and NeMo Microservices datasets. |
