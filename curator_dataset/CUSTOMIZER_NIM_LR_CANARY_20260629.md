# Curator NIM r16 learning-rate canary

The NeMo Microservices five-epoch jobs were cancelled before their first
validation checkpoint. Their GPU workers and training CRs were removed.

Two controlled NIM Curator r16 jobs now test reduced learning rates. All other
training fields match the parent Llama 3.1 8B recipe, except the run is limited
to two epochs to avoid repeating the observed five-epoch overfit.

| Learning rate | Job | Epochs | State |
| ---: | --- | ---: | --- |
| `5e-5` | `cust-9zY9FLQbWpAxb2MVoSeteG` | 2 | Running on Blackwell |
| `6.8e-5` | `cust-Q6ASpeDuedSoBAXsdwfJsH` | 2 | Running on Blackwell |

Controlled settings: Llama 3.1 8B `+80GB`, r16/alpha32, batch 16,
micro-batch 1, warmup 30, cosine AdamW, packing disabled, seed 42, and the same
NIM Curator training/validation entity.

Reference result: the prior `1e-4` r16 job's best validation loss was
`1.4240126609802246` at its second validation checkpoint.

Records:

- `data/customizer/runs/curator-nim-r16-lr-canary-20260629/training_manifest.json`
- `data/customizer/runs/curator-nim-r16-lr-canary-20260629/training_timing.json`
- `data/customizer/runs/curator-nim-r16-lr-canary-20260629/best_checkpoint_promotion.json`
