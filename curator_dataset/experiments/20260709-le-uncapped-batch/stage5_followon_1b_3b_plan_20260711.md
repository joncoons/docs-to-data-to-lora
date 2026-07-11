# Stage 5 Follow-on 1B/3B LoRA Plan - 2026-07-11

## Current State

The 8B LE LoRA SFT jobs are already submitted on the Blackwell node
`ubuntu-local-dev` using `meta/llama-3.1-8b-instruct@v1.0.0+80GB` with
single-GPU LoRA SFT (`num_gpus=1`, `tensor_parallel_size=1`,
`data_parallel_size=1`). The r16 jobs are running first and the r32 jobs are
pending for Blackwell GPU capacity.

No 1B jobs should be submitted until the 8B jobs are complete, so the existing
8B r16 -> r32 queue remains undisturbed.

## 1B Follow-on Training

Target node: `ubuntu-local-dev` Blackwell GPUs.

Recommended template: `meta/llama-3.2-1b-instruct@v1.0.0+80GB`.

The 1B `+80GB` and `+40GB` templates are both single-GPU LoRA SFT in the running
Customizer config. Use `+80GB` on Blackwell for consistency with the 8B
Blackwell run unless an experiment explicitly wants the 40GB template.

Planned jobs after 8B completion:

| Corpus | Dataset entity | Rank | Output model entity |
| --- | --- | ---: | --- |
| NIM | `default/stage3-nim-curated-le-super-v3` | 16 | `default/lora-nim-le-super-v3-e5-llama-3.2-1b-r16-20260711` |
| NeMo Microservices | `default/stage3-nemo-usvcs-curated-le-super-v3` | 16 | `default/lora-nemo-usvcs-le-super-v3-e5-llama-3.2-1b-r16-20260711` |
| NIM | `default/stage3-nim-curated-le-super-v3` | 32 | `default/lora-nim-le-super-v3-e5-llama-3.2-1b-r32-20260711` |
| NeMo Microservices | `default/stage3-nemo-usvcs-curated-le-super-v3` | 32 | `default/lora-nemo-usvcs-le-super-v3-e5-llama-3.2-1b-r32-20260711` |

## 3B Ada Training

Target node: `ubuntu2`, which is the Ada node:

- GPU family: `ada-lovelace`
- GPU product: `NVIDIA-RTX-6000-Ada-Generation-SHARED`
- Physical GPU count: 2
- Current advertised capacity before any time-slicing change: 10 logical GPU
  slots because `timeSlicing.replicas: 5` is configured for `ubuntu2`.

Recommended template: `meta/llama-3.2-3b-instruct@v1.0.0+40GB`.

Use the 3B `+40GB` template for Ada. It is single-GPU LoRA SFT:
`num_gpus=1`, `tensor_parallel_size=1`, and `data_parallel_size=1`. Do not use
`meta/llama-3.2-3b-instruct@v1.0.0+80GB` for this run unless we intentionally
want DP5; the live Customizer config reports that template as `num_gpus=5` and
`data_parallel_size=5`.

Before submitting 3B, either temporarily remove/reduce time-slicing for
`ubuntu2` so Kubernetes advertises 2 GPU slots, or run only one/two jobs at a
time and accept that the device plugin may still place logical GPUs on shared
physical devices. For apples-to-apples adapter training, prefer temporarily
removing time-slicing on `ubuntu2`, restarting the Ada device-plugin/GFD pods,
and verifying `nvidia.com/gpu` capacity drops from 10 to 2.

Ada GPU cleanup completed before this plan:

| Deployment | Namespace | Desired replicas after cleanup |
| --- | --- | ---: |
| `nemoretriever-embedding-ms` | `runai-rag` | 0 |
| `nemoretriever-graphic-elements-v1` | `runai-rag` | 0 |
| `nemoretriever-ocr-v1` | `runai-rag` | 0 |
| `nemoretriever-page-elements-v3` | `runai-rag` | 0 |
| `nemoretriever-table-structure-v1` | `runai-rag` | 0 |

Verification after cleanup:

- `ubuntu2` GPU-requesting pods: none
- `ubuntu2` allocated `nvidia.com/gpu`: `0`
- stale non-running pods on `ubuntu2`: none

Planned 3B jobs, after the time-slicing decision:

| Corpus | Dataset entity | Rank | Output model entity |
| --- | --- | ---: | --- |
| NIM | `default/stage3-nim-curated-le-super-v3` | 16 | `default/lora-nim-le-super-v3-e5-llama-3.2-3b-r16-20260711` |
| NeMo Microservices | `default/stage3-nemo-usvcs-curated-le-super-v3` | 16 | `default/lora-nemo-usvcs-le-super-v3-e5-llama-3.2-3b-r16-20260711` |
| NIM | `default/stage3-nim-curated-le-super-v3` | 32 | `default/lora-nim-le-super-v3-e5-llama-3.2-3b-r32-20260711` |
| NeMo Microservices | `default/stage3-nemo-usvcs-curated-le-super-v3` | 32 | `default/lora-nemo-usvcs-le-super-v3-e5-llama-3.2-3b-r32-20260711` |
