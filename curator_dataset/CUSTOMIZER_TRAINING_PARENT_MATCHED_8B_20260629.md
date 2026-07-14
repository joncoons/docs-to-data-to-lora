# Curator training: parent-matched Llama 3.1 8B e5

## Purpose

This run isolates dataset-generation method as the experimental variable. It
uses the Curator DiverseQA datasets but exactly matches the successful
five-epoch logical-entailment training recipe recorded by the parent project.

The preceding Nemotron Nano 30B run was cancelled because it did not match the
parent experiment's base model or effective training settings.

## Validated baseline

The replacement payload was checked against the live completed Customizer jobs
`cust-GQkpTXu3frr2PnWSnbcgKH`, `cust-EDmfZ5Hi9HaCw4Ko4wyLpH`,
`cust-CiTm8oforcC38Rw67D3RVH`, and `cust-XZMopAuFBH57vNpVJZzBvj`.

| Field | Value |
| --- | --- |
| Base/config | `meta/llama-3.1-8b-instruct@v1.0.0+80GB` |
| Context | 4096 |
| Precision | `bf16-mixed` |
| GPUs per job | 1 |
| Epochs | 5 |
| Batch size | 16 |
| Micro-batch size | 1 |
| Learning rate | `1e-4` |
| Warmup | 30 steps |
| Optimizer | `adamw_with_cosine_annealing` |
| Packing | disabled |
| LoRA | r16/alpha32 and r32/alpha64 |
| Checkpoint | validation-best export |

## Active matrix

| Corpus | Rank | Job | Initial state |
| --- | ---: | --- | --- |
| NIM | 16 | `cust-6cpWwCz38TzoDCmZcMJK6U` | Running on Blackwell |
| NIM | 32 | `cust-EqsxmYZskEV95oRQzqjZRA` | Running on Blackwell |
| NeMo Microservices | 16 | `cust-K7iHeRMzVr8XkEuoSSG4Nt` | Queued behind NIM |
| NeMo Microservices | 32 | `cust-6SLDu4SiecFtLggATJdNUN` | Queued behind NIM |

The first simultaneous four-job reconciliation admitted one NIM and one NeMo
job. Those NeMo attempts were cancelled before meaningful training, retained in
the manifest under `superseded_jobs`, and resubmitted with `-q1-` output names
to enforce the requested NIM-first order.

## Initial health

- Both NIM workers run on `ubuntu-local-dev`; handlers completed on `ubuntu2`.
- Both NeMo workers use the Blackwell selector and are pending for GPUs.
- NIM steps per epoch: 428; total steps: approximately 2,140.
- Initial steady-state step time: approximately 1.35 seconds.
- Initial GPU memory: approximately 20-24 GiB per worker.
- Loss remained stable through and after the 30-step warmup.

## Records

- `data/customizer/runs/curator-parent-matched-e5-8b-20260629/training_manifest.json`
- `data/customizer/runs/curator-parent-matched-e5-8b-20260629/training_timing.json`
- `data/customizer/runs/curator-parent-matched-e5-8b-20260629/best_checkpoint_promotion.json`
- `data/customizer/runs/curator-parent-matched-e5-8b-20260629/watchers/`
