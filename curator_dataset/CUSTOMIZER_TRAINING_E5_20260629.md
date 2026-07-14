# Curator dataset Customizer e5 training — 2026-06-29

This five-epoch run supersedes the stopped two-epoch pilot recorded in `CUSTOMIZER_TRAINING_20260629.md`.

## Policy

- Train all jobs for a maximum of 5 epochs.
- Do not terminate early.
- Use validation loss to identify the best epoch.
- Promote and verify the Customizer output-model revision corresponding to the validation-best checkpoint.
- Record both active GPU time and Customizer end-to-end time.

The historical five-epoch comparison jobs used the same pattern: `epochs: 5`, no early-stopping hyperparameter, and validation-best checkpoint tracking through `status_details.best_epoch` and validation loss.

## Jobs

| Corpus | Rank | Job ID | Output model | Initial worker state |
| --- | ---: | --- | --- | --- |
| NIM | 16 | `cust-AvPjLCJsBhsi3fdLCFFCzA` | `default/lora-nim-curator-nemotron-nano-30b-r16-e5-20260629` | Running on one Blackwell GPU |
| NIM | 32 | `cust-HvcnhuFnT1yvPa8umy9ngJ` | `default/lora-nim-curator-nemotron-nano-30b-r32-e5-20260629` | Running on one Blackwell GPU |
| NeMo Microservices | 16 | `cust-DNU5tkRhPfHzjbq1sShngX` | `default/lora-nemo-usvcs-curator-nemotron-nano-30b-r16-e5-20260629` | Queued for one Blackwell GPU |
| NeMo Microservices | 32 | `cust-MZWf4RV6ykPwRXKHXo3msS` | `default/lora-nemo-usvcs-curator-nemotron-nano-30b-r32-e5-20260629` | Queued for one Blackwell GPU |

All four jobs use:

- `nvidia/nemotron-3-nano-30b-a3b@v1.0+96GB-singleGPU`
- 5 epochs
- batch size 8 and micro batch size 1
- learning rate 1e-4
- 20 warmup steps
- AdamW with cosine annealing
- LoRA alpha equal to rank
- sequence packing disabled

The NIM dataset produces 856 optimizer steps per epoch, or 4,280 maximum steps over five epochs.

## Datasets

- `default/stage3-nim-curated-curator-diverseqa-20260629`
- `default/stage3-nemo-usvcs-curated-curator-diverseqa-20260629`

Both are registered as HF datasets backed by NeMo Data Store Git/LFS repositories.

## Tracking and promotion artifacts

Artifacts are written under:

`data/customizer/runs/curator-diverseqa-e5-20260629/`

- `training_manifest.json`: immutable job identities and configuration intent.
- `gpu_timing.json`: active worker timing and queue delay.
- `gpu_timing_events.jsonl`: worker phase transitions.
- `checkpoint_promotion.json`: best epoch, best validation loss, output artifact status, and promotion verification.
- `timing_watcher.pid`: present while timing collection is active.
- `checkpoint_watcher.pid`: present while best-checkpoint verification is active.

Both watchers exit automatically after all four jobs are terminal and completed artifacts are verified, or terminal failures are recorded.

## Stopped pilot

The original e2 jobs were cancelled before replacement submission. Customizer cancellation did not terminate the two active Volcano workers, so their underlying Volcano Jobs were explicitly deleted before the e5 run. The cancelled pilot job IDs and partial timing artifacts remain in `data/customizer/runs/curator-diverseqa-20260629/` for auditability and must not be treated as adapter results.
