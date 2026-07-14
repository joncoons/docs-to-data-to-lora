# Curator Customizer training: e5 fix2

## Outcome

The four-adapter matrix was relaunched on 2026-06-29 with five epochs, no
early stopping, and validation-best checkpoint promotion tracking.

| Corpus | Rank | Customizer job | Initial GPU state |
| --- | ---: | --- | --- |
| NIM | 16 | `cust-5HhTkEYiKpPvLAnRXJS62w` | Running on `<BLACKWELL_NODE>` |
| NIM | 32 | `cust-MPT4gXsS5wWpLwcz2J3Dhx` | Running on `<BLACKWELL_NODE>` |
| NeMo Microservices | 16 | `cust-SUFrUtLpBiduPqnma44F7t` | Queued for `<BLACKWELL_NODE>` |
| NeMo Microservices | 32 | `cust-8c4VynVebA9CCjwH5p1kHE` | Queued for `<BLACKWELL_NODE>` |

The two NIM jobs consume the two physical Blackwell GPUs. The NeMo jobs are
rendered and pending with `Insufficient nvidia.com/gpu`; Volcano will admit
them as the NIM jobs release GPUs.

## Root causes and fixes

1. Customizer's `training.nodeSelectors` is copied into the dataset handler,
   GPU trainer, and output uploader. Pointing it at the Blackwell node made
   CPU-only handler pods compete with that node's pod capacity.
2. Pointing the same global selector at `<ADA_NODE>` also moved GPU trainers to
   that node, because it overrides `training.container_defaults.nodeSelector`.
3. Correct split placement was produced by pausing NeMo operator
   reconciliation, submitting each pair with handlers rendered for `<ADA_NODE>`,
   patching only
   `spec.trainingWorkload.trainerOverrides.spec.nodeSelector` to
   `<BLACKWELL_NODE>`, and then resuming the operator.
4. NeMo output model names initially failed API validation because the full
   name plus generated `@cust-*` revision exceeded Customizer's 96-character
   limit. The output corpus token was shortened from `nemo-usvcs` to `nemo-ms`.

Handler Jobs are intentionally retained after successful completion; deleting
them can cause NeMo operator reconciliation and retry behavior.

## Tracking

- Manifest: `data/customizer/runs/curator-diverseqa-e5-fix2-20260629/training_manifest.json`
- GPU and end-to-end timing: `data/customizer/runs/curator-diverseqa-e5-fix2-20260629/training_timing.json`
- Best-checkpoint verification: `data/customizer/runs/curator-diverseqa-e5-fix2-20260629/best_checkpoint_promotion.json`
- Watcher state: `data/customizer/runs/curator-diverseqa-e5-fix2-20260629/watchers/`

The timing watcher records submission-to-GPU-start, queue time, container
start/finish, GPU training seconds, and Customizer elapsed time. The checkpoint
watcher records completed epochs, validation loss, `best_epoch`, artifact URL,
and upload status. Promotion is considered verified only when Customizer is
complete, `best_epoch` is present, and Entity Store reports
`upload_completed`.

## Initial health evidence

- NIM handlers completed once on `<ADA_NODE>`.
- Both NIM trainers entered optimizer steps with 856 steps per epoch and 4,280
  maximum steps over five epochs.
- NIM r=16 uses approximately 65.8 GiB GPU memory; r=32 uses approximately
  69.9 GiB.
- Both NeMo handlers completed once on `<ADA_NODE>`.
- Both NeMo trainer pods have the `<BLACKWELL_NODE>` selector and are pending
  only because both physical GPUs are allocated.
