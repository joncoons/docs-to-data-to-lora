# Customizer NFS output-copy deadlock — observed Round 2 r=32; future-work backlog

**Date observed**: 2026-05-26 (Round 2 of MoE Task 4.6).
**Status**: blocking parallel-shard runs at r=32; mitigated operationally by serial-shard submission.

## What happened

Round 2 of the MoE training matrix (`nim_curated × Nemotron-3-Nano-30B-A3B × r=32 × {shard-a, shard-b}`)
hung at the **post-training output copy** phase. Both worker pods finished training
(`max_steps=580 reached`) within ~90 seconds of each other; shard-a then completed its copy
from `/scratch/customizer_output_checkpoints/` to `/pvc/<cust-id>/trained/` (writing 1.64 GB
to the NFS-backed `peft-workspace-pvc`); shard-b never wrote anything. Both worker pods then
sat `Running` with ~0 CPU for hours. The Customizer controller polled both jobs every ~12s
but could not progress them — `EntityHandler_1` (the post-training data-store uploader) never
spawned because the worker pods never exited.

Round 1 (r=16) with the same parallel-shard layout had succeeded earlier the same day. The
difference: r=16 shard outputs are ~885 MB each (1.77 GB concurrent NFS write); r=32 is
~1.64 GB each (3.30 GB concurrent NFS write). The doubled concurrent write tipped the NFS
mount over into lock-contention deadlock.

## Why the prior v3 fix did not cover this

The v3 NFS-deadlock work captured in `NEMOTRON_NANO_LORA_METHODOLOGY.md §2`
moved Lightning's mid-training checkpoint
target via `training.workspace_dir: /pvc/workspace → /scratch/workspace`. That fix is in
place — we verified the live ConfigMap has `workspace_dir: /scratch/workspace`. But the
post-training **final output** copy is a separate code path
(`customizer_training/utils/file_structure.py`) that still writes the trained adapter from
`/scratch/customizer_output_checkpoints/` to `/pvc/<cust-id>/trained/`. The original v3 work
noted this: *"OUTPUT_MODEL_PATH stays on NFS so the final adapter still lands on
<ARTIFACT_ROOT>/ atomically at job end."* That was acceptable because v3 trained one shard
at a time per pair (and the output was smaller). At r=32 with both shards racing, it isn't.

## Recommended fixes

> Keep the single-GPU + TIES approach. It is the right direction for this model. Do not
> go back to the DP2/EP2 NCCL template unless you want to debug a separate FSDP-MoE
> checkpoint-gather issue.
>
> **Fix storage/publication:**
>
> 1. **Move final OUTPUT_MODEL_PATH off NFS too, not just workspace_dir.** Best fix: patch
>    the Customizer/NemoTrainingJob output path so export lands on `/scratch/trained`, then
>    upload from local scratch to Data Store before pod exit.
> 2. **If Customizer requires `/pvc` for the model uploader**, use a real local or per-job
>    PVC for training outputs, not the shared NFS export. A per-job local-path/RWO PVC on
>    the GPU node is safer than shared RWX NFS for these writes.
> 3. **If you cannot patch that yet, serialize finalization.** Run shard A and shard B
>    sequentially for Rank 32, or add a Kubernetes Lease / API-level lock around the final
>    copy/upload step. **Do not use an NFS file lock for this.**
> 4. **Treat current pods as wedged.** If you need to recover artifacts, try node-local
>    recovery before rebooting:
>    - shard A pod UID: `a83ca1eb-4097-4e9a-b374-cf60860a8777`
>    - shard B pod UID: `c7bfdaa4-9f86-4059-b448-d6a6e7536f09`
>    - likely host path:
>      `/var/lib/kubelet/pods/<uid>/volumes/kubernetes.io~empty-dir/scratch/`
>    This needs root on `<BLACKWELL_NODE>`. Without recovery, a kubelet/containerd restart
>    or node reboot may be required.
> 5. **Stay with `alpha = rank` for Nemotron Nano MoE.** Rank 32 should be `alpha=32`, not
>    `64`. Rank 32 also roughly doubles adapter artifact size, which makes the NFS
>    final-copy problem much easier to trigger.
>
> NVIDIA's docs align with what we're seeing: Customizer uses PVCs for model/workspace
> data, and the Helm defaults put `training.workspace_dir` at `/pvc/workspace`; the
> operator docs also describe Customizer-created PVCs for training/model data and RWX
> workspace requirements. In this cluster, the workload pattern is simply too write-heavy
> and failure-sensitive for the current NFS-backed `/pvc` output path.

### Sources cited

- https://docs.nvidia.com/nim-operator/latest/customizer.html
- https://docs.nvidia.com/nemo/microservices/25.9.0/helm/index.html

## Future work (deferred)

In priority order:

1. **Patch Customizer source to direct final output to `/scratch/trained`**. Modify
   `customizer_training/utils/file_structure.py` so the
   `Copying /scratch/customizer_output_checkpoints/<file> to /pvc/<cust-id>/trained/<file>`
   pattern targets `/scratch/trained/<file>` instead. Then add a post-export upload step
   that pushes from `/scratch/trained/` directly to Data Store via the HF API, eliminating
   the NFS write entirely. Customizer container image rebuild required.

2. **Per-job local-path PVCs** instead of the shared RWX NFS `peft-workspace-pvc`.
   k3s/RKE2 `local-path` provisioner on the GPU node would give each job its own
   per-pod-lifetime PVC with no cross-pod contention. Requires changes to the Customizer
   NemoTrainingJob CR (PVC name / storageClass).

3. **Cross-pod K8s Lease around the final copy phase**. Cheaper than a Customizer image
   rebuild — could be added as a small sidecar that acquires
   `coordination.k8s.io/Lease` "moe-output-copy" before letting the worker proceed past
   `Trainer.fit stopped`. Tradeoff: serializes finalization but parallelism during
   training is preserved.

4. **Workspace recovery tooling**. A small script to pull adapter artifacts directly out
   of a wedged pod's emptyDir at
   `/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~empty-dir/scratch/customizer_output_checkpoints/`
   and upload them to data-store. Saves a 3-hour re-run when a pod wedges late.

## Current workaround (in use)

**Serial shard submission** for all remaining MoE rounds. Submit shard-a, wait for full
terminal (incl. EntityHandler_1 upload), then submit shard-b, wait for terminal, then TIES
merge. Doubles per-round wall-clock (from ~3h to ~6h) but eliminates the cross-shard NFS
write contention. No code changes required.

This unblocks Stage 3 Task 4.6 without taking on the Customizer source patch, which can be
prioritized as a future-work item if MoE training becomes part of routine workflow on this
cluster.

## Final methodology (2026-05-27, post-reboot)

After the original NFS wedge required a node reboot and the resumed run surfaced two
further issues, the Stage 3 MoE training plan converged on the following recipe. **Use
this as the canonical Nemotron-3-Nano-30B-A3B MoE LoRA workflow on this hardware**; the
parallel interleave from [`INTERLEAVED_GPU_SCHEDULING.md`](INTERLEAVED_GPU_SCHEDULING.md)
is the dense-Llama default but is unsafe here.

**Recipe:**

| Knob | Value | Rationale |
|---|---|---|
| ranks | **r=16 only** | r=32 with α=rank=32 diverged at end-of-warmup (loss → 9, grad_norm → 1632 at step 18 of `cust-U9RnGyiJTxYzGS1Kxrv4bD`). 2× trainable params + compressed warmup = unstable. |
| α | rank (α/r = 1.0) | Validated for nim r=16. α/r=2.0 also diverges (2026-05-19 incident). |
| lr | 1e-4 | Validated for r=16. If r=32 ever needed, drop to 5e-5 and re-validate. |
| warmup_steps | **20** uniform | Customizer 25.12 enforces `warmup_steps < lr_decay_steps`. Stage 3 MoE shards yield 62-72 optimizer steps over 2 epochs with grad_acc=8. Was 100 (silently clipped before this validation tightened). |
| batch_size | 8 | MoE default; not 16 like dense Llama. |
| epochs | 2 | |
| sequence_packing_enabled | false | MoE + Blackwell sm_120 constraint. |
| template | `nvidia/nemotron-3-nano-30b-a3b@v1.0+96GB-singleGPU` | Avoids the FSDP-MoE NCCL gather wedge from the older DP2/EP2 template. |

**Submission pattern:**

- **Fully serial**, one shard at a time, one GPU at a time.
  - Do **not** dual-lane (r=16 + r=32 on two GPUs) — there is no r=32 lane.
  - Do **not** parallel-shard within a rank (shard-a + shard-b at the same time) —
    concurrent 0.9-1.6 GB NFS post-training writes still risk the wedge even at r=16.
- After both shards reach `status=completed` and the EntityHandler upload finishes,
  TIES-merge a + b with `trim_ratio=0.2`.
- Canonical driver: [`moe_serial_r16_orchestrator.py`](moe_serial_r16_orchestrator.py).

**Wall-clock**: ~90 min per shard × 2 shards = ~3h per corpus, plus ~30s TIES merge.

**Verifying success — don't trust `job.status` alone:**

- The Customizer list endpoint (`GET /v1/customization/jobs`) returns
  `train_loss=null val_loss=null` even for completed jobs; the per-job endpoint
  (`GET /v1/customization/jobs/{id}`) returns the actual loss values.
- A worker pod can sit in container state `1/1 Running` indefinitely after the
  training process has exited with a `ConfigurationException` at second 0. Customizer
  takes minutes to reconcile this into `job.status=failed`. Confirm by greping pod
  logs for `ConfigurationException` or `step [0-9]`.
