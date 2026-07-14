# NeMo Customizer NFS Block Findings

Date: 2026-05-27

## Summary

The current Rank 32 Nemotron-3-Nano-30B-A3B failures do not look like an active NCCL problem. The live jobs were using the intended single-GPU Customizer template, completed training, then stalled during final artifact publication to the NFS-backed `/pvc` path.

The prior mitigation moved `training.workspace_dir` from `/pvc/workspace` to `/scratch/workspace`, which removed the mid-training checkpoint path from NFS. That fix is present in the live cluster. The remaining problem is that `OUTPUT_MODEL_PATH` still points to `/pvc/<job>/trained`, so Customizer still copies the final exported adapter from node-local scratch back to NFS at job end.

For Rank 32, the final `adapter_model.safetensors` is substantially larger than Rank 16, and both shard pods attempted final publication at roughly the same time. That appears to be enough to wedge the NFS/client/container runtime path.

## Live Cluster Evidence

Customizer ConfigMap:

- `nvidia/nemotron-3-nano-30b-a3b@v1.0+96GB-singleGPU` is present.
- The single-GPU template uses:
  - `num_gpus: 1`
  - `num_nodes: 1`
  - `tensor_parallel_size: 1`
  - no `expert_model_parallel_size`
- `training.workspace_dir` is set to `/scratch/workspace`.

Worker pod config:

- Shard A job: `cust-RbCxPuSgJ7Pmwj4KW77MzM`
- Shard B job: `cust-HJLGvuN6ckQ1ADEFgjzBJV`
- Both pods were scheduled to `<BLACKWELL_NODE>/192.168.0.169`.
- Both pods mounted:
  - `/scratch` as `emptyDir`
  - `/pvc` from `peft-workspace-pvc`
  - `/mount/models` from `peft-models-pvc` read-only
- Both pod configs show:
  - `num_gpus: 1`
  - `data_parallel_size: 1`
  - `tensor_parallel_size: 1`
  - `pipeline_parallel_size: 1`
  - `expert_model_parallel_size: null`

PVC/PV setup:

- `peft-workspace-pvc` is bound to `peft-workspace-pv`.
- `peft-workspace-pv` is NFS:
  - server: `192.168.0.169`
  - path: `<ARTIFACT_ROOT>`
  - access mode: `ReadWriteMany`
- The GPU worker pods ran on the same host that is also serving the NFS export. This means the node was writing final artifacts back to its own local disk through NFS.

Job status:

- Both jobs reached:
  - `steps_completed: 580`
  - `epochs_completed: 2`
  - `percentage_done: 100.0`
- Both jobs were later marked `cancelled`.
- Entity Store returned `Model not found` for:
  - `default/lora-nim-nemotron-nano-30b-r32-shard-a`
  - `default/lora-nim-nemotron-nano-30b-r32-shard-b`

Pod/runtime state:

- Both worker pods were stuck in `Terminating`.
- Kubelet repeatedly emitted `FailedKillPod`.
- `kubectl exec` could not enter either container because the runtime namespace was already partially torn down.

Log evidence:

- Training reached the end:
  - `Trainer.fit stopped: max_steps=580 reached.`
- Shard A then entered final artifact copy:
  - `Copying /scratch/customizer_output_checkpoints/hf_adapter/adapter_model.safetensors to /pvc/cust-rbcxpusgj7pmwj4kw77mzm/trained/adapter_model.safetensors`
- This is after training, after local export, and during final write to the NFS-backed workspace PVC.

## Alpha/Rank Correction

The live Rank 32 pod payload showed:

```json
"lora": {
  "adapter_dim": 32,
  "alpha": 32
}
```

The repo's MoE path also emits `alpha = rank` from `scripts/stage3/train_adapter_moe.py`. So the active Rank 32 run was `rank=32, alpha=32`, not `alpha=64`.

That is correct for this Nemotron Nano MoE path. The dense Llama path uses `alpha = 2 * rank`; the MoE path should not.

## NCCL Assessment

This failure does not implicate NCCL in the active Rank 32 run:

- The job is single GPU.
- There is no expert parallelism.
- There is no tensor parallelism.
- There is no data parallel group.
- The logs show completion of training and a final copy to `/pvc`.

The older shipped DP2/EP2 template did have a separate NCCL/FSDP-MoE checkpoint-gather failure mode. That is why the single-GPU template exists. The current issue is a different failure class: NFS-backed final artifact publication.

## Likely Root Cause

The current design still has a shared NFS write path at the end of each job:

```text
/scratch/customizer_output_checkpoints/hf_adapter
  -> /pvc/<customization-id>/trained
  -> model uploader / entity-store finalization
```

The `workspace_dir` fix removed training checkpoints from NFS, but `OUTPUT_MODEL_PATH` remains on NFS. Two Rank 32 shard jobs finishing together create concurrent large writes to the same NFS export. Because the NFS server is the same node running the GPU jobs, the failure can wedge kubelet/containerd cleanup as well as the copy operation.

## Recommendations

1. Keep the single-GPU plus TIES approach.

   For this model and hardware, it avoids the DP2/EP2 FSDP-MoE/NCCL checkpoint-gather path and is still the better architecture.

2. Move final `OUTPUT_MODEL_PATH` off NFS.

   The robust fix is to make Customizer export and final-stage output land on node-local storage, for example:

   ```text
   /scratch/workspace/<job>/workspace
   /scratch/trained/<job>
   ```

   Then upload from local scratch to Data Store before pod exit. The upload step should be the only network publication step.

3. If Customizer cannot be changed quickly, serialize final publication.

   Run shard A and shard B sequentially for Rank 32, or add a Kubernetes/API-level lock around final copy/upload. Do not use an NFS file lock for this.

4. Avoid using shared RWX NFS as the high-volume training output path.

   Better options:

   - node-local `emptyDir` for all training and export intermediates
   - per-job local-path RWO PVC on the GPU node
   - object-store upload for final artifacts
   - a real distributed filesystem validated for concurrent large checkpoint writes, if shared filesystem semantics are required

5. Treat the current stuck pods as wedged runtime state.

   Current pod UIDs:

   - Shard A: `a83ca1eb-4097-4e9a-b374-cf60860a8777`
   - Shard B: `c7bfdaa4-9f86-4059-b448-d6a6e7536f09`

   Potential node-local recovery path on `<BLACKWELL_NODE>`:

   ```text
   /var/lib/kubelet/pods/<pod-uid>/volumes/kubernetes.io~empty-dir/scratch/
   ```

   Access likely requires root on the node. If artifacts are not recoverable there, expect to restart kubelet/containerd or reboot the node to clear the stuck runtime/NFS state.

6. Keep `alpha = rank` for Nemotron Nano MoE.

   Do not switch Rank 32 to `alpha=64` unless intentionally testing instability. The current code and live payload correctly use `alpha=32` for Rank 32.

## Next Concrete Actions

1. Recover or discard the current stuck Rank 32 artifacts from node-local scratch.
2. Clear the stuck pods/runtime state.
3. Patch Customizer or the NemoTrainingJob output path so final artifacts are written locally first.
4. Re-run one Rank 32 shard as a smoke test.
5. Re-run two shards in parallel only after final publication no longer writes directly to shared NFS.
6. Confirm success by checking Entity Store artifact status and cloning the adapter from Data Store, not just by checking Customizer job status.
