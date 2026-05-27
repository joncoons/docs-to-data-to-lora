# Stage 3 MoE Task 4.6 — Resume Instructions After Node Reboot

**Status checkpoint**: 2026-05-27 ~01:40 CDT, before reboot of `ubuntu-local-dev`.

The cluster's NFS server (co-located on `ubuntu-local-dev`) reached the documented
unrecoverable hang state during Round 2 (r=32) of MoE Task 4.6. Per
`[[feedback_nfs_hang_recovery]]`, the only reliable fix is a node reboot. After the
reboot you'll resume training following this file.

---

## What survives the reboot (committed and on persistent storage)

- **Round 1 merged adapter**: `/mnt/nvme2/peft/checkpoints/lora/lora-nim-nemotron-nano-30b-r16/`
  (885 MB BF16 safetensors + adapter_config.json). Usable as-is for the eval matrix.
- **All committed code**: branch `stage3-peft-training` is up to date. See `git log` for
  the last several commits including the two methodology docs
  (`NEMOTRON_NANO_NFS_DEADLOCK.md`, `INTERLEAVED_GPU_SCHEDULING.md`) and the MoE training
  modules (`moe_models.py`, `build_moe_shards.py`, `train_adapter_moe.py`, `ties_merge.py`).
- **All 4 shard datasets in entity-store**: `default/stage3-nim-curated-shard-{a,b}` and
  `default/stage3-nemo-usvcs-curated-shard-{a,b}`.
- **The Customizer template artifact**: committed at
  `scripts/stage3/customizer-templates/nemotron-nano-30b-a3b-singleGPU.yaml`.

## What's lost / abandoned

- **Round 2 r=32 nim shards (both a and b)**. Salvage of shard-a's NFS files
  (`/mnt/nvme2/peft/cust-rbcxpusgj7pmwj4kw77mzm/trained/`) was not completed before the
  hang took the upload offline. Per user direction we are abandoning recovery — both
  shards will be re-run from scratch.

## Remaining work — 6 SFT jobs + 3 TIES merges

| Job | Corpus | Rank | Shard | Notes |
|---|---|---|---|---|
| 1 | nim | r=32 | a | re-run (lost) |
| 2 | nim | r=32 | b | re-run (lost) |
| 3 | nemo_usvcs | r=16 | a | new |
| 4 | nemo_usvcs | r=16 | b | new |
| 5 | nemo_usvcs | r=32 | a | new |
| 6 | nemo_usvcs | r=32 | b | new |

Plus 3 TIES merges (one per `(corpus, rank)` pair).

## Scheduling — rank-lane assignment

Per the lane pattern from
[`INTERLEAVED_GPU_SCHEDULING.md`](INTERLEAVED_GPU_SCHEDULING.md), to eliminate concurrent
NFS writes we assign **one GPU as the r=16 lane and the other as the r=32 lane**. The
r=16 lane is faster (~2.6h/shard), the r=32 lane is slower (~3h/shard); their write
windows naturally don't overlap.

```
r=16 lane (Blackwell A):
  ┌── nemo r=16 a (~2.6h) ──┐   ┌── nemo r=16 b (~2.6h) ──┐  (lane done at ~5.5h)
                            └── write 885 MB ── ✓ NFS write happens alone
                              (because r=32 lane is mid-training)

r=32 lane (Blackwell B):
  ┌── nim r=32 a (~3h) ──┐   ┌── nim r=32 b (~3h) ──┐   ┌── nemo r=32 a (~3h) ──┐   ┌── nemo r=32 b (~3h) ──┐
                          └─ write 1.64 GB ── ✓ alone
                            (lane finishes at ~12h)
```

Total wall-clock: ~12h, driven by the r=32 lane.

## Step-by-step pickup procedure

### 0. Reboot

```bash
echo "Alvin7222" | sudo -S reboot
```

Then wait ~5 minutes. From a fresh shell, verify the node is back:

```bash
ping -c 2 ubuntu-local-dev   # or 192.168.1.187
```

### 1. Verify cluster came up healthy

```bash
kubectl get nodes
# expect: ubuntu-local-dev Ready, ubuntu2 Ready

kubectl get pods -n nemo-peft 2>&1 | grep -iE "Running|Pending|Error" | head -20
# expect: nemo-platform-customizer-..., nemo-platform-entity-store-..., etc. all Running

kubectl get pods -n runai-rag 2>&1 | grep -iE "Running" | head -10
```

### 2. Verify NFS is healthy (the actual recovery test)

```bash
# Should return quickly (no hang); confirms NFS path is responsive:
timeout 10 ls /mnt/nvme2/peft/checkpoints/lora/ | head -5
# expect: should list directories including lora-nim-nemotron-nano-30b-r16/

timeout 10 ls /mnt/nvme2/peft/datasets/v2/ | head -5
# expect: nim_curated, nemo_usvcs_curated
```

If either of these hangs for >10 sec, **do not proceed** — the reboot didn't fully clean
state and another investigation is needed.

### 3. Verify Customizer + Data Store + Entity Store reachable

```bash
curl -sf -m 10 http://10.43.167.101:8000/v1/customization/jobs?page_size=1 | head -c 200
# expect: JSON response with "data": [...]

curl -sf -m 10 http://10.43.197.14:3000/v1/hf/api/models | head -c 200
# expect: JSON response

curl -sf -m 10 http://10.43.187.212:8000/v1/datasets?page_size=1 | head -c 200
# expect: JSON response
```

If any of these returns empty or hangs, give the cluster another 1-2 minutes to settle,
then retry.

### 4. Verify the 4 shard datasets are still registered

```bash
for name in stage3-nim-curated-shard-a stage3-nim-curated-shard-b \
            stage3-nemo-usvcs-curated-shard-a stage3-nemo-usvcs-curated-shard-b; do
  echo "--- $name ---"
  curl -sf -m 10 "http://10.43.187.212:8000/v1/datasets/default/${name}" \
    | /home/joncoons/anaconda3/envs/nat/bin/python3 -c "
import json,sys; d=json.load(sys.stdin)
print(f'  files_url: {d.get(\"files_url\")}  format: {d.get(\"format\")}')"
done
# expect: all 4 print files_url + format=hf
```

### 5. Launch the rank-lane orchestrator

```bash
cd /home/joncoons/claude/docs-to-data-to-lora
/home/joncoons/anaconda3/envs/nat/bin/python3 \
    scripts/stage3/moe_lane_orchestrator.py 2>&1 | tee -a /tmp/moe-lane-status.log
```

Or to launch in the background (recommended — total wall-clock ~12h):

```bash
nohup /home/joncoons/anaconda3/envs/nat/bin/python3 \
    scripts/stage3/moe_lane_orchestrator.py \
    > /tmp/moe-lane-status.log 2>&1 &
echo "PID: $!"
```

Tail the log periodically to watch progress:

```bash
tail -f /tmp/moe-lane-status.log
```

### 6. Sanity-check progress mid-run

Around t=3h, the first r=32 SFT (nim r=32 a) should produce its first val_loss reading.
If you see val_loss reasonable (≤2.0) and train_loss descending smoothly past step 100,
the pipeline is healthy and you can let it continue unattended.

If a job goes `cancelled` mid-training (NOT the "race at end" scenario covered by
`[[feedback_customizer_status_vs_artifact]]`, but a real cancellation), check the worker
pod logs:

```bash
kubectl logs -n nemo-peft <cust-id>-training-job-worker-0 --tail=50
```

### 7. After all 6 jobs done — verify the 3 merged adapters on disk

```bash
ls -la /mnt/nvme2/peft/checkpoints/lora/lora-{nim,nemo-usvcs}-nemotron-nano-30b-r{16,32}/
# expect: 4 directories, each with adapter_config.json + adapter_model.safetensors
```

Note: Round 1 (lora-nim-nemotron-nano-30b-r16) already exists from before the hang.
Rounds 2-4 produce the other 3.

### 8. Append progress to evals/training_session.log

After all 4 MoE adapters are in place, append a "MoE Nano corpus" section to
`evals/training_session.log` with per-shard train/val losses and the final TIES merge
locations.

---

## If NFS hangs again during the rerun

If a worker pod's post-training copy hangs (Round 2 symptom repeats), the orchestrator
will detect it via job-status polling. Recovery steps:

1. Cancel both Round 2 shards via Customizer API.
2. `kubectl delete pod --force --grace-period=0 <worker-pod>` on the wedged pod(s).
3. Check whether the lane-based stagger needs further widening (e.g., add a deliberate
   delay before each r=32 shard submission).
4. Worst case: reboot again. Document any new findings in
   [`NEMOTRON_NANO_NFS_DEADLOCK.md`](NEMOTRON_NANO_NFS_DEADLOCK.md).

The durable fix remains the Customizer source patch described in
`NEMOTRON_NANO_NFS_DEADLOCK.md §Future work` (option 1) or per-job RWO PVCs (option 2).
These are deferred work, not blocking Stage 3 completion.
