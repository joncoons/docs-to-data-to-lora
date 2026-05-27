# Interleaved GPU Scheduling for Multi-Rank / Multi-Model LoRA Training

**Date captured**: 2026-05-26 (originated during MoE Task 4.6 NFS deadlock recovery).
**Applies to**: any multi-shard / multi-rank training where concurrent post-training NFS
writes can saturate the storage path.

## The principle

When two LoRA training jobs run on the same node, they all eventually have to write their
trained adapter checkpoint to NFS-backed storage (per the current Customizer pipeline —
see [`NEMOTRON_NANO_NFS_DEADLOCK.md`](NEMOTRON_NANO_NFS_DEADLOCK.md) for the underlying
issue). If both jobs **start at the same time AND have the same training duration**, they
both finish training at nearly the same instant, and both attempt their NFS final-copy
simultaneously. That concurrent write is what tips the NFS mount into lock-contention
deadlock at scale (~3.3 GB combined write in the observed failure case).

**The fix** is to make sure no two jobs are doing their post-training NFS write at the
same time. The simplest way to achieve this without code changes: ensure jobs finish at
different times by pairing workloads with **different training durations** on the two
GPUs. Then as one GPU frees up, you submit the next job — natural staggering, no explicit
locks.

## Pattern 1 — Multi-rank within the same model

When training the same base model at multiple ranks (e.g., r=16 + r=32 on Llama or
Nano), the per-step time at r=32 is slightly slower than r=16 (more trainable params per
step), so r=16 finishes first.

```
GPU 0:   ┌──── r=16 (~2.6h)  ────┐
                                  ├── r=16 done; start r=32 on freed GPU 0 ─┐
                                                                              ├── r=32 done
GPU 1:   ┌─────── r=32 (~3h) ─────────┐ ── done; start something else
                                       └ r=32 NFS write happens alone
```

The r=16 finishes its NFS write while r=32 is still mid-training; the r=32 finishes its
NFS write while the **next** r=16 (on GPU 0) is mid-training. Writes are temporally
isolated.

## Pattern 2 — Multi-model interleaving

When training adapters for multiple distinct base models or datasets simultaneously (e.g.,
NIM corpus and NeMo USvcs corpus), workloads naturally have different per-step times
because the model architecture and the dataset size differ. Pair them across GPUs:

```
GPU 0:   nim    × r=N ────────────┐
                                   ├── done; submit nemo × r=N on GPU 0
GPU 1:   nemo   × r=M ────────┐
                               └── done; submit nim × r=M on GPU 1
```

The smaller corpus (nemo, ~14% fewer training rows in our case) finishes ~14% faster than
the larger (nim). Their finish times are offset, NFS writes don't overlap.

## Pattern 3 — Mix the two (used by Stage 3 MoE Task 4.6 recovery)

Combining patterns 1 and 2 gives 5 stages for the remaining work after Round 1 r=16
parallel succeeded but Round 2 r=32 parallel hung:

| Stage | t (approx) | Trigger | GPU A | GPU B |
|---|---|---|---|---|
| 1 | t=0 | initial submit | nim r=32 shard-b (~3h) | nemo r=16 shard-a (~2.6h) |
| 2 | ~2.6h | GPU B frees (r=16 < r=32) | (running) | nemo r=16 shard-b |
| 3 | ~3h | GPU A frees | nemo r=32 shard-a | (running) |
| 4 | ~5.6h | GPU B frees | (running) | nemo r=32 shard-b |
| 5 | ~6h | GPU A frees, queue empty | idle | (running) |
| End | ~8.5h | GPU B frees | — | done |

At every NFS-write instant, the other GPU is mid-training. Concurrent write count = 1.

Wall-clock is the same as fully-parallel scheduling (~8.5h vs the ~6h floor set by the
slowest pair), but eliminates the concurrent-write contention without code or
infrastructure changes.

## Application to fresh runs

When starting a multi-rank LoRA training matrix from scratch:

- **Always pair r=16 + r=32 on the two GPUs** (or any rank-pair with notable per-step
  time differential) for each training session, instead of running r=16 in parallel
  shards followed by r=32 in parallel shards.
- **Across multi-model runs**, alternate which corpus is on which GPU as the matrix
  walks — this maximizes the chance that two NFS writes hit the storage path at
  different times.
- **First shard of any rank pair stays on the GPU until the rank is complete**, then the
  second shard rotates in.

This is purely an operational scheduling pattern; no Customizer source change required.
The structural fix (move final OUTPUT_MODEL_PATH off NFS — see
[`NEMOTRON_NANO_NFS_DEADLOCK.md`](NEMOTRON_NANO_NFS_DEADLOCK.md) §"Future work") is still
the right long-term direction, but this pattern unblocks immediate work.

## Caveats

- The pattern relies on different per-step times. If two workloads happen to have nearly
  identical durations (e.g., same corpus, same rank, same base), the offset disappears.
- Network/disk variance can sometimes line up finish times even when training durations
  differ. The pattern reduces concurrent-write probability; it doesn't eliminate it. If
  you observe wedged pods at the post-training copy phase, fall back to fully sequential
  shard submission for the affected pair.
- The `local-path/RWO PVC` and Customizer-patch options remain the durable fix; this
  scheduling pattern is the no-code workaround that buys time.

## Related artifacts

- `/home/joncoons/claude/nim-sft-final/NEMOTRON_NANO_LORA_METHODOLOGY.md` — canonical Nemotron-Nano single-GPU + 2-way TIES methodology
- [`NEMOTRON_NANO_NFS_DEADLOCK.md`](NEMOTRON_NANO_NFS_DEADLOCK.md) — Customizer NFS output-copy issue this scheduling pattern works around
- Memory: `[[project_customizer_output_copy_nfs_deadlock]]` — issue and future-work backlog
- Memory: `[[feedback_nfs_hang_recovery]]` — recovery from the wedged state
