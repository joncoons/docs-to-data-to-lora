# Curator dataset Customizer training — 2026-06-29

## Purpose

Train four independent LoRA PEFT adapters from the Curator DiverseQA datasets on `nvidia/nemotron-3-nano-30b-a3b`, using one physical Blackwell GPU per active job:

1. NIM r=16 and r=32 concurrently.
2. NeMo Microservices r=16 and r=32 queued behind them.

## Dataset entities

- `default/stage3-nim-curated-curator-diverseqa-20260629`
- `default/stage3-nemo-usvcs-curated-curator-diverseqa-20260629`

Both Entity Store records use `format: hf`, `hf://datasets/default/...` file URLs, and the internal Data Store HF endpoint. Data Store persists the backing repositories in Git/LFS format. Each HF tree contains `training.jsonl`, `validation.jsonl`, and `manifests/dataset_version_manifest.json`.

## Training configuration

- Customizer: 25.12
- Base: `nvidia/nemotron-3-nano-30b-a3b`
- Template: `nvidia/nemotron-3-nano-30b-a3b@v1.0+96GB-singleGPU`
- Epochs: 2
- Batch size: 8
- Micro batch size: 1
- Learning rate: 1e-4
- Warmup steps: 20
- Optimizer: AdamW with cosine annealing
- Seed: 42
- Sequence packing: disabled for Blackwell SM120
- LoRA alpha: equal to rank

## Jobs

| Key | Rank | Customizer job | Output model | Initial GPU state |
| --- | ---: | --- | --- | --- |
| NIM | 16 | `cust-MEq4hwgw5EXYYGrGx5xVhb` | `default/lora-nim-curator-nemotron-nano-30b-r16-20260629` | Running, one Blackwell GPU |
| NIM | 32 | `cust-5vVCigA1cTCqtrmV6zJY9C` | `default/lora-nim-curator-nemotron-nano-30b-r32-20260629` | Running, one Blackwell GPU |
| NeMo Microservices | 16 | `cust-V9Kuqsr4BsrsAYWmus3HAK` | `default/lora-nemo-usvcs-curator-nemotron-nano-30b-r16-20260629` | Queued for one Blackwell GPU |
| NeMo Microservices | 32 | `cust-74AFU3baYLiZHUoWQyCiH2` | `default/lora-nemo-usvcs-curator-nemotron-nano-30b-r32-20260629` | Queued for one Blackwell GPU |

The queued NeMo worker pods report `Insufficient nvidia.com/gpu` by design. The two physical GPUs are occupied by the two NIM workers. When either worker terminates, Kubernetes can schedule one queued NeMo worker automatically.

## Timing

Timing artifacts are under:

`data/customizer/runs/curator-diverseqa-20260629/`

- `training_manifest.json`: job IDs, submission timestamps, initial status, datasets, ranks, and output entities.
- `status_events.jsonl`: Customizer status transitions.
- `gpu_timing.json`: current status and authoritative GPU-worker timestamps.
- `gpu_timing_events.jsonl`: worker phase and Customizer status transitions.
- `timing_watcher.pid`: present while the detached watcher is active.
- `timing_watcher_errors.jsonl`: created only if a polling call fails.

Reported measures:

- submission-to-GPU-start delay;
- worker creation-to-container-start queue time;
- actual GPU container start-to-finish training time;
- Customizer end-to-end elapsed time;
- steps, epochs, and completion percentage.

The watcher exits automatically after all four Customizer jobs are terminal.

## Temporary cluster topology

- Scaled to zero in `runai-rag`: `nemotron-parse-v12`, `nim-vlm`.
- Blackwell node: `ubuntu-local-dev`.
- Physical Blackwell GPUs: 2.
- Time-slicing temporarily removed from the `ubuntu-local-dev` entry in `gpu-operator/time-slicing-config`.
- Node schedulable GPU capacity during training: 2.
- Original time-slicing value: 4 replicas per physical GPU, for capacity 8.
- Customizer API and Entity Store were moved to `ubuntu2` to avoid the Blackwell node's 110-pod ceiling.
- Optional NeMo core API/controller/log collector and jobs database remain scaled to zero; the direct Customizer + NeMoTrainingJob path does not use them.

Do not restore time-slicing or the NIM deployments while training workers remain active. After all four jobs are terminal, restore the original `ubuntu-local-dev` time-slicing entry (`replicas: 4`), restart the Blackwell device-plugin/GFD pods, and verify capacity returns to 8 before deciding whether to restore the model deployments.
