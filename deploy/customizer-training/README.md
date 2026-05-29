# Platform Customizer Training Jobs

These Jobs submit Stage 3 LoRA training to NeMo Platform Customizer from inside
Kubernetes. They are submission and polling jobs; the actual GPU training runs
inside NeMo Platform's Customizer workflow.

## Prerequisites

Run these Platform handoff jobs first:

```bash
kubectl apply -f deploy/platform-models/job.yaml
kubectl apply -f deploy/platform-filesets/job.yaml
```

`platform-models` verifies base Model Entities. `platform-filesets` uploads the
training datasets to FileSets such as `fileset://default/stage3-nim-curated`.

## Build Image

```bash
docker build \
  -f deploy/customizer-training/Containerfile \
  --build-arg NEMO_PLATFORM_REQUIREMENTS='nemo-platform' \
  -t <registry>/docs-to-data-to-lora/customizer-training:latest .

docker push <registry>/docs-to-data-to-lora/customizer-training:latest
```

Update the image in `dense-job.yaml` and `moe-job.yaml`.

## Dense LoRA Template

`dense-job.yaml` submits:

```text
collection: nim_curated
base_model: meta/llama-3.2-3b-instruct
rank: 16
dataset: fileset://default/stage3-nim-curated
model: default/llama-3.2-3b-instruct
```

Duplicate the Job and adjust `--collection`, `--base-model`, `--rank`, and
`MLFLOW_RUN_NAME` for the full dense matrix.

## MoE LoRA Template

`moe-job.yaml` submits one Nemotron Nano shard. Duplicate it for shard `b` and
for `nemo_usvcs_curated` once the shard FileSets exist.

The Jobs include `--wait`, which now polls NeMo Platform Customizer by job
name/workspace through the SDK. `CUSTOMIZER_TIMEOUT_S` controls the polling
window without changing the container image.
