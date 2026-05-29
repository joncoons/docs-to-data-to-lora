# TIES Merge Kubernetes Job

This directory contains the first K8s-native template for the Stage 3 adapter
workflow. It runs `scripts/stage3/ties_merge.py` as a Kubernetes Job.

## Build Image

```bash
docker build \
  -f deploy/ties-merge/Containerfile \
  -t <registry>/docs-to-data-to-lora/ties-merge:latest .

docker push <registry>/docs-to-data-to-lora/ties-merge:latest
```

Update `deploy/ties-merge/job.yaml` with that image.

## Secret

When cloning adapters from NeMo Data Store, create a Secret:

```bash
kubectl create secret generic nemo-data-store-git \
  -n nemo-peft \
  --from-literal=DATA_STORE_USER='<username>' \
  --from-literal=DATA_STORE_PASSWORD='<password>'
```

The Job sources `deploy/nemo-platform/service-plane-configmap.yaml` and maps
`DATA_STORE_GIT_BASE` from `NMP_DATASTORE_GIT_BASE`. This is still a direct
Git/Data Store compatibility endpoint because TIES clone mode reads adapter
repositories directly. Do not embed credentials in the image, args, or source
code.

## Data Store Clone Mode

Use the `--adapters` argument when source adapters should be cloned from Data
Store:

```yaml
args:
  - --adapters
  - default/lora-nemo-usvcs-nemotron-nano-30b-r16-shard-a@cust-...
  - default/lora-nemo-usvcs-nemotron-nano-30b-r16-shard-b@cust-...
  - --out
  - /outputs/lora-nemo-usvcs-nemotron-nano-30b-r16
  - --trim-ratio
  - "0.2"
```

## Mounted Directory Mode

Use the `--adapter-dirs` argument when the source adapter directories are
already mounted:

```yaml
args:
  - --adapter-dirs
  - /inputs/shard-a
  - /inputs/shard-b
  - --out
  - /outputs/lora-nemo-usvcs-nemotron-nano-30b-r16
  - --trim-ratio
  - "0.2"
```

This mode avoids Git/Data Store credentials entirely.

## Validate Output

After the Job completes, run the structural inspector:

```bash
python scripts/stage3/inspect_adapter_checkpoint.py \
  /mnt/nvme2/peft/checkpoints/lora/lora-nemo-usvcs-nemotron-nano-30b-r16
```

Structural inspection is not the final validation. The adapter is only viable
after a NIM live-load test succeeds.
