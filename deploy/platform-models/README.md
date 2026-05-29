# Platform Model Entity Verification Job

This Job verifies that the base Model Entities required by Stage 3 Customizer
jobs exist in NeMo Platform and writes an audit manifest for MLflow lineage:

```text
/outputs/observability/platform-models/platform_model_entities_manifest.json
```

NeMo Platform Customizer jobs reference Model Entities and FileSet URIs. NVIDIA's
Platform docs describe the sequence as: create a model FileSet, create a Model
Entity that points to that FileSet, wait for the model spec to populate, then use
`model: "workspace/name"` in Customizer jobs.

## Build Image

```bash
docker build \
  -f deploy/platform-models/Containerfile \
  --build-arg NEMO_PLATFORM_REQUIREMENTS='nemo-platform' \
  -t <registry>/docs-to-data-to-lora/platform-models:latest .

docker push <registry>/docs-to-data-to-lora/platform-models:latest
```

Update `deploy/platform-models/job.yaml` with that image.

## Verify Existing Resources

```bash
kubectl apply -f deploy/nemo-platform/service-plane-configmap.yaml
kubectl apply -f deploy/platform-models/job.yaml
```

The default manifest verifies expected resources only. It does not download
large model checkpoints or create gated Hugging Face resources.

## First Cluster Bootstrap

For an initial cluster, run the same helper with `--create-missing` after the
required Hugging Face licenses have been accepted and the Platform secret exists:

```bash
python scripts/stage3/platform_model_entities.py \
  --verify \
  --create-missing \
  --hf-token-secret stage3-hf-token \
  --out /tmp/platform_model_entities_manifest.json
```

`--hf-token-secret` is the NeMo Platform secret name, not the raw token value.
