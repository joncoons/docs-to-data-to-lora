# Adapter Inspection Kubernetes Job

This Job runs `scripts/stage3/inspect_adapter_checkpoint.py` against a PEFT
LoRA adapter directory. It is intended to run after TIES merge and before a NIM
live-load test.

The inspector is intentionally torch-free. It checks required files, parses
`adapter_config.json`, reads the safetensors header, and writes a JSON report.
It does not prove that NIM can load the adapter.

## Build Image

```bash
docker build \
  -f deploy/adapter-inspection/Containerfile \
  -t <registry>/docs-to-data-to-lora/adapter-inspection:latest .

docker push <registry>/docs-to-data-to-lora/adapter-inspection:latest
```

Update `deploy/adapter-inspection/job.yaml` with that image.

## Run

The template expects:

- `peft-checkpoints` PVC mounted read-only at `/adapters`
- `peft-reports` PVC mounted at `/reports`

Adjust the adapter path and report path in `job.yaml`:

```yaml
args:
  - /adapters/lora-nemo-usvcs-nemotron-nano-30b-r16
  - --json
  - --output-json
  - /reports/lora-nemo-usvcs-nemotron-nano-30b-r16-inspection.json
```

Warnings exit with status `0` by default so the pipeline can continue to the
NIM live-load gate while preserving the report. Add `--fail-on-warn` when the
pipeline should stop on structural warnings.

## Expected Pipeline Position

```text
NeMo Customizer outputs adapters
  -> TIES merge Job
  -> Adapter inspection Job
  -> NIM/NIM Proxy live-load validation
  -> NeMo Evaluator matrix
```
