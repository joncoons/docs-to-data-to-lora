# Platform FileSet Upload Kubernetes Job

This Job uploads finalized Stage 3 train/test datasets and lineage sidecars to
NeMo Platform FileSets. It is the Platform-native handoff to Customizer: a
training dataset named `stage3-nim-curated` becomes
`fileset://default/stage3-nim-curated`.

## Build Image

```bash
docker build   -f deploy/platform-filesets/Containerfile   --build-arg NEMO_PLATFORM_REQUIREMENTS='nemo-platform'   -t <registry>/docs-to-data-to-lora/platform-filesets:latest .

docker push <registry>/docs-to-data-to-lora/platform-filesets:latest
```

Update `deploy/platform-filesets/job.yaml` with that image.

## Service Configuration

The Job sources `deploy/nemo-platform/service-plane-configmap.yaml` and uses:

```text
NMP_BASE_URL
NMP_WORKSPACE
```

The script calls `sdk.files.upload(..., fileset_auto_create=True)` for each
file, preserving split files and lineage sidecars under their repository-style
paths inside the FileSet.

## Dry Run

Use dry-run to validate the FileSet plan and write an audit manifest without
calling the Platform SDK:

```bash
python scripts/eval/upload_platform_filesets.py   --base-dir /mnt/nvme2/peft/datasets/v2   --collections nim_curated nemo_usvcs_curated   --include-train   --include-test   --include-context-test   --observability-dir /tmp/platform-filesets   --dry-run
```

The output manifest is:

```text
/outputs/observability/platform-filesets/platform_filesets_manifest.json
```
