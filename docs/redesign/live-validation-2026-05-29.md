# Live Validation - 2026-05-29

## Scope

Validate the current `docs-to-data-to-lora` NeMo Platform/K8s handoff against
the live cluster before running destructive or training-submission steps.

Validation was intentionally read-only except applying the repository-owned
service-plane ConfigMap after its service URLs were corrected.

## Cluster Snapshot

Kubernetes context: `default`

Namespace: `nemo-peft`

Observed NeMo services:

| Service | NodePort | Health/OpenAPI result |
|---|---:|---|
| `nemo-core-api` | `30915` | `/health/live` returned healthy; OpenAPI exposed `/v1/jobs...` |
| `nemo-customizer` | `30910` | `/health/live` returned healthy; OpenAPI exposed health only |
| `nemo-entity-store` | `30911` | `/health/live` returned healthy; OpenAPI exposed `/v1/datasets` and `/v1/models` |
| `nemo-evaluator` | `30913` | OpenAPI exposed Evaluator v1 and v2 job APIs |
| `nemo-data-store` | `30912` | Service present for HF/Git compatibility |
| `nemo-data-designer` | `30812` | `/health` returned healthy |
| `mlflow` | `30920` | Service and pods running |

The live deployments are still the 25.12 individual microservice images, for
example:

```text
nemo-core-api: nvcr.io/nvidia/nemo-microservices/nmp-core:25.12
nemo-platform-customizer: nvcr.io/nvidia/nemo-microservices/customizer-api:25.12
nemo-platform-entity-store: nvcr.io/nvidia/nemo-microservices/entity-store:25.12
nemo-platform-evaluator: nvcr.io/nvidia/nemo-microservices/evaluator:25.12
```

This is not yet the newer 26.3.1 Platform chart/runtime path tracked in
`deploy/nemo-platform/versions.yaml`.

## SDK Availability

The local `uv run --no-sync` environment does not have `nemo_platform` installed.
Existing live service pods checked (`nemo-core-api`, `nemo-platform-customizer`,
`nemo-platform-entity-store`) also do not include the `nemo_platform` Python SDK.

Result: SDK-based validation of `scripts/stage3/platform_model_entities.py`,
`scripts/eval/upload_platform_filesets.py`, and Platform Customizer job creation
cannot run from the current local environment or existing service containers.
A validation image built with the NVIDIA SDK package source is still required.

## Entity Store Findings

Read-only `GET /v1/models` succeeded and returned 64 model entities.

The exact Platform base Model Entity refs expected by the new handoff were not
present:

```text
default/llama-3.2-1b-instruct
default/llama-3.2-3b-instruct
default/llama-3.1-8b-instruct
default/nemotron-3-nano-30b-a3b
```

Existing model entities are mostly trained LoRA adapter outputs with
`base_model` fields pointing at the desired bases. Those are useful lineage
records, but they are not the base Model Entity refs the Platform Customizer
payload now expects.

Read-only `GET /v1/datasets` succeeded and returned 10 datasets, but none of the
new Stage 3 dataset refs were registered yet:

```text
default/stage3-nim-curated
default/stage3-nim-curated-test
default/stage3-nim-curated-context-test
default/stage3-nemo-usvcs-curated
default/stage3-nemo-usvcs-curated-test
default/stage3-nemo-usvcs-curated-context-test
```

## Service Plane Finding

The `nemo-platform-service-plane` ConfigMap was not present in the live cluster.
Live validation showed this cluster uses service-specific 25.12 names and does
not expose the go-forward `http://nemo-platform-api:8080` Platform API service.

The live-compatible settings are preserved in
`deploy/nemo-platform/service-plane-configmap.legacy-25.12.yaml`:

```text
NMP_BASE_URL=http://nemo-core-api:8000
NMP_CUSTOMIZER_URL=http://nemo-customizer:8000
NMP_DATA_DESIGNER_URL=http://nemo-data-designer:8000
NMP_EVALUATOR_URL=http://nemo-evaluator:7331
NMP_ENTITY_STORE_URL=http://nemo-entity-store:8000
NMP_DATASTORE_URL=http://nemo-data-store:3000
NMP_DATASTORE_HF_ENDPOINT=http://nemo-data-store:3000/v1/hf
NMP_DATASTORE_GIT_BASE=http://nemo-data-store:3000
NMP_INFERENCE_GATEWAY_URL=http://rag-oai-proxy.runai-rag.svc.cluster.local:8080
```

The default `deploy/nemo-platform/service-plane-configmap.yaml` has since been
restored to the go-forward NeMo Platform API control-plane target. Do not apply
the default ConfigMap to this 25.12 cluster until the 26.3.1 Platform API
deployment exists.

## Validation Result

Passed:

- Kubernetes access to `nemo-peft`.
- Core, Customizer, Entity Store, Evaluator, and Data Designer services are reachable from inside the cluster.
- Entity Store model/dataset list APIs are readable.
- Evaluator v1 and v2 APIs are present.
- The live service names are documented and captured in a legacy 25.12 ConfigMap overlay.

Blocked for full Platform SDK validation:

- The live cluster is still running 25.12 individual microservice images.
- `nemo_platform` SDK is not available locally or inside checked service pods.
- The expected base Model Entities are absent.
- The expected Stage 3 dataset refs/FileSets are absent.
- The Platform FileSet/Model Entity SDK helpers need a validation image that
  installs the NVIDIA SDK package.

## Recommended Next Step

Do not submit a Platform Customizer training job yet. First choose one of these
paths:

1. Upgrade/redeploy to the 26.3.1 NeMo Platform chart, apply the default
   `deploy/nemo-platform/service-plane-configmap.yaml`, build validation images
   with `nemo-platform`, then run `deploy/platform-models/` and
   `deploy/platform-filesets/` live.
2. If staying temporarily on the current 25.12 microservice deployment, apply
   `deploy/nemo-platform/service-plane-configmap.legacy-25.12.yaml` and use the
   legacy Customizer payload path while continuing to log MLflow lineage through
   the repository wrapper.

The target showcase should prefer path 1. Path 2 is only a compatibility route
for the cluster as validated on 2026-05-29.
