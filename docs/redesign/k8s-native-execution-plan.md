# K8s-Native Execution Plan

This plan classifies repository workflows by where they should run in the
showcase architecture. The target is a Kubernetes-native pipeline where local
execution remains available for development, but the durable showcase path runs
as Kubernetes Jobs or NeMo/NIM services.

## Principles

- Batch transforms should run as Kubernetes Jobs.
- Long-running inference should run as NIM/NIM Proxy services.
- Training and evaluation should use NeMo Customizer and NeMo Evaluator.
- Dataset bytes should move through NeMo Data Store.
- Dataset/model identities should move through NeMo Entity Store.
- Secrets must come from Kubernetes Secrets, not source code.
- Local scripts should be dry-run/debug entry points, not the only production
  execution path.

## Execution Classes

| Class | Meaning |
|---|---|
| K8s Job | Deterministic batch work packaged in a container. |
| NeMo service | Work owned by NeMo Microservices. |
| NIM service | Model serving owned by NIM/NIM Proxy. |
| Local dev | Useful developer tool, not the showcase execution path. |
| Legacy | Keep for reference until replacement is stable. |

## Script Classification

### Source and Dataset Pipeline

| Component | Target Class | Rationale | Notes |
|---|---|---|---|
| `scripts/sitemap_to_inventory.py` | K8s Job or local dev | Deterministic inventory over a sitemap. | Keep local CLI; add Job when crawl orchestration is containerized. |
| `scripts/build_v2_dataset.py` | Split into K8s Jobs | Current monolith mixes multiple batch stages. | Break into stage Jobs after provenance sidecars stabilize. |
| `scripts/pipeline/stage0_corpus_prep.py` | K8s Job | Reads ES, emits passages and provenance. | Should run close to ES/Data Store. |
| `scripts/pipeline/stage1a_le_kvp.py` | K8s Job shard | Parallel LLM calls over passages. | Shard by passage file partitions. |
| `scripts/pipeline/stage1b_synthesis.py` | K8s Job shard | Parallel kNN + LLM calls. | Needs ES/network access and rate limits. |
| `scripts/pipeline/stage1c_instruction.py` | K8s Job shard | Parallel LLM calls. | Shard by selected passage partitions. |
| `scripts/pipeline/stage1_5_gapfill.py` | Split | Coverage analysis is K8s Job; generation should be NeMo Data Designer. | Direct LLM gap-fill becomes legacy fallback. |
| `scripts/pipeline/stage2_qa_eval.py` | K8s Job shard | Batch refinement/quality gate. | Later may become Evaluator-backed dataset-quality job. |
| `scripts/pipeline/stage3_curator.py` | NeMo service | Generic dedup/quality belongs in NeMo Curator. | Keep Python version as local fallback only. |
| `scripts/pipeline/stage4_validation.py` | K8s Job or NeMo Evaluator | Current external judge gate can run as a Job. | Longer term, express through Evaluator where practical. |

### Data Store and Entity Store Operations

| Component | Target Class | Rationale | Notes |
|---|---|---|---|
| `scripts/eval/upload_test_datasets.py` | K8s Job | Deterministic dataset upload/registration. | Needs refactor to remove hard-coded NodePorts and credentials. |
| `scripts/stage3/build_moe_shards.py` | K8s Job | Batch dataset transformation plus registration. | Keep split logic; use Secret/ConfigMap for Data Store. |
| Dataset/version manifest writing | K8s Job | Should run where dataset artifacts are built. | Output should be registered with dataset bytes. |

### Training and Adapter Workflows

| Component | Target Class | Rationale | Notes |
|---|---|---|---|
| `scripts/stage3/train_adapter.py` | NeMo service driver | Actual training is NeMo Customizer. | CLI should submit configured Customizer jobs. |
| `scripts/stage3/train_adapter_moe.py` | NeMo service driver | Actual training is NeMo Customizer. | Keep MoE config builder; reduce local cluster assumptions. |
| `scripts/stage3/ties_merge.py` | K8s Job | Deterministic batch merge over adapter artifacts. | First concrete K8s template in `deploy/ties-merge/`. |
| `scripts/stage3/inspect_adapter_checkpoint.py` | K8s Job or local dev | Fast structural validation. | Run after TIES merge before NIM load test. |
| `scripts/stage3/*orchestrator*.py` | Legacy to K8s workflow | Current local process managers. | Replace with Jobs, Customizer job submission, and status collection. |

### Evaluation and Serving

| Component | Target Class | Rationale | Notes |
|---|---|---|---|
| NIM base/adapter serving | NIM service | Long-running model serving. | Prefer NIM Operator/NIM Proxy patterns. |
| `scripts/eval/register_evaluator_entities.py` | K8s Job | Idempotent control-plane registration. | Now aligned to NIM Proxy model targets. |
| `scripts/eval/run_evaluation_matrix.py` | K8s Job | Orchestrates Evaluator jobs and records IDs. | NeMo Evaluator owns execution. |
| `deploy/rag-oai-proxy/*` | Legacy | Claude-generated workaround. | Keep until native NIM Proxy/Evaluator path is verified. |
| `scripts/eval/adapter_sync.py` | Legacy or deployment-specific Job | Sidecar sync depends on path-based LoRA serving. | Prefer native NIM LoRA serving. |

## First K8s Template: TIES Merge

TIES merge is the lowest-risk first K8s conversion because:

- It is deterministic.
- It has bounded inputs and outputs.
- It does not require interactive control.
- It can run without GPU in most cases.
- It already has a structural inspector.

Artifacts:

```text
deploy/ties-merge/
  Containerfile
  README.md
  job.yaml
```

The Job supports two input patterns:

1. Clone source adapters from NeMo Data Store using Data Store credentials from
   a Kubernetes Secret.
2. Merge already-mounted adapter directories from a PVC.

Use Data Store cloning for a portable showcase. Use mounted directories when
debugging a local persistent volume or when adapters are already materialized by
Customizer on shared storage.

## Second K8s Template: Adapter Inspection

Adapter inspection is the follow-on validation Job after TIES merge. It is a
structural gate, not a serving gate: it verifies required PEFT files, parses
`adapter_config.json`, reads the safetensors header, and writes a JSON report.
The final viability check is still a NIM/NIM Proxy live-load test.

Artifacts:

```text
deploy/adapter-inspection/
  Containerfile
  README.md
  job.yaml
```

The image is intentionally torch-free and does not need GPU resources. Warnings
exit successfully by default so the pipeline can preserve the inspection report
and continue to live-load validation; use `--fail-on-warn` for strict pipelines.

## Third K8s Template: Evaluator Registration

Evaluator registration is the first control-plane Job. It registers Stage 3
Evaluator targets and configs idempotently, using native NIM Proxy model targets
with `format: nim`. The Job consumes the training-session adapter inventory from
a ConfigMap and keeps service URLs in environment variables instead of source
code.

Artifacts:

```text
deploy/evaluator-registration/
  Containerfile
  README.md
  job.yaml
```

This Job does not register datasets. Dataset bytes and identities belong to NeMo
Data Store and NeMo Entity Store, which should be handled by a separate dataset
registration Job.

## K8s Resource Guidance

| Workload | CPU | Memory | GPU | Storage |
|---|---:|---:|---:|---|
| TIES merge Nano r=16 | 4-8 | 24-32Gi | none initially | PVC or ephemeral + output PVC |
| Adapter inspection | 100m-500m | 512Mi | none | read-only adapter mount |
| Dataset registration | 500m-1 | 1-2Gi | none | workspace + Data Store access |
| Stage 0 corpus prep | 1-4 | 4-16Gi | none | output PVC/Data Store |
| Stage 1 generation shard | 1-4 | 2-8Gi | none client-side | output PVC/Data Store |
| Curator | service-dependent | service-dependent | likely | NeMo Curator |

## Environment and Secret Pattern

Use ConfigMaps for non-secret runtime configuration:

```text
DATA_STORE_GIT_BASE=http://nemo-data-store:3000
TIES_TRIM_RATIO=0.2
OUTPUT_DIR=/outputs/lora-nemo-usvcs-nemotron-nano-30b-r16
```

Use Secrets for credentials:

```text
DATA_STORE_USER
DATA_STORE_PASSWORD
```

Do not put NodePorts, passwords, or host-specific paths in scripts.

## Recommended Conversion Order

1. TIES merge Job. Implemented in `deploy/ties-merge/`.
2. Adapter inspection Job. Implemented in `deploy/adapter-inspection/`.
3. Evaluator target/config registration Job. Implemented in `deploy/evaluator-registration/`.
4. Evaluation matrix orchestration Job.
5. Dataset registration Job.
6. Stage 0 corpus/provenance Job.
7. Stage 1 generation shard Jobs.
8. Gap analysis Job and Data Designer submission.
9. Curator service integration.

This order gives immediate operational value while avoiding a large rewrite of
the source-grounded dataset pipeline.
