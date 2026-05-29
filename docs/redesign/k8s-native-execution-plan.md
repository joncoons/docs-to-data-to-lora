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
- Jobs should emit structured observability files that can be exported to MLflow.
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
| `scripts/pipeline/stage1_5_gapfill.py` | Split | Coverage analysis is K8s Job; generation should be NeMo Data Designer. | Default path now emits `provenance/gap_manifest.json` plus `data_designer/gapfill_requests.jsonl`; direct LLM generation is `legacy-direct`. |
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

## Fourth K8s Template: Evaluation Matrix

Evaluation matrix orchestration is the first execution Job that submits NeMo
Evaluator jobs. It assumes registration has already created targets and configs,
then submits Wave A single-axis jobs, Wave B LoRA-vs-LoRA jobs, and Wave C 49B
comparator jobs. It writes submitted Evaluator job IDs to a PVC-backed JSON file
for result collection and auditability.

Artifacts:

```text
deploy/evaluation-matrix/
  Containerfile
  README.md
  job.yaml
```

The Job can either poll each wave to terminal state or run with `--submit-only`
when a separate controller/result-collection Job should own polling.

## MLflow Observability

MLflow should be the audit plane, not the owner of NeMo resources. Use native
Customizer and Evaluator MLflow export when available, and emit repository-owned
observability JSON files for dataset lineage, Data Designer gapfill, TIES merge,
adapter inspection, and K8s control-plane Jobs. See
`docs/redesign/mlflow-observability-contract.md`.

Dataset registration links the provenance sidecars, dataset version manifest,
NeMo Data Store URI, Entity Store reference, and eventual Customizer/Evaluator
jobs. Stage 0 corpus prep now emits the first upstream observability records for
crawl/passages/source provenance.

## Fifth K8s Template: Dataset Registration

Dataset registration is the first implemented data-plane control Job. It creates
or updates NeMo Data Store dataset repositories, registers Entity Store dataset
refs, and emits MLflow-ready observability files for dataset lineage and
versioning.

Artifacts:

```text
deploy/dataset-registration/
  Containerfile
  README.md
  job.yaml
```

The Job registers the Stage 3 training dataset, bare held-out test dataset, and
context-baked test dataset for each collection. It requires
`manifests/dataset_version_manifest.json` by default, uploads recognized
lineage sidecars into each Data Store dataset repository, and emits
`run_context.json`, `metrics.json`, `artifacts_manifest.json`, and
`service_refs.json` for later MLflow export.

## Sixth K8s Template: Dataset Finalization

Dataset finalization is the versioning step that should run after split/sample
artifacts are complete and before dataset registration. It writes
`manifests/dataset_version_manifest.json`, computes artifact checksums, records
source-system/source-kind/modality composition, and emits MLflow-ready
observability files.

Artifacts:

```text
deploy/dataset-finalization/
  Containerfile
  README.md
  job.yaml
```

The Job can backfill `provenance/dataset_samples.jsonl` from `stage2_eval.jsonl`
when the sidecar is missing, which keeps the local monolith and K8s execution
paths compatible.

## Seventh K8s Template: Webcrawler

The webcrawler is now represented as a first-class K8s source-acquisition
component. It builds from the in-repo `external/rag-crawler` copy, exposes a
ClusterIP FastAPI service for crawl/status APIs, stores crawler registries and
downloaded source assets on PVCs, and provides Job/CronJob templates for
per-collection crawls.

Artifacts:

```text
deploy/webcrawler/
  Containerfile
  README.md
  deployment.yaml
  crawl-job.yaml
  crawl-cronjob.yaml
```

Use the Deployment for ad hoc crawl requests, domain config management, and
status inspection. Use the Job/CronJob templates for deterministic source
collection refreshes that feed Stage 0. The service remains internal by default;
all external dependencies are configured through ConfigMap/Secret/PVC bindings.

## Eighth K8s Template: Stage 0 Corpus Prep

Stage 0 corpus prep is the first implemented source-pipeline Job after source
acquisition. It scrolls the Elasticsearch crawl/vector index, optionally joins
the crawler URL registry, reconstructs URL-grouped passages, emits
`source_revisions.jsonl` and `source_chunks.jsonl`, and writes MLflow-ready
observability files for crawl/passages/source provenance.

Artifacts:

```text
deploy/stage0-corpus-prep/
  Containerfile
  README.md
  job.yaml
```

Run one Job per source collection, such as `nim_curated` and
`nemo_usvcs_curated`, so recrawl retries and delta updates can be tracked as
separate pipeline child runs. Mount the crawler registry export as read-only and
pass `--url-registry /crawler-registry/<collection>_url_registry.json` when the
registry is available.

## Ninth K8s Template: Stage 1A Entailment Shards

Stage 1A entailment extraction is the first implemented sharded LLM Job. It
reads Stage 0 `passages.jsonl`, assigns passages to indexed pods by stable
`passage_id` hash, calls the configured OpenAI-compatible NIM endpoint, and
writes per-shard KVP rows plus `entailments` provenance sidecars.

Artifacts:

```text
deploy/stage1a-entailment-shards/
  Containerfile
  README.md
  job.yaml
```

The Job writes shard-specific outputs such as
`stage1a_le.shard-00000-of-00008.jsonl` and
`provenance/entailments.shard-00000-of-00008.jsonl`. A follow-on aggregation Job
should concatenate shard outputs into the monolithic filenames expected by the
remaining local pipeline stages.


## Tenth K8s Template: Data Designer Gap-Fill

Data Designer gap-fill is the native synthetic-generation handoff after Stage
1.5 coverage analysis. The Job consumes `provenance/gap_manifest.json` and
`data_designer/gapfill_requests.jsonl`, writes a seed CSV/submission plan, can
submit through the NeMo Data Designer SDK, and normalizes generated results back
into `stage1_5_gapfill.jsonl` plus provenance samples. The downstream Stage 2
and dataset-finalization path admits those samples by `sample_id`, preserving
Data Designer job IDs and gap IDs in the final dataset lineage.

Artifacts:

```text
deploy/data-designer-gapfill/
  Containerfile
  README.md
  job.yaml
```

The default manifest mode is `prepare` so clusters without the SDK package source
can still produce reviewable handoff artifacts. Use `submit` after building the
image with the NVIDIA SDK packages for the deployed NeMo Microservices version,
and use `collect` to ingest downloaded Data Designer result records.

## Eleventh K8s Template: NeMo Curator Handoff

NeMo Curator handoff is the native Stage 3 curation path. The Job prepares
`provenance/dataset_samples.jsonl` as Curator-readable JSONL, copies the Curator
config, collects retained/removed Curator records, writes `training.jsonl` and
`validation.jsonl`, and emits MLflow-ready observability. The native filter
Job now runs Curator `filter_documents` in the NeMo Curator container between
`prepare` and `collect`; larger GPU exact/fuzzy/semantic dedup
can use the same prepared input and output directories with a Curator-backed
Dask/Ray cluster.

Artifacts:

```text
deploy/curator/
  Containerfile
  README.md
  job.yaml
  native-filter-job.yaml

configs/curator/
  sft-dedup-quality.yaml
  sft-filter-documents.yaml
```

Dataset finalization records `curator/curation_manifest.json`, the Curator job
ID, Curator config hash, accepted/rejected sample artifacts, and split outputs
into the dataset version manifest.

## K8s Resource Guidance

| Workload | CPU | Memory | GPU | Storage |
|---|---:|---:|---:|---|
| TIES merge Nano r=16 | 4-8 | 24-32Gi | none initially | PVC or ephemeral + output PVC |
| Adapter inspection | 100m-500m | 512Mi | none | read-only adapter mount |
| Dataset registration | 500m-1 | 1-2Gi | none | workspace + Data Store access |
| Stage 0 corpus prep | 1-4 | 4-16Gi | none | output PVC/Data Store |
| Stage 1 generation shard | 1-4 | 2-8Gi | none client-side | output PVC/Data Store |
| Curator prepare/collect | 2-8 | 4-32Gi | none | output PVC |
| Native Curator dedup/filter | workload-dependent | workload-dependent | optional/likely for fuzzy/semantic | output PVC + Curator cache |

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
4. Evaluation matrix orchestration Job. Implemented in `deploy/evaluation-matrix/`.
5. Dataset finalization Job. Implemented in `deploy/dataset-finalization/`.
6. Dataset registration Job. Implemented in `deploy/dataset-registration/`.
7. Stage 0 corpus/provenance Job. Implemented in `deploy/stage0-corpus-prep/`.
8. Stage 1A entailment shard Job. Implemented in `deploy/stage1a-entailment-shards/`.
9. Stage 1B/1C generation shard Jobs.
10. Gap analysis Job and Data Designer submission. Gap manifest/Data Designer seed planning is implemented in `scripts/pipeline/stage1_5_gapfill.py`; prepare/collect K8s execution is implemented in `deploy/data-designer-gapfill/`, with live submission enabled when the NVIDIA SDK is included in the image.
11. Curator service integration. Curator handoff prepare/collect is implemented in `scripts/pipeline/curator_handoff.py` and `deploy/curator/`; native heuristic filtering is implemented in `deploy/curator/native-filter-job.yaml`, with GPU exact/fuzzy/semantic dedup left as a cluster-sized Curator extension.

This order gives immediate operational value while avoiding a large rewrite of
the source-grounded dataset pipeline.
