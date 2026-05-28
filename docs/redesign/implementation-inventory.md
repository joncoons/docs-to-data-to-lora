# Implementation Inventory and Redesign Classification

This inventory classifies the current repository before the native-alignment
refactor. It preserves the intentional custom components while identifying the
parts that should move to NVIDIA-native services, SDKs, or deployment patterns.

Backup taken before this inventory:

```text
/home/joncoons/claude/backups/docs-to-data-to-lora-20260528-131428
```

## Classification Legend

| Status | Meaning |
|---|---|
| Keep custom | Domain-specific value that should remain custom. |
| Keep, refactor | Useful custom logic, but it needs better boundaries, config, tests, or provenance. |
| Replace native | Current custom code should become a native NeMo/NIM service path. |
| Quarantine | Keep for historical reference or temporary fallback, but remove from the showcase path. |
| Remove generated | Generated/runtime artifacts that should not be part of the durable repo. |

## Redesign Boundary

The protected custom layer is the docs-to-reference-data method:

```text
curated crawl
  -> source normalization
  -> source chunking
  -> logical entailment extraction
  -> source-grounded sample creation
  -> coverage and gap analysis
```

The NVIDIA-native execution layer should own the platform services:

```text
NeMo Data Designer
NeMo Curator
NeMo Data Store
NeMo Entity Store
NeMo Customizer
NeMo Evaluator
NIM / NIM Proxy / NIM Operator
```

Custom code should connect these layers, not replace the platform services.

## Current Components

### Stage 1: Curated Crawl

| Path | Classification | Notes | Target Action |
|---|---|---|---|
| `scripts/sitemap_to_inventory.py` | Keep custom | Intentional sitemap inventory and version-scope logic. | Preserve. Add provenance fields for inventory hash and canonical URL policy. |
| `docs/stage-1-curated-crawl.md` | Keep custom | Strong methodology doc. | Update to reference provenance contract and delta recrawls. |
| `examples/nim.md` | Keep custom | Real curation example. | Preserve as showcase input. |
| `examples/nemo-microservices.md` | Keep custom | Real curation example. | Preserve as showcase input. |
| `deployment/kubernetes-runai.md` | Keep, refactor | Useful environment note, but deployment target should be separated from showcase architecture. | Move environment-specific detail under `docs/runbooks/` later. |
| `docs/integrations/*` | Keep, refactor | Useful optional extraction/chunking enhancements. | Reframe as extraction strategy options, with NeMo Retriever Extraction called out where applicable. |

### Stage 2: Source-Grounded Dataset Creation

| Path | Classification | Notes | Target Action |
|---|---|---|---|
| `scripts/build_v2_dataset.py` | Keep, refactor | Useful orchestrator, but currently mixes execution, state, output format, and stage ownership. | Convert to a thin pipeline driver that writes provenance ledgers and delegates native services where appropriate. |
| `scripts/pipeline/stage0_corpus_prep.py` | Keep, refactor | Corpus reconstruction is intentional and now emits source provenance plus Job observability. | Run through `deploy/stage0-corpus-prep/`; next add delta-aware previous-crawl inputs. |
| `scripts/pipeline/stage1a_le_kvp.py` | Keep, refactor | Logical entailment extraction is core custom value and now supports K8s sharding plus shard observability. | Run through `deploy/stage1a-entailment-shards/`; next add aggregation and richer evidence spans. |
| `scripts/pipeline/stage1b_synthesis.py` | Keep, refactor | Cross-passage synthesis can remain custom if evidence is tracked. | Treat as source-grounded synthesis with multi-chunk evidence lineage. |
| `scripts/pipeline/stage1c_instruction.py` | Keep, refactor | Instruction diversity is useful if grounded. | Add entailment/sample lineage and task taxonomy. |
| `scripts/pipeline/stage1_5_gapfill.py` | Split | Bias/gap analysis is custom value; generation is not native today despite the Data Designer label. | Keep coverage analysis. Replace direct LLM gap-fill with `gap_manifest -> NeMo Data Designer`. |
| `scripts/pipeline/stage2_qa_eval.py` | Keep, refactor | Local refinement/quality gate can remain, but should not obscure NeMo Evaluator. | Add provenance-preserving status updates. Consider later Evaluator-backed dataset quality jobs. |
| `scripts/pipeline/stage3_curator.py` | Replace native | File says Curator, but implementation is pure Python exact/MinHash/filter/split. | Move generic dedup/filter to NeMo Curator. Keep local fallback under legacy if needed. |
| `scripts/pipeline/stage4_validation.py` | Keep, refactor | Independent judge gate is useful. | Record validator model/prompt hashes in `entailment` or `dataset_sample` quality metadata. |
| `scripts/pipeline/models.py` | Keep, refactor | Current models are stage-output oriented and lack durable lineage. | Add provenance models or map to `schemas/provenance`. |
| `scripts/pipeline/prompts.py` | Keep custom | Prompt/rubric assets are intentional domain logic. | Add prompt hash generation and version naming. |
| `scripts/pipeline/llm_client.py` | Keep, refactor | Useful generic OpenAI-compatible client. | Config-only endpoints; no hidden local cluster assumptions. |
| `scripts/pipeline/es_client.py` | Keep, refactor | ES is current corpus/vector substrate. | Keep as corpus-source adapter; do not let ES become the provenance source of truth. |
| `scripts/pipeline/config.py` | Quarantine/refactor | Contains local cluster service defaults and kubectl secret fetching. | Replace with explicit environment config files and secret inputs. Keep local runbook examples separate. |

### Data Designer Assets

| Path | Classification | Notes | Target Action |
|---|---|---|---|
| `data-designer-recipes/nim-gapfill.yaml` | Keep, refactor | Useful recipe asset, but not currently wired to Data Designer execution. | Move under `configs/data-designer/` and parameterize from `gap_manifest`. |
| `data-designer-recipes/nemo-usvcs-gapfill.yaml` | Keep, refactor | Same as above. | Move under `configs/data-designer/` and attach job/result lineage. |

Target Data Designer flow:

```text
source-grounded dataset
  -> coverage analysis
  -> gap_manifest.json
  -> NeMo Data Designer job
  -> synthetic_gapfill samples
  -> separate synthetic dataset version
```

### Dataset Registration and Storage

| Path | Classification | Notes | Target Action |
|---|---|---|---|
| `scripts/eval/upload_test_datasets.py` | Replace native | Hard-coded NodePorts, Gitea credentials, and direct git operations. | Replace with configured NeMo Data Store/Entity Store registration helper using environment config and provenance manifests. |
| `scripts/stage3/build_moe_shards.py` | Split | Shard construction is useful; Data Store/Entity Store integration is hard-coded. | Keep shard planning as custom. Replace registration/push code with shared native dataset registrar. |
| `scripts/stage3/ties_merge.py` | Keep, refactor | Adapter merge may be legitimate experiment logic. | Remove embedded Data Store credentials and make artifact lineage explicit. |
| `docs/integration-templates/mlflow-nemo/*` | Keep, refactor | Useful orchestration plan. | Update after native alignment so MLflow tracks NeMo IDs and provenance manifests, not accidental REST wrappers. |

Target registration rule:

```text
dataset bytes and manifests live in Data Store
dataset identity and summary metadata live in Entity Store
row-level provenance lives in registered files, not only custom_fields
```

### Customizer Training

| Path | Classification | Notes | Target Action |
|---|---|---|---|
| `scripts/stage3/customizer_client.py` | Replace native/fallback | Thin REST wrapper with version uncertainty. | Prefer NeMo Platform SDK/CLI. Keep as legacy fallback only if SDK coverage is insufficient. |
| `scripts/stage3/train_adapter.py` | Keep, refactor | Job spec planning is useful, but direct payload shape is brittle. | Convert to config builder for native Customizer client. |
| `scripts/stage3/train_adapter_moe.py` | Keep, refactor | MoE-specific adapter training may be intentional. | Isolate MoE experiment logic from platform submission mechanics. |
| `scripts/stage3/customizer-templates/*` | Keep, refactor | Captures real cluster template lessons. | Move under `configs/customizer/templates/` and document version compatibility. |
| `scripts/stage3/holdout_split.py` | Keep, refactor | Split logic likely useful. | Align with `dataset_version_manifest` split metadata. |
| `scripts/stage3/models.py` | Keep, refactor | Adapter spec models are useful. | Add dataset/model entity refs and provenance IDs. |
| `scripts/stage3/moe_models.py` | Keep, refactor | MoE spec models are useful. | Same as above. |
| `scripts/stage3/*orchestrator*.py` | Quarantine/refactor | Operational one-offs around specific training runs. | Move stable concepts into runbooks/configs; keep run-specific scripts as legacy. |
| `scripts/stage3/*.md` | Keep as runbooks | Captures real operational issues. | Move to `docs/runbooks/customizer/` later. |

### Evaluation and Serving

| Path | Classification | Notes | Target Action |
|---|---|---|---|
| `deploy/rag-oai-proxy/*` | Quarantine/replace native | Explicitly identified as Claude-generated workaround. | Replace showcase path with Evaluator targets pointing at NIM Proxy. Keep only as temporary fallback. |
| `scripts/eval/register_evaluator_entities.py` | Keep | Builder functions now register native NIM Proxy model targets and configs. | Run as the K8s control-plane Job in `deploy/evaluator-registration/`; keep custom proxy path legacy only. |
| `scripts/eval/evaluator_client.py` | Replace native/fallback | Thin REST wrapper. | Prefer NeMo Evaluator SDK/CLI. Keep as fallback if needed. |
| `scripts/eval/run_evaluation_matrix.py` | Keep | Evaluation matrix is useful showcase logic and now has K8s-friendly submission/polling controls. | Run through `deploy/evaluation-matrix/`; add result collection as the next separate Job. |
| `scripts/eval/bake_context_into_testset.py` | Split | Context-baking is useful for controlled comparisons, but not a replacement for RAG targets. | Keep as baseline mode. Add native RAG target mode via Evaluator. |
| `scripts/eval/adapter_sync.py` | Replace native where possible | Sidecar sync assumes path-based LoRA serving. | Prefer NIM LoRA/NIM Operator/NIM Proxy patterns. Keep only for local deployment constraints. |

Target native evaluation flow:

```text
NeMo Evaluator
  -> model target or RAG target
    -> NIM Proxy /v1/chat/completions
      -> base NIM or LoRA-enabled NIM
```

Use Evaluator target configuration and interceptors for payload/response
adaptation before introducing any custom proxy.

### Tests and Generated Artifacts

| Path | Classification | Notes | Target Action |
|---|---|---|---|
| `tests/*` | Keep, refactor | Good test footprint, but currently stale around some moved/renamed behavior and missing deps. | Reset baseline after phase 1. Add provenance schema/model tests first. |
| `docs_to_data_to_lora.egg-info/*` | Remove generated | Build metadata should not be part of the durable source tree. | Remove or ignore in a cleanup phase. |
| `scripts/**/__pycache__`, `tests/**/__pycache__` | Remove generated | Runtime cache artifacts. | Remove and ensure ignored. |
| `.pytest_cache`, `.ruff_cache` | Remove generated | Tool caches. | Remove and ensure ignored. |
| `backups/*.yaml` | Quarantine | Local cluster backup YAMLs. | Move to `docs/runbooks/artifacts/` or external backup storage if still useful. |

## Refactor Phases

### Phase 0: Baseline and Preservation

Status: in progress.

- Create backup copy of current directory.
- Add provenance and delta-update contract.
- Add implementation inventory.
- Do not change runtime pipeline behavior yet.

### Phase 1: Provenance-First Source-Grounded Dataset

Goal: keep current outputs working while adding durable lineage.

Tasks:

- Add Python provenance models corresponding to `schemas/provenance`.
- Add deterministic ID/hash helpers.
- Extend Stage 0 to emit `source_revisions.jsonl` and `source_chunks.jsonl`. Implemented, with K8s Job observability, native ES provenance consumption for source-agnostic revision/chunk IDs, and optional crawler URL registry fallback/enrichment for source-content hashes, HTTP metadata, and registry coverage metrics.
- Vendor an editable webcrawler integration copy under `external/rag-crawler`. Implemented, with source-agnostic ES provenance helpers for web, document, image, audio, and video text sources.
- Add K8s-native webcrawler deployment and crawl Job/CronJob templates. Implemented under `deploy/webcrawler`, using PVCs, ConfigMap/Secret wiring, and an internal ClusterIP service.
- Extend Stage 1A to emit `entailments.jsonl`. Implemented, with K8s shard outputs, observability, and source-system/source-kind/modality lineage propagation from Stage 0 passages.
- Extend sample generation to emit `dataset_samples.jsonl`. Implemented, with per-sample source composition fields available to dataset registration observability.
- Preserve current `training.jsonl` and `validation.jsonl` outputs for compatibility.
- Add schema validation tests.

### Phase 2: Gap Manifest and Native Data Designer

Goal: make Data Designer the actual synthetic generation path.

Tasks:

- Split `stage1_5_gapfill.py` into coverage analysis and synthetic execution.
- Emit `gap_manifest.json` from coverage analysis.
- Parameterize Data Designer recipes from gap records.
- Store Data Designer job IDs and generated sample lineage.
- Keep direct LLM gap-fill only under legacy/fallback.

### Phase 3: Native NeMo Curator

Goal: replace pure-Python generic curation with NeMo Curator in the showcase path.

Tasks:

- Define Curator config for exact, fuzzy, semantic dedup, and quality filters.
- Run Curator against `dataset_samples.jsonl` or a normalized intermediate.
- Record Curator job/config hash in `dataset_version_manifest`.
- Keep local dedup as fallback for offline testing only.

### Phase 4: Native Dataset Registration

Goal: remove hard-coded Gitea/NodePort registration code.

Tasks:

- Create shared dataset registrar config.
- Upload dataset bytes and manifests to NeMo Data Store.
- Register dataset entities with Entity Store using lineage pointers.
- Remove credentials from source code.
- Add dry-run output for reviewable payloads.

### Phase 5: Native Customizer

Goal: submit training through stable NeMo-native clients/configs.

Tasks:

- Convert adapter specs into versioned Customizer config files.
- Prefer NeMo SDK/CLI for job submission.
- Track Customizer job ID and output model entity in dataset/model lineage.
- Move MoE run-specific orchestration into experiment configs and runbooks.

### Phase 6: Native Evaluator and NIM Proxy

Goal: remove `rag-oai-proxy` from the default showcase path.

Tasks:

- Register Evaluator targets pointing directly to NIM Proxy.
- Use Evaluator model targets for base/LoRA comparisons.
- Use Evaluator RAG targets for native retrieval-aware evaluation where applicable.
- Replace `<think>` stripping and payload tweaks with Evaluator config/interceptors where possible.
- Keep context-baked datasets only as controlled baseline mode.

### Phase 7: Repo Cleanup and Documentation

Goal: make the repository read as a NeMo/NIM showcase, not a run artifact dump.

Tasks:

- Rewrite README around the target architecture.
- Move local-cluster runbooks under `docs/runbooks/`.
- Move stable configs under `configs/`.
- Move legacy scripts under `scripts/legacy/`.
- Remove generated artifacts and caches.
- Fix tests and dependencies.

## Immediate Phase 1 File Targets

The first code refactor should be narrow and non-disruptive.

Add:

```text
scripts/pipeline/provenance.py
scripts/pipeline/provenance_io.py
tests/test_provenance_models.py
tests/test_provenance_ids.py
```

Modify:

```text
scripts/pipeline/stage0_corpus_prep.py
scripts/pipeline/stage1a_le_kvp.py
scripts/pipeline/models.py
scripts/build_v2_dataset.py
```

Do not modify yet:

```text
deploy/rag-oai-proxy/*
scripts/stage3/*
scripts/eval/*
```

Those are later phases after source-grounded lineage is stable.

## Open Decisions

- Whether to store provenance JSONL, Parquet, or both.
- Whether chunk text should be embedded in `source_chunks.jsonl` or stored by pointer for large binary corpora.
- Whether current Elasticsearch chunk IDs are stable enough to use as external source IDs.
- Whether the external Claude validation gate remains outside NeMo Evaluator or becomes an Evaluator-backed dataset-quality job.
- Whether MoE adapter experiments are part of the core showcase or an advanced appendix.
