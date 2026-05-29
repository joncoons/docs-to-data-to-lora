# MLflow Observability Contract

This contract defines how the redesigned NeMo/NIM showcase should emit
observability into MLflow without making MLflow the owner of NeMo resources.

NeMo Data Store, Entity Store, Data Designer, Customizer, Evaluator, and NIM
remain the execution and artifact planes. MLflow is the audit plane that links
dataset versions, job IDs, model artifacts, evaluation results, and promotion
decisions.

## Run Model

Use one MLflow parent run per durable dataset/model build.

```text
parent run: <collection>/<build-or-adapter-name>
  child: stage0-corpus-prep
  child: stage1a-entailment-shard(s)
  child: dataset-lineage
  child: data-designer-gapfill
  child: customizer-training
  child: ties-merge
  child: adapter-inspection
  child: evaluator-registration
  child: evaluator-matrix
  child: evaluator-results
  child: promotion
```

Customizer and Evaluator may also export directly to MLflow when the deployed
NeMo version supports it. In that case, keep a repository-owned wrapper or
collector run that logs the NeMo job IDs and the exported MLflow run IDs so both
views can be reconciled.

## Kubernetes Pattern

Do not require every lightweight Job image to install the MLflow client.

Each K8s Job should write structured observability files to its output PVC:

```text
/outputs/observability/<job-name>/
  run_context.json
  metrics.json
  artifacts_manifest.json
  service_refs.json
```

A dedicated MLflow export Job can later upload those files and metrics to
MLflow. For Jobs that already need the MLflow client, direct logging is fine as
long as the same JSON files are still emitted for replay/debugging.

Standard environment variables:

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | Usually `docs-to-data-to-lora` |
| `MLFLOW_PARENT_RUN_ID` | Existing parent run to attach child runs to |
| `MLFLOW_RUN_NAME` | Explicit child run name when direct logging |
| `OBSERVABILITY_DIR` | Output directory for structured JSON files |
| `PIPELINE_RUN_ID` | Stable pipeline/build identifier |
| `DATASET_VERSION_ID` | Stable dataset version from provenance manifests |

## Stage 0 Corpus Prep

Stage 0 is repository-owned because it reconstructs source-grounded passages
from the crawl/vector index. It should emit observability files before any
entailment extraction or dataset registration happens.

Tags:

| Tag | Example |
|---|---|
| `pipeline.stage` | `stage0-corpus-prep` |
| `source_collection` | `nim_curated` |
| `crawl_run_id` | stable crawl run ID |
| `es_index` | `nim_curated` |

Metrics:

| Metric | Meaning |
|---|---|
| `stage0.raw_hits.count` | ES hits read from the crawl/vector index |
| `stage0.chunks.extracted` | Chunks with usable URL/text fields |
| `stage0.passages.count` | Reconstructed passages after filtering |
| `stage0.source_revisions.count` | Source revision records emitted |
| `stage0.source_chunks.count` | Source chunk records emitted |
| `stage0.urls.count` | Distinct source URLs retained |
| `stage0.tokens.total` | Passage token total |
| `stage0.tokens.mean` | Mean passage token count |
| `stage0.doc_kind.html` | HTML/grouped-web passage count |
| `stage0.doc_kind.pdf` | Per-chunk source text count, including PDF/binary, captured document, image caption, audio transcript, and video summary text |
| `stage0.es_provenance.chunks.count` | Extracted ES chunks carrying source-agnostic provenance or upstream IDs |
| `stage0.es_provenance.source_revisions.count` | Distinct upstream source revision IDs observed in ES metadata |
| `stage0.es_provenance.source_chunks.count` | Distinct upstream source chunk IDs observed in ES metadata |
| `stage0.url_registry.records.count` | URL registry records loaded |
| `stage0.url_registry.matched_urls.count` | Retained Stage 0 URLs with a registry match |
| `stage0.url_registry.unmatched_urls.count` | Retained Stage 0 URLs missing a registry match |
| `stage0.url_registry.content_hash.count` | Registry records with a valid source content hash |
| `stage0.url_registry.etag.count` | Registry records with an ETag |
| `stage0.url_registry.last_modified.count` | Registry records with Last-Modified metadata |
| `stage0.url_registry.redirects.count` | Registry records that indicate redirect/final URL metadata |

Artifacts:

| Artifact | Source |
|---|---|
| `passages.jsonl` | Stage 0 passage output |
| `manifests/crawl_run.json` | Crawl/run manifest |
| `provenance/source_revisions.jsonl` | Source revision sidecar |
| `provenance/source_chunks.jsonl` | Source chunk sidecar |
| `<collection>_url_registry.json` | Optional crawler URL registry input artifact |

## Stage 1A Entailment Extraction

Stage 1A is repository-owned because it turns source passages into logical
entailments and source-grounded KVP rows. In Kubernetes, each shard should emit
its own observability files; an aggregation Job can later emit combined metrics
for the full collection.

Tags:

| Tag | Example |
|---|---|
| `pipeline.stage` | `stage1a-le-kvp` |
| `source_collection` | `nim_curated` |
| `shard_label` | `shard-00000-of-00008` |
| `llm_model` | `nvidia/nemotron-3-super-120b-a12b` |

Metrics:

| Metric | Meaning |
|---|---|
| `stage1a.passages.input.count` | Total Stage 0 passages visible to the shard |
| `stage1a.passages.selected.count` | Passages assigned to this shard |
| `stage1a.rows.count` | KVP rows emitted by the shard |
| `stage1a.entailments.count` | Distinct entailment IDs emitted |
| `stage1a.source_revisions.count` | Distinct source revisions referenced |
| `stage1a.source_chunks.count` | Distinct source chunks referenced |
| `stage1a.source_system.<name>.rows` | Rows by upstream source producer, such as `web_crawl` or `image_dense_caption` |
| `stage1a.source_kind.<name>.rows` | Rows by upstream source kind, such as `web_page` or `dense_caption` |
| `stage1a.modality.<name>.rows` | Rows by source modality, such as `text`, `document`, `image`, `audio`, or `video` |
| `stage1a.shard.index` | Numeric shard index |
| `stage1a.shard.count` | Total shard count |

Artifacts:

| Artifact | Source |
|---|---|
| `stage1a_le.<shard>.jsonl` | Stage 1A shard KVP rows |
| `provenance/entailments.<shard>.jsonl` | Entailment provenance sidecar |
| `passages.jsonl` | Stage 0 input passage artifact reference |

## Dataset Finalization And Registration

Dataset finalization closes the lineage gap before registration. It writes
`manifests/dataset_version_manifest.json` from split files and provenance
sidecars, then registration publishes the split files plus recognized lineage
sidecars to NeMo Data Store and registers Entity Store dataset refs. Log these
as MLflow params/tags/artifacts and include back-references in Entity Store
metadata when the deployed schema allows it.

Tags:

| Tag | Example |
|---|---|
| `pipeline.stage` | `dataset-finalization` or `dataset-registration` |
| `source_collection` | `nim_curated` |
| `dataset_entity` | `default/stage3-nim-curated` |
| `nemo_data_store_uri` | `hf://datasets/default/stage3-nim-curated` |
| `nemo_entity_store_ref` | `default/stage3-nim-curated` |
| `dataset_version_id` | stable ID from `dataset_version_manifest.json` |
| `crawl_run_id` | stable crawl run ID |
| `delta_manifest_id` | stable delta manifest ID, if applicable |

Metrics:

| Metric | Meaning |
|---|---|
| `dataset.rows.training` | Training rows |
| `dataset.rows.validation` | Validation rows |
| `dataset.rows.test` | Test rows |
| `dataset.samples.provenance.count` | Samples found in `provenance/dataset_samples.jsonl` |
| `dataset.sources.count` | Distinct source URLs in sample lineage |
| `dataset.source_revisions.count` | Distinct source revisions in sample lineage |
| `dataset.source_chunks.count` | Distinct source chunks in sample lineage |
| `dataset.entailments.count` | Extracted logical entailments in sample lineage |
| `dataset.samples.synthetic.count` | Data Designer/generated samples |
| `dataset.samples.grounded.count` | Source-grounded samples |
| `dataset.synthetic_ratio` | Synthetic samples / total samples |
| `dataset.lineage_files.uploaded.count` | Lineage files uploaded with dataset bytes |
| `dataset.source_system.<name>.samples` | Samples by upstream source producer or synthetic producer |
| `dataset.source_kind.<name>.samples` | Samples by upstream source kind |
| `dataset.modality.<name>.samples` | Samples by source modality |
| `dataset.delta.added_chunks` | Added chunks since previous crawl |
| `dataset.delta.changed_chunks` | Changed chunks since previous crawl |
| `dataset.delta.deleted_chunks` | Deleted chunks since previous crawl |
| `dataset.gap_count` | Coverage gaps selected for generation |

Artifacts:

| Artifact | Source |
|---|---|
| `manifests/crawl_run.json` | Crawl/run manifest |
| `provenance/source_revisions.jsonl` | Source revision sidecar |
| `provenance/source_chunks.jsonl` | Source chunk sidecar |
| `provenance/entailments.jsonl` | Extracted entailments |
| `provenance/dataset_samples.jsonl` | Final sample lineage |
| `manifests/dataset_version_manifest.json` | Dataset version summary |
| `provenance/delta_manifest.json` | Re-crawl delta summary |
| `provenance/gap_manifest.json` | Gap/bias selection for Data Designer |
| `dataset/training.manifest.json` | Rows, bytes, checksum |
| `dataset/validation.manifest.json` | Rows, bytes, checksum |
| `dataset/test.manifest.json` | Rows, bytes, checksum |

Registration observability should mark uploaded split files as
`uploaded_dataset_file`, uploaded lineage sidecars as `uploaded_lineage_file`,
and include the uploaded lineage file list in `service_refs.json`.

## Data Designer

Synthetic generation should be visibly separate from source-grounded data.

Tags:

| Tag | Example |
|---|---|
| `pipeline.stage` | `data-designer-gapfill` |
| `nemo_data_designer_job_id` | deployed service job ID |
| `gap_manifest_id` | source gap selection ID |

Metrics:

| Metric | Meaning |
|---|---|
| `gapfill.gaps.requested` | Number of gaps submitted |
| `gapfill.seed_records.count` | Seed records handed to Data Designer |
| `gapfill.records.requested` | Data Designer records requested |
| `gapfill.pairs.requested` | Target pair count from the gap manifest |
| `gapfill.samples.generated` | Candidate synthetic samples |
| `gapfill.samples.accepted` | Samples admitted after QA |
| `gapfill.acceptance_rate` | Accepted/generated |

Artifacts:

| Artifact | Source |
|---|---|
| `data_designer/request_manifest.json` | Planned Data Designer handoff summary |
| `data_designer/gapfill_requests.jsonl` | Per-gap seed records with retrieved chunks and generation briefs |
| `data_designer/seed_dataset.csv` | Datastore-ready seed dataset for Data Designer |
| `data_designer/submission_plan.json` | Exact native submission plan and model settings |
| `data_designer/request.json` | Exact submitted generation request, if exported by the SDK path |
| `data_designer/result_manifest.json` | Returned job/result metadata |
| `data_designer/generated_samples.jsonl` | Generated samples with source gap refs |
| `provenance/data_designer_samples.jsonl` | Provenance sample sidecar for Data Designer outputs |

## Customizer

Prefer native Customizer MLflow export when available. The repository wrapper or
collector should still log:

| Type | Name |
|---|---|
| tag | `nemo_customizer_job_id` |
| tag | `nemo_output_model_entity` |
| tag | `nemo_output_model_uri` |
| param | `customizer.template_ref` |
| param | `dataset_entity` |
| metric | `train.loss` |
| metric | `train.val_loss` |
| metric | `train.wall_time_s` |
| artifact | `customizer/job_payload.json` |
| artifact | `customizer/job_final.json` |

When Customizer exports directly to MLflow, also log:

| Type | Name |
|---|---|
| tag | `customizer_mlflow_run_id` |
| tag | `customizer_mlflow_experiment_id` |

## TIES Merge and Adapter Inspection

These are repository-owned Jobs, so they should emit observability directly or
through the export Job.

Metrics:

| Metric | Meaning |
|---|---|
| `ties.source_adapters.count` | Number of adapters merged |
| `ties.trim_ratio` | TIES trim ratio |
| `adapter.tensor_count` | Safetensors tensor count |
| `adapter.lora_a_count` | LoRA A tensor count |
| `adapter.lora_b_count` | LoRA B tensor count |
| `adapter.non_lora_count` | Non-LoRA tensor count |
| `adapter.total_elements` | Total tensor elements |

Artifacts:

| Artifact | Source |
|---|---|
| `ties/source_adapters.json` | Source adapter refs and revisions |
| `ties/merge_config.json` | Merge parameters |
| `adapter_inspection/report.json` | Structural inspection report |

## Evaluator

Prefer native Evaluator MLflow export when available. The repository should
still persist and log the control-plane references.

Tags:

| Tag | Example |
|---|---|
| `pipeline.stage` | `evaluator-matrix` |
| `evaluator_config_singleaxis` | `default/stage3-singleaxis-rubric` |
| `evaluator_config_pairwise` | `default/stage3-pairwise-tournament` |

Metrics:

| Metric | Meaning |
|---|---|
| `eval.jobs.wave_a.count` | Single-axis job count |
| `eval.jobs.wave_b.count` | LoRA pairwise job count |
| `eval.jobs.wave_c.count` | 49B comparator job count |
| `eval.accuracy.mean` | Normalized accuracy mean |
| `eval.completeness.mean` | Normalized completeness mean |
| `eval.faithfulness.mean` | Normalized faithfulness mean |
| `eval.win_rate_vs_base` | Pairwise win rate over base |
| `eval.win_rate_vs_49b` | Pairwise win rate over 49B comparator |

Artifacts:

| Artifact | Source |
|---|---|
| `evaluator/targets.json` | Registered targets |
| `evaluator/configs.json` | Registered configs |
| `evaluator/jobs.json` | Submitted job IDs and payloads |
| `evaluator/results/<job_id>.json` | Raw Evaluator results |
| `evaluator/summary.json` | Aggregated metrics |

When Evaluator exports directly to MLflow, also log:

| Type | Name |
|---|---|
| tag | `evaluator_mlflow_run_id` |
| tag | `evaluator_mlflow_experiment_id` |

## Promotion

Promotion decisions should use MLflow tags plus immutable artifacts.

Tags:

| Tag | Values |
|---|---|
| `promotion_status` | `candidate`, `approved`, `rejected`, `deployed` |
| `promotion_reason` | short machine-readable reason |

Artifacts:

| Artifact | Source |
|---|---|
| `promotion/decision.json` | Thresholds, metrics, decision |
| `promotion/deployment_manifest.yaml` | NIM/NIM Proxy deployment ref |

## Implementation Order

1. Add Stage 0 corpus prep observability files. Implemented in `deploy/stage0-corpus-prep/`.
2. Add Stage 1A entailment shard observability files. Implemented in `deploy/stage1a-entailment-shards/`.
3. Add dataset finalization and registration observability files. Implemented in `deploy/dataset-finalization/` and `deploy/dataset-registration/`.
4. Add result collection and MLflow export for Evaluator job results.
5. Add Customizer exporter reconciliation tags when training is refactored.
6. Add Data Designer gapfill job IDs and generated-sample lineage.
7. Add TIES/adapter inspection observability JSON files.
