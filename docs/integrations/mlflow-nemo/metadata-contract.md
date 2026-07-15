# Metadata Contract

This contract defines metadata used by deployable MLflow-ready exports and by
the future integration wrapper. Deployable repo support covers dataset registration
observability and evaluation export. Parent/child run orchestration, Customizer
child runs, and promotion metadata are target fields for the wrapper.

| Metadata area | Status |
|---|---|
| Dataset registration observability fields | Deployable now |
| Evaluation export fields | Deployable now |
| Parent run, Customizer child run, and promotion fields | Future extension |

The purpose is to make any adapter answerable in both directions:

- From an MLflow run, find every NeMo object used to build and evaluate it.
- From a NeMo dataset, job, or model entity, find the MLflow run that recorded
  the decision trail.

## Naming

Use names already established by the repository.

| Object | Pattern | Example |
|---|---|---|
| MLflow experiment | `docs-to-data-to-lora` | `docs-to-data-to-lora` |
| Parent run name | `<collection>/<base-short>/r<rank>` | `domain_a_curated/llama-3.2-1b/r16` |
| Adapter name | `lora-<domain-short>-<base-short>-r<rank>` | `lora-domain-a-llama-3.2-1b-r16` |
| Entity Store dataset | `default/stage3-<collection-with-dashes>` | `default/stage3-domain-a-curated` |
| Entity Store output model | `default/<adapter-name>` | `default/lora-domain-a-llama-3.2-1b-r16` |
| Data Store dataset URI | `hf://datasets/default/<dataset-name>` | `hf://datasets/default/stage3-domain-a-curated` |

## Target MLflow Parent Run Tags

| Tag | Required | Example |
|---|---|---|
| `pipeline` | yes | `docs-to-data-to-lora` |
| `collection` | yes | `domain_a_curated` |
| `domain_short` | yes | `domain-a` |
| `base_model` | yes | `meta/llama-3.2-1b-instruct` |
| `base_short` | yes | `llama-3.2-1b` |
| `adapter_name` | yes | `lora-domain-a-llama-3.2-1b-r16` |
| `dataset_entity` | yes | `default/stage3-domain-a-curated` |
| `output_model_entity` | yes | `default/lora-domain-a-llama-3.2-1b-r16` |
| `promotion_status` | no | `candidate` |
| `nemo_workspace` | no | `default` |
| `project` | no | `docs-to-data-to-lora` |

## Target MLflow Parent Run Params

| Param | Example |
|---|---|
| `lora.rank` | `16` |
| `lora.alpha` | `32` |
| `train.epochs` | `2` |
| `train.batch_size` | `16` |
| `train.learning_rate` | `0.0001` |
| `train.seed` | `42` |
| `customizer.config_template` | `meta/llama-3.2-1b-instruct@v1.0.0+80GB` |
| `customizer.sequence_packing_enabled` | `false` |

## Dataset Registration Metadata

Run name: `dataset-registration`

Tags:

| Tag | Example |
|---|---|
| `nemo_data_store_uri` | `hf://datasets/default/stage3-domain-a-curated` |
| `nemo_entity_store_ref` | `default/stage3-domain-a-curated` |
| `dataset_format` | `hf` |
| `source_collection` | `domain_a_curated` |
| `dataset_version_id` | stable ID from `dataset_version_manifest.json` |
| `crawl_run_id` | stable crawl run ID |
| `delta_manifest_id` | stable delta manifest ID, if applicable |

Params:

| Param | Example |
|---|---|
| `rows.training` | `4860` |
| `rows.validation` | `540` |
| `rows.test` | `540` |
| `split.seed` | `42` |
| `split.train_ratio` | `0.90` |
| `source_revisions.count` | `128` |
| `source_chunks.count` | `1024` |
| `entailments.count` | `4096` |
| `samples.synthetic.count` | `320` |

Artifacts:

| Artifact | Source |
|---|---|
| `dataset/training.manifest.json` | File name, rows, byte size, checksum |
| `dataset/validation.manifest.json` | File name, rows, byte size, checksum |
| `dataset/test.manifest.json` | Optional test-set manifest |
| `provenance/dataset_version_manifest.json` | Dataset version, source counts, sample counts, checksums |
| `provenance/dataset_samples.jsonl` | Per-sample lineage to entailments/source chunks |
| `provenance/entailments.jsonl` | Logical entailments extracted from crawled docs |
| `provenance/source_revisions.jsonl` | URL revision fingerprints |
| `provenance/source_chunks.jsonl` | Chunk fingerprints and offsets |
| `provenance/delta_manifest.json` | Re-crawl delta summary, if present |
| `provenance/gap_manifest.json` | Gap/bias selections for Data Designer, if present |
| `dataset/bias_report.json` | Stage 2 bias report, if present |
| `dataset/validation_report.json` | Stage 2 validation report, if present |

If using MLflow dataset tracking, log the NeMo dataset URI as the dataset
source and attach context `training`, `validation`, or `evaluation`.

## Target Customizer Child Run

Run name: `customizer-training`

Tags:

| Tag | Example |
|---|---|
| `nemo_customizer_job_id` | `cust-...` |
| `nemo_customizer_status` | `completed` |
| `nemo_output_model_entity` | `default/lora-domain-a-llama-3.2-1b-r16` |
| `nemo_output_path` | `hf://models/default/lora-domain-a-llama-3.2-1b-r16` |

Metrics:

| Metric | Source |
|---|---|
| `train.loss` | Customizer job detail, if available |
| `train.val_loss` | Customizer job detail, if available |
| `train.lr` | Customizer job detail, if available |
| `train.grad_norm` | Customizer job detail, if available |
| `train.wall_time_s` | Wrapper-measured duration |

Artifacts:

| Artifact | Source |
|---|---|
| `customizer/job_payload.json` | Exact request submitted to Customizer |
| `customizer/job_final.json` | Final job detail response |

## Evaluation Export Metadata

Run name: `evaluator-matrix`

Tags:

| Tag | Example |
|---|---|
| `nemo_evaluator_dataset` | `default/stage3-nim-curated-test` |
| `eval_scope` | `singleaxis,pairwise,reference` |

Artifacts:

| Artifact | Source |
|---|---|
| `evaluator/jobs.json` | List of submitted Evaluator job payloads and IDs |
| `evaluator/results/<job_id>.json` | Raw Evaluator result response |
| `evaluator/summary.json` | Normalized aggregate metrics |
| `evaluator/export_refs.json` | Native Evaluator MLflow export run IDs, when available |

Metrics:

| Metric | Example |
|---|---|
| `eval.accuracy.mean` | `4.62` |
| `eval.completeness.mean` | `4.45` |
| `eval.faithfulness.mean` | `4.71` |
| `eval.clarity.mean` | `4.83` |
| `eval.win_rate_vs_base` | `0.68` |
| `eval.win_rate_vs_reference` | `0.54` |

## NeMo Resource Back-References

When a NeMo schema allows free-form metadata, include these fields:

| Field | Value |
|---|---|
| `mlflow_tracking_uri` | Tracking server URI |
| `mlflow_experiment` | `docs-to-data-to-lora` |
| `mlflow_run_id` | Parent run ID |
| `mlflow_parent_run_name` | `<collection>/<base-short>/r<rank>` |
| `adapter_name` | Adapter name |
| `source_collection` | ES collection name |
| `dataset_version_id` | Stable dataset version ID |
| `mlflow_parent_run_id` | Parent run for dataset/model build |

For Entity Store datasets, include the MLflow run ID in `description` if no
structured custom field is available in the deployed schema.
