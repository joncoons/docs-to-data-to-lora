# MLflow + NeMo Orchestration Template

This template plans the integration discussed in the MLflow/NeMo notes:
MLflow acts as the orchestration and audit layer, while NeMo Microservices
remain the execution plane for dataset registration, LoRA customization,
evaluation, and adapter artifact storage.

The intent is not to make MLflow pretend to own NeMo resources. Instead, each
MLflow run records durable cross-references to the NeMo objects that did the
work.

## Fit With This Repository

The current repository already has most of the NeMo-side operations:

| Repository area | Existing role | Integration use |
|---|---|---|
| `scripts/build_v2_dataset.py` | Builds Stage 2 `training.jsonl` and `validation.jsonl` | Input step for a tracked MLflow run |
| `scripts/stage3/train_adapter.py` | Builds and submits NeMo Customizer LoRA jobs | Called by the MLflow orchestrator |
| `scripts/stage3/build_moe_shards.py` | Creates shard datasets in NeMo Data Store and Entity Store | Reused for MoE adapter dataset registration |
| `scripts/eval/upload_test_datasets.py` | Uploads held-out test sets to NeMo Data Store and registers Entity Store datasets | Reused for evaluation dataset lineage |
| `scripts/eval/register_evaluator_entities.py` | Builds Evaluator target/config payloads and dataset payload shape | Source of canonical Evaluator and dataset metadata |
| `scripts/eval/run_evaluation_matrix.py` | Submits Evaluator jobs for adapter/base/RAG comparisons | Called after Customizer jobs complete |

## Directory Contents

| File | Use |
|---|---|
| [integration-plan.md](integration-plan.md) | End-to-end architecture, phases, and ownership boundaries |
| [metadata-contract.md](metadata-contract.md) | Required MLflow params, tags, artifacts, and NeMo IDs |
| [mlflow-orchestrator-template.md](mlflow-orchestrator-template.md) | Python orchestration skeleton and run layout |
| [config.example.yaml](config.example.yaml) | Environment-specific config skeleton |

## High-Level Flow

```text
Stage 1 ES corpus
    |
    v
Stage 2 dataset build
    |
    v
NeMo Data Store repo + NeMo Entity Store dataset
    |
    v
MLflow parent run
    |
    +-- child run: dataset registration
    +-- child run: NeMo Customizer LoRA job
    +-- child run: NeMo Evaluator job matrix
    +-- optional child run: promotion/deployment
```

MLflow should record:

- the dataset files and checksums produced by Stage 2,
- the NeMo Data Store `hf://datasets/...` URI,
- the NeMo Entity Store `namespace/name` dataset reference,
- the Customizer job ID and output model entity,
- the Evaluator job IDs and exported metrics,
- the adapter artifact location or promotion target.

NeMo should remain authoritative for:

- dataset file storage,
- dataset entity registration,
- Customizer job lifecycle,
- Evaluator target/config/job lifecycle,
- LoRA adapter artifacts.

## First Implementation Target

Start with a single dense LoRA path:

```text
collection: nim_curated
base_model: meta/llama-3.2-1b-instruct
rank: 16
dataset_entity: default/stage3-nim-curated
output_model_entity: default/lora-nim-llama-3.2-1b-r16
```

After that path is reliable, extend to:

- `nemo_usvcs_curated`,
- rank 32 variants,
- MoE shard/TIES workflows,
- registry or webhook-driven promotion.
