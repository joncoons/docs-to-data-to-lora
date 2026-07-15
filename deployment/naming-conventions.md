# Deployment Naming Runbook: Kubernetes + Run.ai

This runbook gives non-local naming conventions for adapting the repo to a
Kubernetes + Run.ai environment. The names are examples, not required values.
Replace `d2dl`, `dev`, and `domain-a` with your organization, environment, and
domain identifiers.

## Naming Inputs

Use short, stable inputs and derive every Kubernetes, Run.ai, NeMo, and artifact
name from them.

| Input | Example | Purpose |
|---|---|---|
| Project prefix | `d2dl` | Short name for docs-to-data-to-lora workloads |
| Environment | `dev`, `stage`, `prod` | Separates lifecycle environments |
| Domain ID | `domain-a` | Stable corpus or business-domain identifier |
| Model size | `1b`, `3b`, `8b`, `70b` | Serving and adapter target |
| LoRA rank | `r16`, `r32` | Adapter variant |

Avoid local machine names, user names, hostnames, personal paths, and one-off
experiment timestamps in public manifests. Use labels and annotations for run
IDs when you need traceability.

## Namespaces

Use separate namespaces for pipeline jobs and serving workloads when your
cluster policy allows it.

| Purpose | Pattern | Example |
|---|---|---|
| Pipeline jobs | `<project>-<env>-pipeline` | `d2dl-dev-pipeline` |
| Model serving | `<project>-<env>-serving` | `d2dl-dev-serving` |
| Observability | `<project>-<env>-observability` | `d2dl-dev-observability` |

```bash
PROJECT=d2dl
ENV=dev
PIPELINE_NS="${PROJECT}-${ENV}-pipeline"
SERVING_NS="${PROJECT}-${ENV}-serving"
OBS_NS="${PROJECT}-${ENV}-observability"

kubectl create namespace "$PIPELINE_NS"
kubectl create namespace "$SERVING_NS"
kubectl create namespace "$OBS_NS"
```

## Run.ai Projects

Keep the Run.ai project name aligned with the Kubernetes environment. If your
cluster maps Run.ai projects to namespaces, use the same stem.

| Purpose | Pattern | Example |
|---|---|---|
| Run.ai project | `<project>-<env>` | `d2dl-dev` |
| GPU workload class | `<project>-<env>-<role>` | `d2dl-dev-serving` |

Example pod annotation:

```yaml
metadata:
  annotations:
    run.ai/gpu-memory: "42000M"
  labels:
    app.kubernetes.io/part-of: docs-to-data-to-lora
    app.kubernetes.io/component: model-serving
    docs-to-data-to-lora.io/domain: domain-a
```

## Core Services

Use service names that describe the role, not the node, GPU, or local deployment
history.

| Role | Pattern | Example |
|---|---|---|
| Embedding NIM | `<project>-embed` | `d2dl-embed` |
| Reranker NIM | `<project>-rerank` | `d2dl-rerank` |
| LoRA-capable LLM NIM | `<project>-llm-<size>-lora` | `d2dl-llm-8b-lora` |
| Dense reference LLM NIM | `<project>-llm-<size>-base` | `d2dl-llm-70b-base` |
| RAG API or proxy | `<project>-rag-api` | `d2dl-rag-api` |
| Web crawler | `<project>-crawler` | `d2dl-crawler` |
| Elasticsearch | `<project>-es` | `d2dl-es` |

Cluster-local endpoint examples:

```bash
PIPELINE_ES_HOST="https://d2dl-es.d2dl-dev-serving.svc.cluster.local:9200"
PIPELINE_NIM_ENDPOINTS="http://d2dl-llm-8b-lora.d2dl-dev-serving.svc.cluster.local:8000/v1"
```

## Secrets

Use secret names based on the credential purpose. Do not encode usernames,
ticket IDs, or local file paths into secret names.

| Secret | Keys | Example |
|---|---|---|
| Hugging Face access | `HF_TOKEN` | `d2dl-hf-token` |
| LLM endpoint key | `LLM_API_KEY` | `d2dl-llm-api-key` |
| Elasticsearch credentials | `username`, `password`, `ca.crt` | `d2dl-es-credentials` |
| MLflow tracking | `MLFLOW_TRACKING_URI`, optional auth keys | `d2dl-mlflow` |
| NeMo service credentials | service-specific keys | `d2dl-nemo-services` |

```bash
kubectl create secret generic d2dl-hf-token \
  -n "$PIPELINE_NS" \
  --from-literal=HF_TOKEN='<redacted>'
```

## PVCs And Artifact Roots

Use PVCs for durable artifacts that are consumed by later jobs. Mount them under
generic in-container paths.

| PVC | Mount | Contents |
|---|---|---|
| `<project>-corpus-artifacts` | `/corpus` | Crawl inventory, source chunks, manifests |
| `<project>-dataset-artifacts` | `/datasets` | Stage 2 JSONL and provenance outputs |
| `<project>-adapter-artifacts` | `/adapters` | LoRA adapter checkpoints and manifests |
| `<project>-eval-artifacts` | `/evals` | Completion captures and evaluator summaries |
| `<project>-tokenizer-cache` | `/tokenizers` | Optional tokenizer/config artifacts |

For local tutorial runs, `outputs/` is ignored by git. For Kubernetes, prefer a
PVC and set:

```bash
PIPELINE_STAGE3_TOKENIZER=/tokenizers/llama-3.1-8b-instruct
```

If the tokenizer is not already available, download only tokenizer/config
artifacts before Stage 3:

```bash
python scripts/pipeline/download_stage3_tokenizer.py \
  --output-dir /tokenizers/llama-3.1-8b-instruct
```

The default model repo is
[`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
That repository is gated; users must have accepted the model terms and supplied
an authorized Hugging Face token.

## NeMo Entity And Dataset Names

Keep NeMo names stable and aligned with the corpus and stage.

| Object | Pattern | Example |
|---|---|---|
| Training dataset | `stage3-<domain>-<method>` | `stage3-domain-a-le` |
| Golden test dataset | `golden-<domain>-v<version>` | `golden-domain-a-v1` |
| Adapter entity | `lora-<domain>-<base>-<rank>` | `lora-domain-a-llama-3.1-8b-r32` |
| Evaluation run | `eval-<domain>-<base>-<mode>` | `eval-domain-a-8b-norag` |

Use NeMo workspace or namespace fields for ownership and access control; do not
duplicate that information in every object name unless your platform requires it.

## Minimal Job Environment

Every K8s job should be able to run without local filesystem assumptions:

```yaml
env:
  - name: PIPELINE_ES_HOST
    value: "https://d2dl-es.d2dl-dev-serving.svc.cluster.local:9200"
  - name: PIPELINE_NIM_ENDPOINTS
    value: "http://d2dl-llm-8b-lora.d2dl-dev-serving.svc.cluster.local:8000/v1"
  - name: PIPELINE_STAGE3_TOKENIZER
    value: "/tokenizers/llama-3.1-8b-instruct"
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: d2dl-hf-token
        key: HF_TOKEN
```

Mount artifacts by purpose:

```yaml
volumeMounts:
  - name: dataset-artifacts
    mountPath: /datasets
  - name: tokenizer-cache
    mountPath: /tokenizers
volumes:
  - name: dataset-artifacts
    persistentVolumeClaim:
      claimName: d2dl-dataset-artifacts
  - name: tokenizer-cache
    persistentVolumeClaim:
      claimName: d2dl-tokenizer-cache
```

## Promotion Checklist

Before publishing or promoting a deployment manifest:

- Replace local namespaces such as personal project names with the environment
  naming convention.
- Replace hardcoded service DNS with names derived from the service table.
- Ensure secrets are referenced by Kubernetes Secret names only.
- Ensure tokenizer paths are PVC paths or explicitly documented local tutorial
  paths under `outputs/`.
- Ensure PVCs are named by artifact purpose, not by node, disk, or user.
- Ensure Run.ai GPU memory annotations are workload-specific and documented.
