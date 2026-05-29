# Evaluator Registration Kubernetes Job

This Job registers Stage 3 NeMo Evaluator model targets and evaluation configs.
It is an idempotent control-plane step: reruns tolerate existing targets/configs
when Evaluator returns HTTP 409.

The registered model targets point at native NeMo NIM Proxy using Evaluator's
`format: nim` target shape. This is the preferred showcase path over the legacy
`deploy/rag-oai-proxy` workaround.

## Build Image

```bash
docker build \
  -f deploy/evaluator-registration/Containerfile \
  -t <registry>/docs-to-data-to-lora/evaluator-registration:latest .

docker push <registry>/docs-to-data-to-lora/evaluator-registration:latest
```

Update `deploy/evaluator-registration/job.yaml` with that image.

## Training Session Log Input

The registration script derives adapter targets from `evals/training_session.log`.
For Kubernetes, create a ConfigMap from the current inventory:

```bash
kubectl create configmap stage3-training-session-log \
  -n nemo-peft \
  --from-file=training_session.log=evals/training_session.log
```

If the adapter inventory changes, recreate or patch this ConfigMap before
rerunning the Job.

## Service Configuration

The Job uses environment variables so service names are cluster-local and do
not require hard-coded NodePorts:

```text
EVALUATOR_URL=http://nemo-evaluator:8000
NIM_PROXY_URL=http://nemo-nim-proxy:8000
```

Adjust `EVALUATOR_URL` to match the Service and port exposed by your NeMo
Evaluator deployment. `NIM_PROXY_URL` must resolve from the Evaluator service
runtime, because Evaluator invokes the registered model endpoint.

If Evaluator requires bearer auth, create an optional Secret:

```bash
kubectl create secret generic nemo-evaluator-api \
  -n nemo-peft \
  --from-literal=EVALUATOR_API_KEY='<token>'
```

## Registered Entities

The template runs:

```bash
python scripts/eval/register_evaluator_entities.py \
  --all \
  --log-path /inputs/training_session.log
```

`--all` registers:

- LoRA adapter model targets from the training-session log
- Dense Llama base reference targets
- Nemotron-Super-49B comparator target
- Stage 3 single-axis and pairwise Evaluator configs

Datasets are not registered by this Job. Stage 3 datasets belong in NeMo Entity
Store/Data Store and should be handled by the dataset registration Job.
