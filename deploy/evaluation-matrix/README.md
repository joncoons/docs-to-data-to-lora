# Evaluation Matrix Kubernetes Job

This Job submits the Stage 3 NeMo Evaluator matrix:

- Wave A: single-axis rubric for LoRA targets and dense Llama base targets
- Wave B: LoRA-vs-LoRA pairwise tournament within each corpus
- Wave C: Nemotron-Super-49B comparator vs each LoRA target

It assumes Evaluator targets/configs already exist. Run
`deploy/evaluator-registration/` first.

## Build Image

```bash
docker build \
  -f deploy/evaluation-matrix/Containerfile \
  -t <registry>/docs-to-data-to-lora/evaluation-matrix:latest .

docker push <registry>/docs-to-data-to-lora/evaluation-matrix:latest
```

Update `deploy/evaluation-matrix/job.yaml` with that image.

## Inputs

The Job consumes the same adapter inventory ConfigMap as Evaluator
registration:

```bash
kubectl create configmap stage3-training-session-log \
  -n nemo-peft \
  --from-file=training_session.log=evals/training_session.log
```

The Job writes submitted Evaluator job IDs to a PVC:

```text
/outputs/evaluator_job_ids.json
```

Use this file for result collection and later auditability.

## Service Configuration

The Job uses cluster-local service URLs:

```text
EVALUATOR_URL=http://nemo-evaluator:8000
```

If Evaluator requires bearer auth, reuse the optional Secret from
`deploy/evaluator-registration/`:

```bash
kubectl create secret generic nemo-evaluator-api \
  -n nemo-peft \
  --from-literal=EVALUATOR_API_KEY='<token>'
```

## Polling Modes

By default, the Job submits each wave and waits for terminal status before
moving to the next wave:

```yaml
args:
  - --wave
  - all
  - --poll-interval
  - "30"
  - --max-wait-s
  - "21600"
```

For a controller-style workflow, add `--submit-only`. That mode submits jobs,
writes `evaluator_job_ids.json`, and exits without polling. A separate result
collection Job can then watch those IDs.

## Expected Pipeline Position

```text
Evaluator registration Job
  -> Evaluation matrix Job
  -> Evaluation result collection Job
  -> Promotion/reporting gate
```
