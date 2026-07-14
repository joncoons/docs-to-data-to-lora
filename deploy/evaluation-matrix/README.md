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
EVALUATOR_URL=http://nemo-evaluator:7331
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


## Durable Completion Collection

Before sending rows to an LLM judge, collect target model completions as durable artifacts. The collector runs against the OpenAI-compatible RAG proxy, strips `<think>...</think>` from saved scoring text, and still records raw responses plus token usage for ROI analysis.

Run the in-cluster completion job with:

```bash
deploy/evaluation-matrix/run-collect-completions.sh
```

The helper publishes `scripts/eval/collect_completions.py` as a ConfigMap and creates a generated-name Job in `runai-rag`. The Job runs both dense test sets with an explicit generation budget:

```text
--max-tokens 8192
```

The current 1B dense matrix is:

```text
nim_curated:         llama-3.2-1b base, lora-nim-r16, lora-nim-r32
nemo_usvcs_curated:  llama-3.2-1b base, lora-nemo-usvcs-r16, lora-nemo-usvcs-r32
```

A matching 3B matrix can run concurrently after `nim-llm-3b-ada-lora` is Ready:

```bash
RUN_ID=<shared-run-id> deploy/evaluation-matrix/run-collect-completions-3b.sh
```

```text
nim_curated:         llama-3.2-3b base, lora-nim-r16, lora-nim-r32
nemo_usvcs_curated:  llama-3.2-3b base, lora-nemo-usvcs-r16, lora-nemo-usvcs-r32
```

Outputs are written to:

```text
<EVAL_ROOT>/completions/<dataset_slug>/<base_slug>/<target_slug>/<rank_slug>/<run_id>/
```

Each run directory contains `responses.jsonl`, `errors.jsonl`, `manifest.json`, and `token_summary.json`. The `rank_slug` component is `base`, `r16`, or `r32`, so dense adapter outputs do not collide.

## Kimi Judge Secret

Create the judge API key Secret before running the live RAGAS smoke Job:

```bash
kubectl create secret generic kimi-judge-api \
  -n nemo-peft \
  --from-literal=KIMI_KEY="${KIMI_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

Do not commit the key into a manifest. The smoke Job mounts this Secret as the
`KIMI_KEY` environment variable and the smoke script redacts it from saved JSON
artifacts.

## Live RAGAS Smoke And Token ROI

RAGAS metrics are only supported by Evaluator's live endpoint. The live endpoint
accepts `rows` or `dataset` targets, not model targets, so the smoke path first
precomputes target responses and then submits those rows to `/v1/evaluation/live`.

From a workstation that can reach the cluster services, run:

```bash
python scripts/eval/run_live_ragas_smoke.py \
  --evaluator-url http://<evaluator-cluster-ip>:7331 \
  --target-api-url http://<rag-oai-proxy-cluster-ip>:8080 \
  --judge-api-url https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/kimi-k2-6/v1 \
  --judge-model-id kimi-k2-6 \
  --judge-api-key-env KIMI_KEY \
  --limit 1 \
  --judge-max-retries 3 \
  --metrics faithfulness \
  --out evals/live_ragas_smoke.json
```

Or run the Kubernetes Job after building/pushing the evaluation image and
creating `kimi-judge-api`:

```bash
kubectl apply -f deploy/evaluation-matrix/live-ragas-smoke-job.yaml
```

The script uses Kimi K2 as the default LLM judge via
`https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/kimi-k2-6/v1` and reads
its API key from `$KIMI_KEY`. The key is sent to Evaluator for the live request
but redacted from the JSON artifact.

The script uses `max_tokens=8192` by default for target and judge calls so
reasoning models have room for both `<think>` output and the final answer. It
also strips `<think>...</think>` defensively before RAGAS scoring.

`evals/live_ragas_smoke.json` includes `token_roi_summary` with:

- target prompt, raw completion, estimated cleaned completion, and estimated
  stripped reasoning tokens
- judge token usage when Evaluator/RAGAS exposes it in live logs
- combined raw token total for cost and throughput comparisons

Use the target-generation section, not the judge-scoring section, when comparing
mid-sized reasoning models against SLM+PEFT or MoE+PEFT inference cost. Judge
cost is still captured, but it is an evaluation overhead rather than target model
serving cost.

## Evaluation Token Accounting

Direct Kimi single-axis and pairwise jobs write token usage into each
`summary.json` artifact:

- `target_generation`: prompt/completion/total tokens from the saved model
  completions being evaluated
- `judge_scoring`: prompt/completion/total tokens consumed by the Kimi judge
- `combined_total_tokens_raw`: target plus judge tokens for full experiment
  accounting

Roll these per-run summaries into experiment-level JSON and CSV artifacts:

```bash
python scripts/eval/summarize_token_usage.py \
  --eval-root <EVAL_ROOT> \
  --run-id 20260530T195331Z \
  --out <EVAL_ROOT>/token-usage/20260530T195331Z/token_usage_summary.json \
  --csv-out <EVAL_ROOT>/token-usage/20260530T195331Z/token_usage_summary.csv
```

Use `judge_scoring.total_tokens_raw` for Kimi/API evaluation spend. Use
`target_generation.total_tokens_raw` for model-serving ROI. Keep those separated
in reports so judge overhead does not get attributed to the model being tested.


## 8B completion collection

The 8B collector runs both ground-truth held-out test sets at parity with 1B/3B:

```text
nim_curated:         llama-3.1-8b base, lora-nim-r16, lora-nim-r32
nemo_usvcs_curated:  llama-3.1-8b base, lora-nemo-usvcs-r16, lora-nemo-usvcs-r32
```

The 8B NIMService should expose four LoRA adapters with `--max-loras 4`. For the Blackwell rerun, use two replicas and `--concurrency 8` in the collector.

```bash
RUN_ID=20260529T142723Z deploy/evaluation-matrix/run-collect-completions-8b.sh
```

## Pairwise Release Gate

Stage 1/2 pairwise jobs are queued as suspended Jobs so they do not add Kimi
judge pressure while single-axis scoring is still running. To keep total pairwise
pressure to two Jobs at a time, use a two-gate release sequence. First, release
Stage 1 after the dense 1B, 3B, and 8B single-axis Jobs succeed:

```bash
kubectl apply -f deploy/evaluation-matrix/direct-kimi-pairwise-release-gate-job.yaml
```

The release gate watches these dependencies:

```text
direct-kimi-singleaxis-dense-1b-20260530t195331z
direct-kimi-singleaxis-dense-3b-20260530t195331z
direct-kimi-singleaxis-dense-8b-20260530t195331z
```

When all three have `status.succeeded > 0`, it patches the four Stage 1/2
pairwise Jobs to `spec.suspend=false`. If any dependency fails, it exits
non-zero and leaves pairwise suspended.

Then release Stage 2 after the two Stage 1 pairwise Jobs succeed:

```bash
kubectl apply -f deploy/evaluation-matrix/direct-kimi-pairwise-stage2-release-gate-job.yaml
```

The Stage 2 release gate watches:

```text
direct-kimi-pairwise-stage1-nim-20260530t195331z
direct-kimi-pairwise-stage1-nemo-20260530t195331z
```

It then patches only these Jobs to `spec.suspend=false`:

```text
direct-kimi-pairwise-stage2-nim-20260530t195331z
direct-kimi-pairwise-stage2-nemo-20260530t195331z
```

