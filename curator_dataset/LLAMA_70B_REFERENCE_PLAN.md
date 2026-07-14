# Parked plan: dense 70B comparison target

Date captured: 2026-07-09

Use a hosted dense Llama 3.3 70B Instruct endpoint as the comparison target for
smaller dense Llama LoRA adapters, not as the judge.

Intended comparison:

- Baseline/control targets: smaller dense Llama base models and LE-trained LoRA
  adapters.
- Treatment targets: smaller dense Llama LoRA adapters trained on the
  Curator-generated DiverseQA dataset.
- Reference target: Llama 3.3 70B Instruct via `inference.nvidia.com`.
- Judge: keep the existing independent judge path; do not use 70B as the judge.

Credential assumption:

- Read the NVIDIA hosted inference API key from a Kubernetes Secret.
- Do not write API key values into manifests, logs, repo files, or generated
  reports.

Implementation note:

- Add a hosted-NVIDIA completion collection mode that writes responses into the
  existing `evals/completions/.../responses.jsonl` layout so the current
  single-axis and pairwise evaluators can consume the 70B reference responses.
- The concrete Python client example is pending and should be attached here or
  referenced from the implementation commit when resumed.

## 2026-07-12 golden evaluation update

The dense 70B reference is now part of the `golden-v1` evaluation plan under `curator_dataset/experiments/20260709-curator-vs-le/golden_eval/`.

- Golden dataset: context-baked, HTML-only, exact prompt-overlap filtered against current LE and Curator train/validation files.
- Reference target: `nvidia/meta/llama-3.3-70b-instruct` on `https://inference-api.nvidia.com/v1`.
- Primary judge: `azure/moonshotai/kimi-k2.6` on `https://inference-api.nvidia.com/v1`; verified available on 2026-07-12 using the NVIDIA inference Kubernetes Secret.
- Fallback judge: Claude Sonnet 4.6 through NeMo Evaluator if Kimi is unavailable or unstable.
- Evaluation order: single-axis for standalone efficacy, pairwise LE vs Curator by matched corpus/base/rank, then pairwise best LoRA winner vs Llama 3.3 70B.

The collector now supports a target API key via `--target-api-key-env` or `--target-api-key` so hosted 70B completions can be written to the same durable `responses.jsonl` layout as local LoRA completions.
