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
