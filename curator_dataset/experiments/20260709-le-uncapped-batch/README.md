# LE Uncapped Batched Rerun - 2026-07-09

Purpose: rerun logical-entailment dataset generation after removing the inherited Stage 1A entailment/premise caps and switching to the batched KVP path.

## Stage 0 Extraction

| Corpus | ES index | Passages | Output |
| --- | --- | ---: | --- |
| NIM | `nim_curated` | 498 | `stage0/nim_curated/passages.jsonl` |
| NeMo Microservices | `nemo_usvcs_curated` | 486 | `stage0/nemo_usvcs_curated/passages.jsonl` |

Stage 0 used `PIPELINE_ES_HOST=https://10.43.233.46:9200` and the in-cluster `rag-eck-elasticsearch-es-elastic-user` secret.

## Stage 1A Rerun

Run both corpora with Nemotron 3 Super for logical-entailment extraction and
KVP expansion. Treat `nvidia/nvidia/nemotron-3-super-v3` and
`nvidia/nemotron-3-super-120b-a12b` as the same Super 120B model for this work.

Ultra 550B is parked for Stage 1A. It remains the preferred frontier-level
model for downstream synthesis and instruction-diversity stages such as 1B and
1C, where cross-passage synthesis and controlled rewriting matter more than raw
LE extraction throughput.

Required generation settings:

- Stage 1A mode: batched KVP expansion
- Temperature: `0.95`
- LE max tokens: `16384`
- Batched KVP max tokens: `16384`
- No entailment count cap
- No premise count cap

After extraction, continue through the remaining grounded LE dataset pipeline.
For this rerun, Stage 1.5 synthetic gap-fill is intentionally skipped; proceed
from Stage 1C to Stage 2 unless a later experiment explicitly opts into
synthetic augmentation.

## Optional Stage 1.5 Synthetic Augmentation

Stage 1.5 is optional. It is most useful when the source-grounded dataset is
small, product coverage is materially imbalanced, or an experiment explicitly
tests synthetic-data lift. When enabled, Data Designer or any equivalent
OpenAI-compatible generation path should use a frontier-grade model so synthetic
rows preserve source constraints and add useful task diversity. Smaller models
are acceptable for smoke-testing the handoff mechanics, not for production
synthetic generation.

The current NIM/NeMo LE rerun skips Stage 1.5.

## Downstream Runner Defaults

`run_le_downstream.py` is intended to stay runnable with either one inference endpoint or several.
Use repeated `--target` arguments to add targets:

```bash
# Single frontier endpoint
python run_le_downstream.py   --collection nim_curated   --output-dir runs/nim_curated/super-v3   --stage 1b   --resume   --target https://inference-api.nvidia.com/v1=nvidia/nvidia/nemotron-3-ultra

# Two OpenAI-compatible frontier endpoints, load-balanced
python run_le_downstream.py   --collection nim_curated   --output-dir runs/nim_curated/super-v3   --stage 1b   --resume   --target https://inference-api.nvidia.com/v1=nvidia/nvidia/nemotron-3-ultra   --target https://frontier.example.com/v1=vendor/frontier-model@32768   --api-key "$FRONTIER_API_KEY"
```

Target format is `ENDPOINT=MODEL` or `ENDPOINT=MODEL@MAX_CONTEXT_TOKENS`.
The endpoint must implement OpenAI-compatible chat completions. Hosted NVIDIA
endpoints use the configured external API key; local OpenAI-compatible endpoints
use a placeholder `local` key unless `--api-key` is supplied. The code does not
require Nemotron specifically: any model ID accepted by the target endpoint can
be used, and repeated `--target` values are load-balanced. For this experiment,
Stage 1B and Stage 1C should use a frontier-level model such as Nemotron 3
Ultra. Stage 1A extraction should stay on Super unless an experiment explicitly
overrides it.

Stage 2 has separate target knobs so QA admission can stay cost-conscious even
when Stage 1B/1C synthesis uses a frontier target. Use `--stage2-target` and
`--stage2-canonical-model` to override the default hosted Super target or to
load-balance Stage 2 across several compatible endpoints.

Stage 2 QA/admission should default to a Super 120B-class model such as
Nemotron 3 Super, while retaining an explicit option to escalate to Nemotron 3
Ultra 550B or another foundation/frontier-grade model for critical audits or
small high-value datasets. The execution surface should migrate to
Curator-backed LLM quality filtering/refinement where available.
Operationally, Stage 2 is the pre-Curator semantic gate: it repairs or rejects
source-grounded QA rows before Stage 3 spends work on deduplication, heuristic
quality filters, and train/validation splitting. This keeps raw LLM-generated
rows from polluting the Curator input contract and preserves a local
admission/rejection reason in `stage2_dropped.jsonl` plus
`provenance/stage2_quality.jsonl`. The direct QA runner remains useful for
smoke tests, fallback execution, and ablation; Stage 4 should remain
independent, for example Claude Sonnet 4.6 via the NVIDIA-hosted
OpenAI-compatible endpoint.

Stage 3 length filtering uses the production tokenizer, not a generic proxy. For
this Llama 3.1 8B Customizer experiment, the default resolves from the NIM cache
at `$LOCAL_NIM_CACHE/ngc/hub/models--nim--meta--llama-3.1-8b-instruct/snapshots/fp8-tool-calling`;
this run used `/mnt/nvme4/nim_cache/nim/ngc/hub/models--nim--meta--llama-3.1-8b-instruct/snapshots/fp8-tool-calling`
with
`--stage3-min-question-tokens 12 --stage3-min-answer-tokens 8`. The initial
`cl100k_base`/`question >= 8`/`answer >= 25` filter over-dropped concise
technical answers; the revised run is captured in `stage3_curator_reductions.md`.

Stage 4 validates the finalized Stage 3 `training.jsonl`, not the full pre-Curator
Stage 2 set. The runner restores source context by joining each curated
prompt/completion back to `stage2_eval.jsonl`, then judges a stratified sample
with Claude Sonnet 4.6 via `https://inference-api.nvidia.com/v1` using
`azure/anthropic/claude-sonnet-4-6`. Outputs are `validation_report.json`,
`validation_sample.jsonl`, and `validation_judgments.jsonl`.
The completed 100-row-per-corpus validation passed for both corpora; see
`stage4_validation_summary.md` for the comparison and sampled failure notes.

The runner defaults to all source document kinds for turnkey use. Pass `--source-doc-kind html` only when an experiment intentionally excludes parsed PDFs. When a source filter is active, the runner writes selected KVP and lineage sidecars so Data Store publication can use only the selected provenance.

If `--allow-incomplete-stage1a` is supplied, selected passages whose latest Stage 1A status is incomplete are excluded from downstream stages and recorded in `provenance/source_filter_excluded_passages.jsonl` with a `stage1a_*` reason. This keeps downstream augmentation grounded in completed Stage 1A rows without deleting the raw extraction artifacts.

Stage 1B is durable: `stage1b_synthesis.jsonl` is appended as each passage finishes, and `stage1b_passage_results.jsonl` records per-passage status. Resume runs skip passages that already have persisted rows or terminal no-work statuses.

Stage 1C is durable the same way: `stage1c_instruction.jsonl` is appended as passages finish, and `stage1c_passage_results.jsonl` records per-passage status. The default selection mode is `stratified`; pass `--stage1c-selection-mode top_density` for legacy high-density-only behavior or `--stage1c-selection-mode all` for full instruction augmentation coverage.

Stage 1B retrieves kNN neighbors from the target corpus Elasticsearch index selected by `--collection`; for example, `nim_curated` neighbors come from `nim_curated`, not from the NeMo Microservices corpus.

## Stage 5 Customizer Training - 2026-07-11

The finalized LE datasets were registered in Entity/Data Store and submitted to
Customizer for 5-epoch LoRA SFT on `meta/llama-3.1-8b-instruct` using the
`meta/llama-3.1-8b-instruct@v1.0.0+80GB` template. The selected template is
single-GPU LoRA SFT: `num_gpus=1`, `tensor_parallel_size=1`, and
`data_parallel_size=1`.

Before submission, non-critical GPU serving was scaled down and stale pods were
cleaned:

- `runai-rag/nim-devstral-small-fp8` scaled to `0` replicas.
- Stale terminating pod `nrl265-durable-isolated/durable-isolated-durable-ingest-caption-bf47b5688-lgcrs` force deleted.
- Customizer `training.nodeSelectors` and `training.container_defaults.nodeSelector` patched to `ubuntu-local-dev`; the pre-patch cluster backup is local-only under `.local_archive/`.

Submitted jobs:

| Corpus | Dataset entity | Rank | Job ID | Output model entity | Initial runtime state |
| --- | --- | ---: | --- | --- | --- |
| NIM | `default/stage3-nim-curated-le-super-v3` | 16 | `cust-VM3mbWVx7FcPdtTG84UiJs` | `default/lora-nim-le-super-v3-e5-llama-3.1-8b-r16-20260711` | running on `ubuntu-local-dev` |
| NeMo Microservices | `default/stage3-nemo-usvcs-curated-le-super-v3` | 16 | `cust-5HqsCLjwYzyy2AyiW3EiNW` | `default/lora-nemo-usvcs-le-super-v3-e5-llama-3.1-8b-r16-20260711` | running on `ubuntu-local-dev` |
| NIM | `default/stage3-nim-curated-le-super-v3` | 32 | `cust-BBbUtEYXHNW2zjzGZhqMpY` | `default/lora-nim-le-super-v3-e5-llama-3.1-8b-r32-20260711` | pending for GPU capacity |
| NeMo Microservices | `default/stage3-nemo-usvcs-curated-le-super-v3` | 32 | `cust-8GE3t81yPq76FhK3b4b7Ts` | `default/lora-nemo-usvcs-le-super-v3-e5-llama-3.1-8b-r32-20260711` | pending for GPU capacity |

The submission artifact with full request payloads, observed statuses, and pod
placement is `stage5_customizer_training_5epoch_20260711.json`.

## Stage 5 Follow-on 1B/3B Plan - 2026-07-11

Follow-on training is captured in `stage5_followon_1b_3b_plan_20260711.md`.
The 1B jobs should run on Blackwell after all 8B jobs complete. The 3B jobs
should run on `ubuntu2` with the Ada-safe `meta/llama-3.2-3b-instruct@v1.0.0+40GB`
template; the `+80GB` 3B template is DP5 in the live Customizer config and is
not the desired single-GPU TP1/DP1 path.

Ada GPU cleanup has been completed: the NeMo Retriever GPU deployments on
`ubuntu2` are scaled to zero, `ubuntu2` has no GPU-requesting pods, and allocated
`nvidia.com/gpu` is zero. Before submitting 3B, decide whether to temporarily
remove/reduce `ubuntu2` time-slicing so the scheduler exposes 2 physical GPU
slots instead of 10 logical shared slots.
