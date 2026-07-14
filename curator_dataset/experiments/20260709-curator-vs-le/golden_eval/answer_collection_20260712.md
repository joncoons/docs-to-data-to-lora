# Answer Collection - Golden v1 Question-Only

Date captured: 2026-07-12

## Scope

This run collected no-RAG, question-only answer sets for the immutable `golden-v1` QA set. The goal is to compare dataset-generation technique behavior, LE vs Curator DiverseQA, without retrieval or context injection skewing the answer source.

## Inputs

- NIM corpus: `golden-v1/nim_curated/test_set.jsonl`, 424 rows.
- NeMo Microservices corpus: `golden-v1/nemo_usvcs_curated/test_set.jsonl`, 431 rows.
- Context-baked companions remain available as diagnostics only: `golden-v1/*/test_set_with_context.jsonl`.

## Serving Method

LoRA adapters were materialized into NIM PEFT source directories under `/data/nim-cache/` and served through local NIMServices with direct ClusterIP calls. `rag-oai-proxy` was not used.

- 1B LoRA: `nim-llm-1b-bw-lora` on `<BLACKWELL_NODE>`, `NIM_PEFT_SOURCE=/model-store/lora-adapters-golden-1b`.
- 8B LoRA: `nim-llm-8b-bw-lora` on `<BLACKWELL_NODE>`, `NIM_PEFT_SOURCE=/model-store/lora-adapters-golden-8b`.
- 3B LoRA: `nim-llm-3b-ada-lora` on `<ADA_NODE>`, `NIM_PEFT_SOURCE=/model-store/lora-adapters-golden-3b`.
- 70B baseline: `nim-llm` on `<BLACKWELL_NODE>`, no adapter, `meta/llama-3.3-70b-instruct`, NIM 2.0.5, local `nvfp4` cached profile selected by the image.

The first 70B attempt used the local snapshot ref `nvfp4-klvd4-nzbq-tool-calling` as `NIM_MODEL_PROFILE`, which NIM rejected because it is not a manifest profile id. Removing the explicit profile override allowed NIM to resolve profile `13b2cabe93f6e3d81e056b10d747d4baffc8dc6708977963dbc02c378fe6e7fd` against the mounted cache.

## Collection Parameters

- Collector: `scripts/eval/collect_completions.py`.
- Output root: `<EVAL_ROOT>/completions-question-only`.
- LoRA run id: `golden-v1-qonly-lora-20260712`.
- 70B run id: `golden-v1-qonly-70b-20260712`.
- Temperature: `0.0001`.
- Max tokens: `1024`.
- LoRA concurrency: `2`.
- 70B concurrency: `1`.
- Resume and retry were enabled with `--resume --retry-failed --max-attempts 3 --retry-backoff-s 5`.

## Completion Results

| Corpus | Model family | Method | Rank | Rows completed | Failed rows |
| --- | --- | --- | --- | ---: | ---: |
| NIM | Llama 3.2 1B | Curator DiverseQA | r16, r32 | 424 each | 0 |
| NIM | Llama 3.2 1B | LE | r16, r32 | 424 each | 0 |
| NIM | Llama 3.2 3B | Curator DiverseQA | r16, r32 | 424 each | 0 |
| NIM | Llama 3.2 3B | LE | r16, r32 | 424 each | 0 |
| NIM | Llama 3.1 8B | Curator DiverseQA | r16, r32 | 424 each | 0 |
| NIM | Llama 3.1 8B | LE | r16, r32 | 424 each | 0 |
| NIM | Llama 3.3 70B | Base reference | none | 424 | 0 |
| NeMo Microservices | Llama 3.2 1B | Curator DiverseQA | r16, r32 | 431 each | 0 |
| NeMo Microservices | Llama 3.2 1B | LE | r16, r32 | 431 each | 0 |
| NeMo Microservices | Llama 3.2 3B | Curator DiverseQA | r16, r32 | 431 each | 0 |
| NeMo Microservices | Llama 3.2 3B | LE | r16, r32 | 431 each | 0 |
| NeMo Microservices | Llama 3.1 8B | Curator DiverseQA | r16, r32 | 431 each | 0 |
| NeMo Microservices | Llama 3.1 8B | LE | r16, r32 | 431 each | 0 |
| NeMo Microservices | Llama 3.3 70B | Base reference | none | 431 | 0 |

## Provenance

- Adapter materialization manifest: `lora_answer_collection_materialization_20260712.json`.
- Per-model outputs include `responses.jsonl`, `errors.jsonl`, `request_manifest.json`, and `token_summary.json` under the output root above.
- The local NIMServices were scaled down after LoRA collection; the 70B baseline NIM remained running at capture time.
