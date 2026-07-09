# NIM Curator DiverseQA dataset result

Completed: 2026-06-29

## Authorization and source audit

The user explicitly authorized transmitting the Curator-ready corpus to
NVIDIA's hosted inference API after an external-data warning. The local audit
found 451 public-source URL documents:

- 439 `docs.nvidia.com` URLs
- 12 `raw.githubusercontent.com` URLs
- no private or internal hosts

The API key was read at runtime from Kubernetes Secret
`runai-rag/nvidia-inference-key` key `api-key`; its value was never persisted.

## Corpus preparation

- Raw Elasticsearch hits: 2,086
- HTML chunks: 2,018
- Non-HTML chunks excluded: 68
- Acknowledgement chunks excluded: 340
- Exact duplicate chunks removed within URL: 586
- Curator-ready URL documents: 451
- Input SHA-256:
  `abf36f4c74ba2364694d911b68bd84ab799c49bd9054ff835c8eb41c13d7aa45`

## Curator generation

- Container: `nvcr.io/nvidia/nemo-curator:26.04`
- Container digest:
  `sha256:11635749967ea52fb82ccc2b82d326a4de3dfc50a51605c95a993a007e8219ac`
- Curator version: `1.2.0+f07fa0e1`
- Stage: native Nemotron-CC `DiverseQAStage`
- Native preprocessing: `add_preprocessing_pipeline`
- Hosted model: `nvidia/nvidia/nemotron-3-super-v3`
- Model equivalence: actual run manifests record `nvidia/nvidia/nemotron-3-super-v3`; the catalog shorthand `nvidia/nemotron-3-super-v3` is treated as exact-equivalent to `nvidia/nemotron-3-super-120b-a12b` for project provenance.
- Endpoint: `https://inference-api.nvidia.com/v1`
- Temperature: 0.5
- Top-p: 0.9
- Seed: 42
- Thinking: disabled
- Maximum input tokens per segment: 1,000
- Maximum output tokens: 600
- Successful run time: 1,833.275 seconds
- Raw segments: 1,286
- Raw response SHA-256:
  `d7b242f8aac2c5874975e6becab3a3d0ac6d5e8f5a94a00625f4a899f2f39ba4`

The initial full attempt used 8 GiB shared memory and failed when a Ray writer
actor segfaulted. The successful retry used 64 GiB shared memory. A temporary
burst of HTTP 429 responses was handled by Curator's native exponential retry
logic and did not cause output loss.

## Normalization and finalization

- Parsed segments: 1,286 / 1,286
- Parse failures: 0
- Parsed QA pairs after exact `(question, answer)` deduplication: 9,962
- Exact duplicate pairs removed: 310
- Additional case-insensitive exact question duplicates removed: 205
- Rows removed by 32-pair-per-document cap: 2,097
- Final QA samples: 7,660 across 442 documents
- Training: 6,845 rows across 398 documents
- Validation: 815 rows across 44 documents
- Document overlap between splits: 0
- Remaining exact duplicate questions: 0
- Remaining acknowledgement rows: 0
- Empty prompts, completions, or contexts: 0

Six input documents containing only 3–17 words were removed by Curator's native
30-token minimum. Three 22-word EULA pages reached generation but produced only
duplicate boilerplate questions; none survived final question deduplication.

## Primary artifacts

```text
curator_dataset/data/nim_curated/url_documents.curator_ready.jsonl
curator_dataset/data/nim_curated/url_documents.curator_ready.manifest.json
curator_dataset/data/nim_curated/qa/runs/full-20260629t2032z-retry1/
  run_manifest.json
  raw/f5908f6cdabb.jsonl
  dataset/qa_pairs.jsonl
  dataset/manifest.json
  dataset/final/qa_samples.jsonl
  dataset/final/training.jsonl
  dataset/final/validation.jsonl
  dataset/final/training_samples.jsonl
  dataset/final/validation_samples.jsonl
  dataset/final/manifest.json
```

Final Customizer-format hashes:

- Training:
  `095dec04be3c75ba3877535a26b6d8c0aa2586a37dad93fcf8dba90bae0f45b0`
- Validation:
  `aeae824049b9e52a34ad047be9d3c791b207e8440aa5f7fb192539030879ee67`

No existing pipeline code or canonical dataset was modified.
