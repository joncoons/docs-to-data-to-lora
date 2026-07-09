# NeMo Microservices Curator results

Run date: 2026-06-29

This run is the NeMo Microservices counterpart to the NIM Curator experiment. It used only HTML-like Elasticsearch passages, grouped and concatenated by canonical URL. Acknowledgment and EULA pages were excluded, and exact duplicate passages within each URL were removed before any text was sent for generation.

## Extraction

- Elasticsearch index: `nemo_usvcs_curated`
- Raw Elasticsearch hits: 1,289
- Raw URL documents: 600
- Exact duplicate passage instances removed: 24
- Acknowledgment documents excluded: 1
- EULA documents excluded: 1
- Final Curator input documents: 598
- Source hosts: 597 `docs.nvidia.com` documents and 1 `raw.githubusercontent.com` document after the final exclusions
- Curator input SHA-256: `18ea68c8ab28fcb6395093bd13ebeaf2acf98bd40d81e82fcb233fe32f79f77e`
- Input: `data/nemo_usvcs_curated/url_documents.curator_input.jsonl`

## Generation

- Container: `nvcr.io/nvidia/nemo-curator:26.04`
- Container digest: `sha256:11635749967ea52fb82ccc2b82d326a4de3dfc50a51605c95a993a007e8219ac`
- Curator version: `1.2.0+f07fa0e1`
- Native pipeline: Nemotron-CC `DiverseQAStage`
- Model: `nvidia/nvidia/nemotron-3-super-v3`
- Model equivalence: actual run manifests record `nvidia/nvidia/nemotron-3-super-v3`; the catalog shorthand `nvidia/nemotron-3-super-v3` is treated as exact-equivalent to `nvidia/nemotron-3-super-120b-a12b` for project provenance.
- Hosted endpoint: `https://inference-api.nvidia.com/v1`
- API key source: Kubernetes Secret reference; the value was not persisted
- Maximum input tokens per Curator segment: 1,000
- Minimum document and segment tokens: 30
- Maximum concurrent hosted requests: 2
- Elapsed time: 2,701.364 seconds
- Curator output segments: 1,145
- Raw artifact SHA-256: `de9d89c368b44f6c4bc562a87e49a67d88cd04a2f4d0d24aedc61c79ecf33143`
- Run: `data/nemo_usvcs_curated/qa/runs/full-20260629t2120z/`

## Normalization and finalization

- Parse failures: 0
- Exact duplicate segment-level QA pairs removed: 2,572
- Normalized QA pairs: 6,591
- Normalized QA SHA-256: `3a16da852edb9384309886777e1fb9573c5724896f51031331be637e334dad41`
- Global duplicate-question rows removed: 286
- Rows removed by the 32-pair-per-URL cap: 1,960
- Final rows: 4,345 across 410 source URL documents
- Training: 3,865 rows across 369 documents
- Validation: 480 rows across 41 documents
- Training SHA-256: `533ebfab9726157aece6e4251dd5bcc4ac3f7200858a2c3b1dc50e2941d2d025`
- Validation SHA-256: `e0d9de41b05fec6d0e706625e862193739f3d03a91d821912306176d1692b8bc`
- Final artifacts: `data/nemo_usvcs_curated/qa/runs/full-20260629t2120z/dataset/final/`

The 90/10 split is grouped by source document ID. Integrity checks found no document leakage, duplicate normalized questions, acknowledgment/EULA URLs, or empty required fields. The maximum retained yield is 32 QA pairs per source URL.

## Quantitative comparison with NIM

| Corpus | Curator input documents | Curator segments | Normalized QA | Final QA | Final source documents |
| --- | ---: | ---: | ---: | ---: | ---: |
| NIM | 451 | 1,286 | 9,962 | 7,660 | 442 |
| NeMo Microservices | 598 | 1,145 | 6,591 | 4,345 | 410 |

The different yields confirm a substantial corpus-dependent delta, but yield alone does not establish which dataset is better. A fair conclusion requires the same groundedness, answerability, duplication, and downstream evaluation protocol for both Curator and the prior logical-entailment datasets.

## Verification

The isolated test suite passed: 13 tests, 0 failures. The final artifact audit reported:

- duplicate questions: 0
- training/validation document overlap: 0
- acknowledgment or EULA rows: 0
- empty required fields: 0
- maximum pairs per document: 32
