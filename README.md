# docs-to-data-to-lora

A reproducible three-stage pipeline for turning unstructured enterprise
domain content into domain-adapted LoRA adapters. The goal is not to create
a vendor-product adapter specifically; it is to teach a smaller base model
the nomenclature, procedures, constraints, and evaluation language of any
enterprise domain or sub-domain while preserving source provenance.

```
   ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
   │      STAGE 1        │    │      STAGE 2        │    │      STAGE 3        │
   │  Corpus Ingestion   │───▶│  Dataset Creation   │───▶│  PEFT Training      │
   │                     │    │                     │    │                     │
   │  crawl / extract    │    │  entailment + Data  │    │  LoRA SFT on small  │
   │  → chunk / embed    │    │  Designer augment   │    │  base model         │
   │  → collection       │    │  → SFT JSONL        │    │  → adapter weights  │
   └─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## NVIDIA AI Enterprise Foundation

This reference implementation is centered on NVIDIA AI Enterprise (NVAIE)
assets. NVAIE provides the production-grade foundation; this repo shows how
those building blocks can be composed, extended, and governed to solve real
business needs.

The case study uses NVAIE capabilities across the lifecycle:

- **NIM and NVIDIA-hosted inference endpoints** for foundation/frontier model
  generation, local serving, LoRA-enabled inference, embeddings, reranking,
  and dense-model baselines.
- **NeMo Curator and Data Designer** for dataset transformation, quality
  filtering, diverse QA generation, and optional targeted augmentation.
- **NeMo Customizer** for LoRA SFT over smaller dense models.
- **NeMo Evaluator and MLflow** for independent judging, artifact export,
  lineage, and publishable evaluation evidence.
- **NV-Ingest-style durable orchestration** for PDF and raw-document
  extraction through the companion
  [`nv-ingest-265-durable-orchestration`](https://github.com/joncoons/nv-ingest-265-durable-orchestration)
  path.

The repo adds solution-level glue around those assets: LE-based QA/KVP
extraction, resumable JSONL stages, retry/failure accounting, multi-endpoint
load sharing, corpus-scoped RAG evaluation, golden-test construction, and
documentation-ready result graphics.

Several methodology choices are deliberate:

- Nemotron 3 Super 120B-equivalent generation is the default for Stage 1A
  logical-entailment extraction and QA/KVP generation because it provides a
  strong quality/throughput balance for full-corpus grounded extraction.
- The evaluation matrix spans 1B, 3B, and 8B dense LoRA adapters, then compares
  the strongest LoRA candidates against a dense Llama 3.3 70B reference target
  to support practical model downselection.
- The LE records preserve premise, conclusion, source context, and provenance,
  which makes the same data shape a plausible foundation for future NeMo RL or
  RLHF/RLAIF-style preference and reward workflows.
- The LE path complements NeMo Curator; it is aimed at small corpus-specific
  domain adaptation, while Curator remains the preferred foundation for massive
  scale curation, filtering, synthetic data workflows, and full-SFT preparation.

See [`docs/methodology-rationale.md`](docs/methodology-rationale.md) for the
full rationale.

## What you get at the end

A LoRA adapter you can load onto a base model (via `NIM_PEFT_SOURCE` or
your preferred PEFT runtime) that biases responses toward a specific
domain or sub-domain without retraining the base model. The value is
domain specificity: understanding local terminology, abbreviations,
operating procedures, policy constraints, failure modes, and the way people
inside the domain ask and answer questions.

The reference Stage 1 path uses a curated web crawl, but that is only one
way to gather raw unstructured data. The pipeline can start from any
source that can produce provenance-preserving chunks in a vector-backed
collection.

Common raw-source paths include:

- **Curated web crawl** over HTML pages in an explicit prefix list.
- **Linked binaries** (`.pdf`, `.docx`, `.pptx`) referenced by those pages,
  fetched from allowed hosts and parsed via a document extraction model.
- **Linked plain-text** (`.txt`, `.md`, `.rst`) including GitHub-hosted
  READMEs or internal engineering docs, chunked inline alongside HTML.
- **Durable document extraction** for PDF-heavy corpora, offline archives,
  knowledge bases, policy manuals, runbooks, and other non-web sources. The
  sibling repo
  [`nv-ingest-265-durable-orchestration`](https://github.com/joncoons/nv-ingest-265-durable-orchestration)
  is the preferred reference path for reliability/durability around PDF and
  raw document processing.

See [`docs/stage-1-curated-crawl.md`](docs/stage-1-curated-crawl.md) for
the full scope details and host-allowlist guidance.

## Why not RAG instead?

RAG retrieves at inference time; LoRA fine-tunes once. They're
complementary:

- **RAG** is right when answers must cite source passages, when docs
  change frequently, or when you can't trust the model to memorize.
- **LoRA SFT** is right when the model needs to *speak the domain* — use
  local nomenclature correctly, follow domain procedures, understand
  acronyms and process-specific constraints, and produce answers in the
  expected operational style without consulting source material every turn.

For stable workflows, operating procedures, policy interpretation,
troubleshooting patterns, and domain vocabulary, a LoRA adapter can handle
the bulk of queries faster and cheaper than RAG. RAG fills in for the long
tail, fast-changing facts, and source-cited answers.

## Repo layout

```
docs-to-data-to-lora/
├── README.md                              ← you are here
├── LICENSE                                ← Apache 2.0
├── docs/
│   ├── stage-1-curated-crawl.md           ← one corpus-ingestion methodology
│   ├── stage-2-dataset-creation.md        ← entailment + augmentation (WIP)
│   ├── stage-3-peft-training.md           ← LoRA SFT pipeline (WIP)
│   ├── methodology-rationale.md           ← why LE, Curator, model sizing, RL
│   ├── integration-templates/              ← MLflow/NeMo orchestration plans
│   └── integrations/                      ← optional Stage 1 enhancements
│       ├── README.md                      (decision table — when to use which)
│       ├── 01-semantic-chunking.md        (element-aware chunker for HTML/MD/PDF)
│       ├── 02-cross-page-text-stitching.md (sentence stitching across PDF pages)
│       ├── 03-visual-stitching-and-routing.md (table/figure reassembly via Nemotron-Parse 1.2)
│       └── 04-durable-document-extraction.md (external durable PDF/document ingestion path)
├── examples/
│   ├── nim.md                             ← NVIDIA Inference Microservices walkthrough
│   └── nemo-microservices.md              ← NVIDIA NeMo Microservices walkthrough
├── scripts/
│   └── sitemap_to_inventory.py            ← sitemap → CSV inventory tool
└── deployment/
    └── kubernetes-runai.md                ← one concrete deployment target
```

## Status

| Stage | Status | What's done |
|---|---|---|
| 1 — Corpus Ingestion | ✅ ready to use | Curated crawl methodology, inventory tool, two worked examples (NIM, NeMo Microservices), Kubernetes deployment reference, optional durable document extraction path |
| 2 — Dataset Creation | 🚧 in progress | Approach outlined, pipeline not yet built |
| 3 — PEFT Training | 🚧 in progress | Approach outlined, training scripts not yet committed |

Stage 1 is self-contained — you can use it today to build domain corpora
without committing to the later stages.

## Quick start (Stage 1)

```bash
# 1. Inventory a public or internal docs site
python scripts/sitemap_to_inventory.py \
    --sitemap https://docs.example.com/<domain-area>/sitemap.xml \
    --path-prefix /<domain-area>/ \
    --output inventory.csv

# 2. Inspect the source-area summary printed to stdout. Identify sections
#    that need pinned versions (no /latest/) and review for completeness.

# 3. Build an allowed_url_prefixes list following the methodology in
#    docs/stage-1-curated-crawl.md. See examples/ for two worked patterns.

# 4. Hand the prefix list to a BFS crawler that supports prefix scoping.
#    See deployment/kubernetes-runai.md for one concrete setup using
#    rag-crawler (https://github.com/joncoons/rag-crawler).
```


## Tutorial Notebooks

A notebook-first walkthrough now lives under [`tutorial/`](tutorial/). It covers the complete flow from curated crawl through dataset creation, LoRA training, and evaluation, with companion markdown docs for execution modes, artifacts, and eval strategy.

## Prerequisites

- **Python 3.10+** for the inventory script (no external deps; stdlib only).
- **For web-sourced corpora: a BFS crawler with prefix scoping.** This repo references rag-crawler;
  any equivalent (Scrapy with a custom downloader middleware, your own
  Selenium-based crawler) will work as long as it can:
  - accept a list of allowed URL prefixes as scope
  - render single-page-application docs sites (Fern, modern Sphinx with
    JS-heavy themes — many vendor sites need this)
  - persist a URL registry across restarts
- **For PDF-heavy or offline document corpora: a durable document extraction path.**
  The companion
  [`nv-ingest-265-durable-orchestration`](https://github.com/joncoons/nv-ingest-265-durable-orchestration)
  repo is the preferred reference for reliable PDF/raw-document extraction,
  retry, checkpoint, and provenance capture before chunks enter this pipeline.
- **A vector store** (Elasticsearch in our reference setup; pgvector, Milvus,
  Qdrant all work).
- **An embedding endpoint** (an NIM, an OpenAI-compatible API, or any
  HTTP-accessible embedding model).
- **For Stages 2-3:** NVIDIA AI Enterprise assets in the reference path: NIM,
  NeMo Curator/Data Designer, NeMo Customizer, and NeMo Evaluator. Equivalent
  tools can be substituted, but the case study is intentionally NVAIE-centered.

## Worked examples

Two are included to show the methodology applied to different shapes of
documentation. They are case studies, not the boundary of the approach:

- [**examples/nim.md**](examples/nim.md) — NVIDIA Inference Microservices
  (`docs.nvidia.com/nim`). 50 sub-products under one umbrella, mixed
  versioning conventions, one dominant product creating corpus bias.
- [**examples/nemo-microservices.md**](examples/nemo-microservices.md) —
  NVIDIA NeMo Microservices (`docs.nvidia.com/nemo/microservices`). A
  sprawling product family narrowed to a single-prefix scope.

Both are real curations used to build the case-study LoRA adapters. They
demonstrate domain adaptation over enterprise technical documentation, but
the same pattern applies to other domains such as internal operations,
support playbooks, manufacturing procedures, compliance manuals, research
methods, or field-service runbooks.

## Adapting this for your domain

The methodology generalizes to any domain corpus that can be chunked,
embedded, and stored with provenance. A curated web crawl works well when
the source of truth is a documentation site with a sitemap. For PDF-heavy,
offline, or internal repositories, use a durable extraction pipeline first
and treat its chunks as the Stage 1 collection.

To apply this to your own target:

1. Define the domain boundary: department, workflow, platform area,
   regulatory topic, product family, or other sub-domain.
2. Choose an ingestion path: curated web crawl, durable PDF/document
   extraction, internal export, or a combination.
3. Preserve source provenance, source kind, timestamps or snapshot IDs,
   and chunk metadata before dataset generation.
4. Store chunks in a collection scoped to that domain boundary.
5. Use the worked examples as templates for scoping and provenance, not as
   product-only patterns.

## Stage 2: Dataset Creation

Methodology and pipeline for converting Stage 1 ES corpora into training-ready
SFT JSONL via three grounded generation strategies, RAG-augmented gap-fill, and
an external-judge validation gate.

See [`docs/stage-2-dataset-creation.md`](docs/stage-2-dataset-creation.md) for
the full walkthrough. Pipeline implementation: `scripts/build_v2_dataset.py`.

The pipeline runs independently per collection and produces a `training.jsonl` +
`validation.jsonl` pair in NeMo Customizer SFT format (90/10 split, single-turn
`{prompt, completion, system}`). Two Stage 1 collections → two parallel runs →
two independent datasets for two domain-specific LoRA adapters.

Generation strategies:
- **Stage 1A — Logical Entailment → KVP**: extract premises and conclusions from
  each passage; convert to Q+A pairs.
- **Stage 1B — Semantic Neighborhood Synthesis**: kNN retrieval per passage;
  generate BRIDGING and CONTRASTIVE questions requiring cross-passage synthesis.
- **Stage 1C — Instruction Diversity**: top-25% passages by density score;
  SUMMARY / LISTICLE / PROCEDURAL instruction-following examples.
- **Stage 1.5 — Gap analysis + Data Designer handoff**: bias analysis by
  domain-slice metadata; emit `gap_manifest.json` and Data Designer seed
  records for native synthetic generation of under-represented slices.
- **Stage 4 — External judge**: an independent judge model spot-checks 100
  pairs per collection; 90% grounding pass threshold.

## Integration templates

For teams that want MLflow to orchestrate and audit the NVIDIA AI Enterprise
/ NeMo Microservices lifecycle, see
[`docs/integration-templates/mlflow-nemo`](docs/integration-templates/mlflow-nemo/).
The template keeps NeMo Data Store, Entity Store, Customizer, and Evaluator as
the execution plane while MLflow records lineage, IDs, metrics, and promotion
state.

## Contributing

Issues and PRs welcome. The current open work is captured in each stage
doc under "Open work" — Stage 3 is the active edge.

## License

Apache 2.0 — see [LICENSE](LICENSE).
