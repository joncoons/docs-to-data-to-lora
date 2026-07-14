# docs-to-data-to-lora

A reproducible three-stage pipeline for turning unstructured enterprise
domain content into domain-adapted LoRA adapters. The purpose is to show how
NVIDIA AI Enterprise (NVAIE) assets can be composed into a repeatable domain
specificity workflow: ingest a scoped corpus, generate grounded training data,
train LoRA adapters, and evaluate the result. The NIM and NeMo corpora are
representative public examples used to make the workflow concrete; the reusable
pattern is domain adaptation for any scoped enterprise corpus.

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
business needs across any enterprise domain or sub-domain.

The representative case study uses NVAIE capabilities across the lifecycle:

- **NIM and OpenAI-compatible inference endpoints** for foundation/frontier model
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

Several methodology choices are deliberate and are independent of the example corpora:

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

## LoRA, RAG, Or Both?

RAG retrieves at inference time; LoRA fine-tunes once. They're
complementary:

- **RAG** is right when answers must cite source passages, when docs
  change frequently, or when you can't trust the model to memorize.
- **LoRA SFT** is right when the model needs to *speak the domain* - use
  local nomenclature correctly, follow domain procedures, understand
  acronyms and process-specific constraints, and produce answers in the
  expected operational style with minimal per-request overhead.

For latency-sensitive stable workflows, operating procedures, policy
interpretation, troubleshooting patterns, and domain vocabulary, a LoRA
adapter can reduce the amount of retrieval-time work needed to produce useful
domain-specific answers. That does not make LoRA a replacement for
RAG. When source grounding, freshness, or auditability matters, RAG remains the
right retrieval layer. In many production systems, LoRA plus RAG is the
strongest pattern: the adapter improves domain fluency and procedural behavior,
while retrieval supplies current, citable context.

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
│   ├── results/                            ← curated case-study results + SVGs
│   ├── integration-templates/              ← MLflow/NeMo orchestration plans
│   └── integrations/                       ← optional Stage 1 enhancements
│       ├── README.md                      (decision table — when to use which)
│       ├── 01-semantic-chunking.md        (element-aware chunker for HTML/MD/PDF)
│       ├── 02-cross-page-text-stitching.md (sentence stitching across PDF pages)
│       ├── 03-visual-stitching-and-routing.md (table/figure reassembly via Nemotron-Parse 1.2)
│       └── 04-durable-document-extraction.md (external durable PDF/document ingestion path)
├── examples/
│   ├── nim.md                             ← NVIDIA Inference Microservices walkthrough
│   └── nemo-microservices.md              ← NVIDIA NeMo Microservices walkthrough
├── scripts/
│   ├── pipeline/                           ← dataset creation stages
│   ├── stage3/                             ← Customizer LoRA training helpers
│   ├── eval/                               ← completion capture and evaluation
│   └── sitemap_to_inventory.py             ← sitemap → CSV inventory tool
└── deployment/
    └── kubernetes-runai.md                ← one concrete deployment target
```

## Status

| Stage | Status | What's done |
|---|---|---|
| 1 — Corpus Ingestion | ready to use | Curated crawl methodology, inventory tool, two worked examples, Kubernetes deployment reference, optional durable document extraction path |
| 2 — Dataset Creation | ready to adapt | Resumable LE pipeline, optional Data Designer gap-fill, Curator handoff/finalization, provenance sidecars, validation gate |
| 3 — PEFT Training | ready to adapt | Customizer LoRA SFT helpers for dense Llama bases, evaluator registration, completion capture, MLflow export, golden-test workflow |

The repo is intended to be adapted to your domain corpus. Generated datasets,
training logs, raw evaluation traces, and cluster-local run manifests are
excluded from the public branch by design.

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



## Case-Study Results

The public repo includes distilled evaluation outcomes, but not raw run logs or
generated corpora. See [`docs/results/evaluation-summary.md`](docs/results/evaluation-summary.md)
for the validation-loss comparison, golden-test methodology, no-RAG/RAG score
charts, and the current RAGAS diagnostic status.

## Tutorial Notebooks

A notebook-first walkthrough now lives under [`tutorial/`](tutorial/). It covers the complete flow from curated crawl through dataset creation, LoRA training, and evaluation, with companion markdown docs for execution modes, artifacts, and eval strategy.

## Prerequisites

- **Python 3.10+** for the inventory script (no external deps; stdlib only).
- **For web-sourced corpora: a BFS crawler with prefix scoping.** This repo references rag-crawler;
  any equivalent (Scrapy with a custom downloader middleware, your own
  Selenium-based crawler) will work as long as it can:
  - accept a list of allowed URL prefixes as scope
  - render single-page-application docs sites (Fern, modern Sphinx with
    JS-heavy themes — many documentation sites need this)
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

Two representative public corpora are included to show the methodology
applied to different documentation shapes. They are case studies, not the
boundary of the approach:

- [**examples/nim.md**](examples/nim.md) — a broad public technical
  documentation corpus with many source areas under one umbrella, mixed
  versioning conventions, and strong source-area imbalance.
- [**examples/nemo-microservices.md**](examples/nemo-microservices.md) —
  a platform-operations documentation corpus narrowed to one coherent
  single-prefix scope.

Both are real curations used to build the representative LoRA adapters.
They demonstrate domain adaptation over enterprise technical documentation,
but the same pattern applies to other domains such as internal operations,
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
   regulatory topic, operating function, knowledge area, or other sub-domain.
2. Choose an ingestion path: curated web crawl, durable PDF/document
   extraction, internal export, or a combination.
3. Preserve source provenance, source kind, timestamps or snapshot IDs,
   and chunk metadata before dataset generation.
4. Store chunks in a collection scoped to that domain boundary.
5. Use the worked examples as templates for scoping and provenance, not as
   limits on the subject matter.

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

Issues and PRs welcome. Keep generated corpora, logs, credentials, cluster-specific
run manifests, and one-off experiment outputs out of public commits.

## License

Apache 2.0 — see [LICENSE](LICENSE).
