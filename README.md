# docs-to-data-to-lora

A reproducible three-stage pipeline for turning a vendor's documentation
site into a domain-expert LoRA adapter, grounded entirely in the vendor's
published docs.

```
   ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
   │      STAGE 1        │    │      STAGE 2        │    │      STAGE 3        │
   │  Curated Crawl      │───▶│  Dataset Creation   │───▶│  PEFT Training      │
   │                     │    │                     │    │                     │
   │  sitemap → filter   │    │  entailment + Data  │    │  LoRA SFT on small  │
   │  → prefix scope     │    │  Designer augment   │    │  base model         │
   │  → ES collection    │    │  → SFT JSONL        │    │  → adapter weights  │
   └─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## What you get at the end

A LoRA adapter you can load onto a base model (via `NIM_PEFT_SOURCE` or
your preferred PEFT runtime) that biases responses toward a specific
product domain — without retraining the base, and grounded entirely in
the vendor's official documentation.

The Stage 1 corpus covers more than just HTML:

- **HTML pages** in the curated prefix list (primary source).
- **Linked binaries** (`.pdf`, `.docx`, `.pptx`) referenced by those pages,
  fetched from the docs host and known vendor CDNs and parsed via a PDF
  extraction model.
- **Linked plain-text** (`.txt`, `.md`, `.rst`) including GitHub-hosted
  READMEs from the vendor's own organization, chunked inline alongside the
  HTML content.

See [`docs/stage-1-curated-crawl.md`](docs/stage-1-curated-crawl.md) for
the full scope details and host-allowlist guidance.

## Why not RAG instead?

RAG retrieves at inference time; LoRA fine-tunes once. They're
complementary:

- **RAG** is right when answers must cite source passages, when docs
  change frequently, or when you can't trust the model to memorize.
- **LoRA SFT** is right when the model needs to *speak the domain* — get
  product names right, follow API conventions, understand vendor
  terminology — without consulting docs every turn.

For deployment guides, troubleshooting, "which NIM should I use for X"
type questions, a LoRA adapter handles the bulk of queries faster and
cheaper than RAG. RAG fills in for the long tail.

## Repo layout

```
docs-to-data-to-lora/
├── README.md                              ← you are here
├── LICENSE                                ← Apache 2.0
├── docs/
│   ├── stage-1-curated-crawl.md           ← methodology + crawl logic
│   ├── stage-2-dataset-creation.md        ← entailment + augmentation (WIP)
│   ├── stage-3-peft-training.md           ← LoRA SFT pipeline (WIP)
│   ├── integration-templates/              ← MLflow/NeMo orchestration plans
│   └── integrations/                      ← optional Stage 1 enhancements
│       ├── README.md                      (decision table — when to use which)
│       ├── 01-semantic-chunking.md        (element-aware chunker for HTML/MD/PDF)
│       ├── 02-cross-page-text-stitching.md (sentence stitching across PDF pages)
│       └── 03-visual-stitching-and-routing.md (table/figure reassembly via Nemotron-Parse 1.2)
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
| 1 — Curated Crawl | ✅ ready to use | Methodology, inventory tool, two worked examples (NIM, NeMo Microservices), Kubernetes deployment reference |
| 2 — Dataset Creation | 🚧 in progress | Approach outlined, pipeline not yet built |
| 3 — PEFT Training | 🚧 in progress | Approach outlined, training scripts not yet committed |

Stage 1 is self-contained — you can use it today to build curated docs
collections without committing to the later stages.

## Quick start (Stage 1)

```bash
# 1. Inventory a vendor's docs site
python scripts/sitemap_to_inventory.py \
    --sitemap https://docs.example.com/<product>/sitemap.xml \
    --path-prefix /<product>/ \
    --output inventory.csv

# 2. Inspect the per-product summary printed to stdout. Identify products
#    that need pinned versions (no /latest/) and review for completeness.

# 3. Build an allowed_url_prefixes list following the methodology in
#    docs/stage-1-curated-crawl.md. See examples/ for two worked patterns.

# 4. Hand the prefix list to a BFS crawler that supports prefix scoping.
#    See deployment/kubernetes-runai.md for one concrete setup using
#    rag-crawler (https://github.com/joncoons/rag-crawler).
```

## Prerequisites

- **Python 3.10+** for the inventory script (no external deps; stdlib only).
- **A BFS crawler with prefix scoping.** This repo references rag-crawler;
  any equivalent (Scrapy with a custom downloader middleware, your own
  Selenium-based crawler) will work as long as it can:
  - accept a list of allowed URL prefixes as scope
  - render single-page-application docs sites (Fern, modern Sphinx with
    JS-heavy themes — many vendor sites need this)
  - persist a URL registry across restarts
- **A vector store** (Elasticsearch in our reference setup; pgvector, Milvus,
  Qdrant all work).
- **An embedding endpoint** (an NIM, an OpenAI-compatible API, or any
  HTTP-accessible embedding model).
- **For Stages 2-3:** NVIDIA NeMo Platform 26.3.1, including Data
  Designer, Customizer, Evaluator, Entity/File services, and the NIM-backed
  inference gateway used by the showcase jobs.

## Worked examples

Two are included to show the methodology applied to different shapes of
documentation:

- [**examples/nim.md**](examples/nim.md) — NVIDIA Inference Microservices
  (`docs.nvidia.com/nim`). 50 sub-products under one umbrella, mixed
  versioning conventions, one dominant product creating corpus bias.
- [**examples/nemo-microservices.md**](examples/nemo-microservices.md) —
  NVIDIA NeMo Microservices (`docs.nvidia.com/nemo/microservices`). A
  sprawling product family narrowed to a single-prefix scope.

Both are real curations used to build the case-study LoRA adapters this
project is centered on.

## Adapting this for your domain

The methodology generalizes to any documentation site with a published
sitemap. The patterns the examples surface — multiple products under one
umbrella, `/latest/` symlinks, version anomalies, name collisions between
OSS and platform variants — show up across most large vendor docs sites.

To apply this to your own target:

1. Find the sitemap (`robots.txt` or `/sitemap.xml`).
2. Run `scripts/sitemap_to_inventory.py` and read the per-product summary.
3. Follow the four-step methodology in [stage-1-curated-crawl.md](docs/stage-1-curated-crawl.md).
4. The worked examples are templates — copy the closest one and adjust
   the prefix list.

## Stage 2: Dataset Creation

Methodology and pipeline for converting Stage 1 ES corpora into training-ready
SFT JSONL via three grounded generation strategies, RAG-augmented gap-fill, and
an external-judge validation gate.

See [`docs/stage-2-dataset-creation.md`](docs/stage-2-dataset-creation.md) for
the full walkthrough. Pipeline implementation: `scripts/build_v2_dataset.py`.

The pipeline runs independently per collection and produces a `training.jsonl` +
`validation.jsonl` pair in NeMo Customizer SFT format (90/10 split, single-turn
`{prompt, completion, system}`). Two Stage 1 collections → two parallel runs →
two independent datasets for two product-specific LoRA adapters.

Generation strategies:
- **Stage 1A — Logical Entailment → KVP**: extract premises and conclusions from
  each passage; convert to Q+A pairs.
- **Stage 1B — Semantic Neighborhood Synthesis**: kNN retrieval per passage;
  generate BRIDGING and CONTRASTIVE questions requiring cross-passage synthesis.
- **Stage 1C — Instruction Diversity**: top-25% passages by density score;
  SUMMARY / LISTICLE / PROCEDURAL instruction-following examples.
- **Stage 1.5 — Gap analysis + Data Designer handoff**: bias analysis by
  `product_family`; emit `gap_manifest.json` and Data Designer seed records for
  native synthetic generation of under-represented products.
- **Stage 4 — External judge**: Claude Sonnet 4.6 (independent of generation)
  spot-checks 100 pairs per collection; 90% grounding pass threshold.

## Integration templates

For teams that want MLflow to orchestrate and audit the NeMo Platform
lifecycle, see
[`docs/integration-templates/mlflow-nemo`](docs/integration-templates/mlflow-nemo/).
The template keeps NeMo Platform as the execution plane while MLflow records
lineage, IDs, metrics, artifacts, and promotion state.

## Contributing

Issues and PRs welcome. The current open work is captured in each stage
doc under "Open work" — Stage 3 is the active edge.

## License

Apache 2.0 — see [LICENSE](LICENSE).
