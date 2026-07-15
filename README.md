# docs-to-data-to-lora

A reproducible NVIDIA AI Enterprise-centered workflow for turning scoped
enterprise domain content into domain-adapted LoRA adapters. The goal is domain
specificity, not a product-specific adapter recipe: start with unstructured
source material, preserve provenance, create grounded SFT data, train small dense
LoRA targets, and evaluate whether the adapter improves how a model handles the
domain's terminology, procedures, constraints, and answer style.

The NIM and NeMo Microservices corpora are representative public case studies
used to make the workflow concrete. The reusable pattern applies to any scoped
enterprise domain or sub-domain: operations runbooks, support playbooks,
policy manuals, field-service procedures, research methods, compliance content,
or internal platform documentation.

## Start Here

Use the notebooks when you want the fastest guided path through the repo. They
run local dry-run cells first and require live services only when you explicitly
turn on the guarded execution flags.

```bash
python -m pip install -e ".[dev]"
```

The tutorial notebook smoke test executes the local Stage 3 tokenizer path. If you already have the Llama 3.1 8B tokenizer locally, set `PIPELINE_STAGE3_TOKENIZER` to that directory. If not, request access to [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), set `HF_TOKEN`, and download only the tokenizer/config artifacts:

```bash
python scripts/pipeline/download_stage3_tokenizer.py \
  --output-dir outputs/tokenizers/llama-3.1-8b-instruct
export PIPELINE_STAGE3_TOKENIZER="$PWD/outputs/tokenizers/llama-3.1-8b-instruct"
```

```bash
python -m pytest tests/test_tutorial_notebooks.py
```

Then open [`tutorial/`](tutorial/):

| Notebook | Focus | Live systems needed |
|---|---|---|
| `00_overview_and_setup.ipynb` | Repository orientation, config, environment checks | none |
| `01_curated_crawl.ipynb` | Sitemap inventory, version curation, crawler payload design | optional crawler/network |
| `02_dataset_creation.ipynb` | Corpus prep, grounded KVP curation, Customizer JSONL output | optional Elasticsearch/NIM |
| `03_lora_training.ipynb` | LoRA adapter specs and NeMo Customizer job payloads | optional NeMo Customizer |
| `04_evaluation.ipynb` | Completion collection, scoring, pairwise comparison, token accounting | optional NIM/Evaluator/judge API |

For a pure documentation path, read the stages in order:

1. [`docs/stage-1-curated-crawl.md`](docs/stage-1-curated-crawl.md) - one
   ingestion methodology for web-sourced corpora.
2. [`docs/stage-2-dataset-creation.md`](docs/stage-2-dataset-creation.md) -
   grounded LE, synthesis, Curator finalization, and validation.
3. [`docs/stage-3-peft-training.md`](docs/stage-3-peft-training.md) - LoRA SFT,
   completion capture, and evaluation.
4. [`docs/results/evaluation-summary.md`](docs/results/evaluation-summary.md) -
   distilled case-study outcomes and SVG result graphics.

## NVIDIA AI Enterprise Foundation

This reference implementation is intentionally centered on NVIDIA AI Enterprise
(NVAIE) assets. NVAIE provides the production-grade foundation; this repo shows
how those building blocks can be composed, extended, and governed to solve real
business needs across enterprise domains.

The case study uses NVAIE capabilities across the lifecycle:

- **NIM and OpenAI-compatible inference endpoints** for generation, local
  serving, LoRA-enabled inference, embeddings, reranking, and dense-model
  baselines.
- **NeMo Curator and Data Designer** for dataset transformation, quality
  filtering, diverse QA generation, and optional targeted augmentation.
- **NeMo Customizer** for LoRA SFT over smaller dense models.
- **NeMo Evaluator and MLflow** for independent judging, artifact export,
  lineage, and publishable evaluation evidence.
- **NV-Ingest-oriented durable orchestration** through the companion
  [`nv-ingest-265-durable-orchestration`](https://github.com/joncoons/nv-ingest-265-durable-orchestration)
  path for PDF-heavy or file-based raw document processing.

The repo adds solution-level glue around those assets: LE-based QA/KVP
extraction, resumable JSONL stages, retry/failure accounting, multi-endpoint
load sharing, corpus-scoped RAG evaluation, golden-test construction, and
result graphics suitable for documentation.

## Pipeline Shape

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

Stage 1 can be a curated web crawl, durable PDF/document extraction, an internal
export, or any other source that produces provenance-preserving chunks in a
vector-backed collection. The curated crawl is only the reference ingestion
path used in this repo.

Stage 2 converts those chunks into grounded SFT rows. The LE path is designed
for small corpus specificity: it extracts premises, conclusions, and QA/KVP
rows with source provenance. Curator remains fundamental for deduplication,
filtering, dataset finalization, and scale-oriented data workflows.

Stage 3 trains and evaluates LoRA adapters on dense base models. The evaluation
matrix compares 1B, 3B, and 8B LoRA targets with rank 16 and rank 32 adapters,
then compares the strongest LoRA candidates against a dense Llama 3.3 70B
no-adapter reference target.

## What You Get

A LoRA adapter that can be loaded onto a base model through `NIM_PEFT_SOURCE` or
your preferred PEFT runtime. The adapter is meant to bias responses toward a
stable domain or sub-domain without retraining the base model. The value is
fluency with local terminology, abbreviations, operating procedures, policy
constraints, failure modes, and the way people inside the domain ask and answer
questions.

LoRA and RAG are complementary:

- **RAG** is right when answers need current or citable source passages.
- **LoRA SFT** is right when stable domain behavior and low per-request latency
  are important.
- **LoRA plus RAG** often gives the best production behavior: the adapter
  improves domain fluency and procedural style, while retrieval supplies fresh,
  auditable context.

## Case-Study Evidence

The public repo keeps distilled outcomes and graphics, not raw run logs,
generated corpora, or cluster-local artifacts. Start with
[`docs/results/evaluation-summary.md`](docs/results/evaluation-summary.md).

Included result graphics:

- [`docs/results/le-vs-curator-validation-loss.svg`](docs/results/le-vs-curator-validation-loss.svg)
- [`docs/results/golden-eval-score-table.svg`](docs/results/golden-eval-score-table.svg)
- [`docs/results/rag-vs-norag-axis-heatmap.svg`](docs/results/rag-vs-norag-axis-heatmap.svg)
- [`docs/results/rag-vs-norag-composite.svg`](docs/results/rag-vs-norag-composite.svg)
- [`docs/results/ragas-coverage-status.svg`](docs/results/ragas-coverage-status.svg)

Several methodology choices are deliberate:

- Nemotron 3 Super 120B-equivalent generation is the default for Stage 1A
  logical-entailment extraction and QA/KVP generation because it provides a
  strong quality/throughput balance for full-corpus grounded extraction.
- The evaluation matrix spans multiple dense model sizes to support practical
  downselection of the smallest LoRA target that meets the quality bar.
- The LE records preserve premise, conclusion, source context, and provenance,
  which makes the same data shape a plausible foundation for future NeMo RL or
  RLHF/RLAIF-style workflows.
- The LE approach complements NeMo Curator. It is aimed at small
  corpus-specific LoRA adaptation, while Curator remains the preferred
  foundation for massive-scale curation, filtering, synthetic data workflows,
  and full-SFT preparation.

See [`docs/methodology-rationale.md`](docs/methodology-rationale.md) for the
full rationale.

## Adapting This To Your Domain

The methodology generalizes to any corpus that can be chunked, embedded, and
stored with provenance. A curated web crawl works well when the source of truth
is a documentation site with a sitemap. For PDF-heavy, offline, or internal
repositories, run a durable extraction pipeline first and treat its chunks as
the Stage 1 collection.

Common raw-source paths include:

- Curated web crawl over HTML pages in an explicit prefix list.
- Linked binaries such as `.pdf`, `.docx`, and `.pptx`, parsed through a
  document extraction model after the crawl records a manifest.
- Linked plain text such as `.txt`, `.md`, and `.rst`, chunked inline alongside
  HTML.
- Durable document extraction for offline archives, policy manuals, runbooks,
  and file shares using a reliability-oriented ingestion path such as
  [`nv-ingest-265-durable-orchestration`](https://github.com/joncoons/nv-ingest-265-durable-orchestration).

For a new target:

1. Define the domain boundary: department, workflow, platform area, regulatory
   topic, operating function, knowledge area, or other sub-domain.
2. Choose an ingestion path: curated web crawl, durable PDF/document extraction,
   internal export, or a combination.
3. Preserve source provenance, source kind, timestamps or snapshot IDs, and
   chunk metadata before dataset generation.
4. Store chunks in a collection scoped to that domain boundary.
5. Use the worked examples as templates for scoping and provenance, not as
   limits on the subject matter.

## Reference Status

| Stage | Status | What is included |
|---|---|---|
| 1 - Corpus Ingestion | ready to use | Curated crawl methodology, inventory tool, two worked examples, deployment reference, companion durable extraction reference |
| 2 - Dataset Creation | ready to adapt | Resumable LE pipeline, optional Data Designer gap-fill, Curator handoff/finalization, provenance sidecars, validation gate |
| 3 - PEFT Training | ready to adapt | Customizer LoRA SFT helpers for dense Llama bases, evaluator registration, completion capture, MLflow export, golden-test workflow |

Generated datasets, training logs, raw evaluation traces, credentials, and
cluster-local run manifests are excluded from the public branch by design.

## Repo Layout

```
docs-to-data-to-lora/
├── README.md                              ← you are here
├── LICENSE                                ← Apache 2.0
├── docs/
│   ├── stage-1-curated-crawl.md           ← one corpus-ingestion methodology
│   ├── stage-2-dataset-creation.md        ← LE dataset creation + Curator finalization
│   ├── stage-3-peft-training.md           ← Customizer LoRA SFT + evaluation workflow
│   ├── methodology-rationale.md           ← why LE, Curator, model sizing, RL
│   ├── results/                           ← curated case-study results + SVGs
│   └── integrations/                      ← MLflow export + future orchestration guidance
│       └── mlflow-nemo/
├── examples/
│   ├── nim.md                             ← broad technical-docs corpus walkthrough
│   └── nemo-microservices.md              ← platform-operations corpus walkthrough
├── tutorial/                              ← dry-run-first notebook walkthrough
├── tests/                                 ← unit and smoke coverage for runnable paths
├── scripts/
│   ├── pipeline/                          ← dataset creation stages
│   ├── stage3/                            ← Customizer LoRA training helpers
│   ├── eval/                              ← completion capture and evaluation
│   └── sitemap_to_inventory.py            ← sitemap → CSV inventory tool
└── deployment/
    ├── kubernetes-runai.md                ← one concrete deployment target
    └── naming-conventions.md              ← non-local K8s/Run.ai naming runbook
```

## Quick Start: Stage 1 Inventory

```bash
python scripts/sitemap_to_inventory.py \
    --sitemap https://docs.example.com/<domain-area>/sitemap.xml \
    --path-prefix /<domain-area>/ \
    --output inventory.csv
```

Inspect the source-area summary printed to stdout, choose one canonical version
per source area, and build an `allowed_url_prefixes` list following
[`docs/stage-1-curated-crawl.md`](docs/stage-1-curated-crawl.md). The examples
in [`examples/`](examples/) show two concrete scoping patterns.

## Prerequisites

- **Python 3.12+** for the full repo. The sitemap inventory script itself is
  stdlib-only.
- **For web-sourced corpora: a BFS crawler with prefix scoping.** This repo
  references [`rag-crawler`](https://github.com/joncoons/rag-crawler); any
  equivalent crawler works if it can render the target docs, persist URL state,
  honor explicit prefix scopes, and capture linked files through narrow host
  allowlists.
- **For PDF-heavy or offline corpora: a durable document extraction path.** The
  companion [`nv-ingest-265-durable-orchestration`](https://github.com/joncoons/nv-ingest-265-durable-orchestration)
  repo is the preferred reference for reliable PDF/raw-document extraction,
  retry, checkpoint, and provenance capture before chunks enter this pipeline.
- **A vector store**, Elasticsearch in the reference path.
- **Embedding and reranking endpoints** for retrieval-backed stages and optional
  RAG/RAGAS diagnostics.
- **For Stages 2-3:** NIM, NeMo Curator/Data Designer, NeMo Customizer, and
  NeMo Evaluator in the reference path. Equivalent tools can be substituted,
  but the case study is intentionally NVAIE-centered.

## Worked Examples

Two representative public corpora show the methodology applied to different
documentation shapes:

- [examples/nim.md](examples/nim.md) - a broad public technical documentation
  corpus with many source areas under one umbrella, mixed versioning
  conventions, and strong source-area imbalance.
- [examples/nemo-microservices.md](examples/nemo-microservices.md) - a
  platform-operations documentation corpus narrowed to one coherent
  single-prefix scope.

Both are case studies, not the boundary of the approach.

## Stage 2 And Stage 3

Stage 2 converts Stage 1 collections into `training.jsonl` and
`validation.jsonl` in NeMo Customizer SFT format:

- Stage 1A - Logical Entailment to KVP rows.
- Stage 1B - Semantic neighborhood synthesis.
- Stage 1C - Instruction diversity.
- Stage 1.5 - optional Data Designer gap-fill.
- Stage 2 - QA admission and refinement.
- Stage 3 - Curator finalization.
- Stage 4 - independent validation gate.

See [`docs/stage-2-dataset-creation.md`](docs/stage-2-dataset-creation.md).
Implementation entry point: `scripts/build_v2_dataset.py`.

Stage 3 registers datasets, trains dense-model LoRA adapters with NeMo
Customizer, collects no-RAG and optional RAG completions, and evaluates the
results. See [`docs/stage-3-peft-training.md`](docs/stage-3-peft-training.md).

## Integrations

For MLflow lineage around the NVIDIA AI Enterprise / NeMo lifecycle, see
[`docs/integrations/mlflow-nemo`](docs/integrations/mlflow-nemo/).

Deployable today:

- MLflow-ready dataset registration observability.
- Evaluation artifact and metric export.
- Stage observability artifacts that can be uploaded to MLflow by an export step.

Future extension guidance covers a fuller MLflow parent/child run graph across
Data Store, Entity Store, Customizer, Evaluator, and promotion.

## Contributing

Issues and PRs welcome. Keep generated corpora, logs, credentials,
cluster-specific run manifests, and one-off experiment outputs out of public
commits.

## License

Apache 2.0 - see [LICENSE](LICENSE).
