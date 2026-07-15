# docs-to-data-to-lora Tutorial

This directory turns the docs-to-data-to-lora pipeline into a notebook-first tutorial. The notebooks are ordered so a reader can run local inspection and dry-run cells first, then opt into live cluster actions only when the required services are available.

## Notebook Order

| Notebook | Focus | Live systems needed |
|---|---|---|
| `notebooks/00_overview_and_setup.ipynb` | Repository orientation, config, environment checks | none |
| `notebooks/01_curated_crawl.ipynb` | Sitemap inventory, version curation, crawler payload design | optional network/crawler |
| `notebooks/02_dataset_creation.ipynb` | Stage 0 corpus prep, KVP curation, Customizer JSONL output | optional Elasticsearch/NIM |
| `notebooks/03_lora_training.ipynb` | LoRA adapter specs and NeMo Customizer job payloads | optional NeMo Customizer |
| `notebooks/04_evaluation.ipynb` | Completion collection, single-axis scoring, pairwise scoring, token accounting | optional NIM/Evaluator/judge API |

Start with `00_overview_and_setup.ipynb`. It defines the local repository path and the live-run guard pattern used by the rest of the tutorial.

## Companion Docs

- [execution-modes.md](execution-modes.md): how to move from local dry runs to live ES, Customizer, and Evaluator calls.
- [artifacts.md](artifacts.md): expected files produced by each stage.
- [evaluation.md](evaluation.md): evaluation waves, direct judge fallback, and token accounting.

## Safety Defaults

The notebooks are intentionally conservative:

- Cells that call live infrastructure are guarded by boolean flags.
- Training and evaluation examples prefer `--dry-run`, `--submit-only`, or `--limit` options where the underlying CLI supports them.
- Output examples write under `tutorial/_outputs/`, which is ignored by the tutorial narrative and can be deleted between runs.

## Prerequisites

Install the project dependencies before running the local cells:

```bash
python -m pip install -e ".[dev]"
```

The notebook smoke test uses the production-tokenizer path exercised by Stage 3. Set `LOCAL_NIM_CACHE` to a local NIM cache containing the Llama 3.1 8B tokenizer, or set `PIPELINE_STAGE3_TOKENIZER` directly to that tokenizer directory before running `python -m pytest tests/test_tutorial_notebooks.py`.

Live stages need access to the services configured in [execution-modes.md](execution-modes.md), including Elasticsearch, a generation NIM endpoint, NeMo Data Store, NeMo Entity Store, NeMo Customizer, NeMo Evaluator, and any external judge API used for release gates.
