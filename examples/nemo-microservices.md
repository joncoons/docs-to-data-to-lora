# Example: NVIDIA NeMo Microservices

A worked example of [Stage 1 methodology](../docs/stage-1-curated-crawl.md)
applied to a public platform-operations documentation corpus — and a study
in **how to narrow a sprawling documentation umbrella to a single coherent
scope**.

This is an interesting case because what looks like one documentation area is
actually a portfolio of component areas at different maturity levels, with two
distinct URL prefixes on the same docs host, and several name-collisions
between OSS libraries and platform microservices.

## The documentation-umbrella problem

`docs.nvidia.com/nemo/` has 14,440 URLs in its main sitemap. A broad crawl
gets all of it. Most of it is **not** NeMo Microservices.

Top-level structure under `/nemo/`:

| Source area | URLs | Versioning | What it is |
|---|---|---|---|
| `microservices/` | 8,834 | CalVer + `/latest/` | The hosted microservices platform |
| `retriever/` | 3,858 | CalVer + `/latest/` | Standalone retrieval extraction service |
| `agent-toolkit/` | 1,689 | semver 1.0→1.6, **no `/latest/`** | Agent-building SDK |
| `rl/` | 55 | 0.x | RL training library |
| `export-deploy/` | 4 | versioned | Model export tooling |
| `curator/` (separate sitemap) | 568 | unversioned (Fern) | Data curation SDK |
| `gym/` (separate sitemap) | 107 | unversioned (Fern) | Evaluation harness |

**Plus** `docs.nvidia.com/nemo-framework/` (note the hyphen — different
path prefix): 2,036 URLs, the NeMo Framework user guide. Easy to miss.

**Plus** five live component portals that have **no sitemap at all**:
`/nemo/automodel/latest/`, `/nemo/evaluator/latest/`, `/nemo/megatron-bridge/latest/`,
`/nemo/run/latest/`, `/nemo/guardrails/`.

If you start with "I want NeMo docs in my SFT corpus," you have to choose
which of these to include. For this example, the choice is **NeMo
Microservices only** — the hosted platform.

## Step 1 — Find the sitemap

```
$ curl -s https://docs.nvidia.com/s3-sitemap-index.xml \
    | grep -oE 'https://[^<]+sitemap[^<]+' \
    | grep -i 'nemo'
https://docs.nvidia.com/nemo/nemo-s3-sitemap.xml          ← contains microservices/
https://docs.nvidia.com/nemo-framework/...                ← different documentation area
https://docs.nvidia.com/nemotron/...                      ← different documentation family
...
```

NeMo Microservices URLs live in the main `nemo-s3-sitemap.xml` under the
`/nemo/microservices/` path prefix.

## Step 2 — Inventory the URL list

```bash
python scripts/sitemap_to_inventory.py \
    --sitemap https://docs.nvidia.com/nemo/nemo-s3-sitemap.xml \
    --path-prefix /nemo/microservices/ \
    --output nemo_microservices_inventory.csv
```

Version distribution (from sitemap):

```
microservices/26.3.0    639 URLs   ← /latest/ symlink target
microservices/26.3.1    303 URLs
microservices/25.12.0  1955 URLs
microservices/25.11.0  1768 URLs
microservices/25.10.0  1002 URLs
... (older versions)
```

`microservices/latest/` resolves to `microservices/26.3.0/` and serves the
639 URLs you want.

## Step 3 — Curate

Unlike the [NIM example](nim.md), this scope is one coherent documentation
area with a single canonical version. No per-source-area enumeration needed. The curated
prefix list has exactly one entry:

```json
[ "https://docs.nvidia.com/nemo/microservices/latest" ]
```

## Step 4 — Crawl

```json
{
  "start_url": "https://docs.nvidia.com/nemo/microservices/latest/",
  "collection_name": "nemo_usvcs_curated",
  "use_product_url_map": false,
  "max_depth": null,
  "use_sitemap": false,
  "max_pages": null,
  "extract_linked_files": true,
  "allowed_url_prefixes": ["https://docs.nvidia.com/nemo/microservices/latest"],
  "unblock_url_patterns": ["github.com"],
  "binary_host_allowlist": [
    "raw.githubusercontent.com/nvidia",
    "raw.githubusercontent.com/NVlabs",
    "raw.githubusercontent.com/Project-MONAI",
    "raw.githubusercontent.com/rapidsai",
    "raw.githubusercontent.com/triton-inference-server",
    "raw.githubusercontent.com/isaac-sim",
    "raw.githubusercontent.com/isaac-for-healthcare",
    "raw.githubusercontent.com/nvpro-samples",
    "raw.githubusercontent.com/nv-tlabs",
    "raw.githubusercontent.com/OE4T",
    "raw.githubusercontent.com/NVDLI"
  ]
}
```

`use_product_url_map: false` is a rag-crawler-specific legacy field name for
binary routing. It is not a statement about product-scoped adaptation; it keeps
all linked binaries grouped under the explicit `collection_name`.

Resulting corpus: **~639 HTML URLs** from `/microservices/latest/`, plus
whatever binaries and inline-text files those pages link out to (counted
post-crawl from the binary manifest).

## Step 5 — Phase 3 binary parse

```bash
curl -sf -X POST http://<ingestor-host>:8082/batch-ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_name": "nemo_usvcs_curated",
    "method": "auto",
    "force_reingest": false,
    "max_files": 500,
    "max_concurrent": 4
  }'
```

NeMo Microservices docs lean heavily on inline code samples and YAML
configs rather than linked PDFs — expect fewer binaries than the NIM crawl
proportionally, with the binary path mostly catching GitHub READMEs
linked from setup/integration sections.

## What's inside `/microservices/latest/`

The single-prefix scope naturally captures sub-component docs that
share names with standalone OSS projects:

| Sub-path | URLs | What it covers |
|---|---|---|
| `pysdk/` | 86 | Python SDK for the platform |
| `set-up/` | 81 | Installation and configuration |
| `evaluate/` | 50 | Evaluation workflows |
| `fine-tune/` | 42 | Customization workflows |
| `guardrails/` | 34 | **Guardrails microservice** (platform-ops view) |
| `customizer/` | 30 | Customizer microservice |
| `evaluator/` | 24 | **Evaluator microservice** (platform-ops view) |
| ...(others)... | | |

This is what we want for a platform-operator SFT adapter.

## What we deliberately excluded

The trickiest curation decision was around **same-name, different-audience**
areas. Three pairs exist:

| Standalone portal | Microservices subdir |
|---|---|
| `/nemo/evaluator/latest/` → **Evaluator SDK** (Python library, CLI runners) | `/nemo/microservices/latest/evaluator/` → Evaluator-as-microservice (REST API, job mgmt) |
| `/nemo/guardrails/` → **Guardrails OSS toolkit** (Colang, Python lib) | `/nemo/microservices/latest/guardrails/` → Guardrails-as-microservice (managed configs, REST API) |
| `/nemo/automodel/latest/` → AutoModel SDK | (no microservices equivalent) |

For a **platform-operator** SFT adapter, only the microservices subdirs are
relevant. The standalone OSS portals teach "how to use the library in your
app" — a different audience. By scoping to a single `/microservices/latest`
prefix, we automatically include the right view and exclude the OSS view.

For a **developer-using-OSS-tools** SFT adapter, the choice would be reversed:
include the standalone portals, exclude the microservices subdirs.

## Lessons for other broad documentation umbrellas

1. **Component names overload across portals.** "Foo Evaluator" and "Foo
   Evaluator Microservice" are often distinct source areas with overlapping docs.
   Inspect index pages before assuming dedup will save you.
2. **Sitemap structure is not navigation structure.** A flat sitemap can
   hide important sub-divisions. Always do a `awk -F'/<umbrella>/' '{print $2}'
   | awk -F'/' '{print $1}' | sort | uniq -c` pass to discover sub-areas
   before curating.
3. **Documentation splits happen.** NeMo Framework (the monolithic doc) was
   split into ~6 separate component portals in version 26.02. Some component
   portals have no sitemap; only the framework's index page links to them.
   If you only inspect the sitemap, you miss them. A `WebFetch` against the
   framework's index page surfaces these links.
4. **Two URL prefixes for one domain area.** `docs.nvidia.com/nemo/...`
   and `docs.nvidia.com/nemo-framework/...` are both part of "NeMo" but
   live at different paths. Easy to miss if you assume one domain area = one
   path prefix.

## Stage 2 dataset

Built via the Stage 2 pipeline against the `nemo_usvcs_curated` ES collection
(1,289 chunks → 486 passages after Stage 0 grouping + noise filter).

**Runtime**: ~12h 10min end-to-end on the in-cluster super-120b NIM.

### Yield through each stage

| Stage | KVPs |
|---|---:|
| Stage 1A (LE → KVP, multi-entailment) | ~4,790 |
| Stage 1B (kNN bridging + contrastive) | ~890 |
| Stage 1C (instruction diversity) | ~330 |
| Stage 1.5 (gap-fill) | 0 (no under-represented domain slices flagged) |
| **Pre-Curator total** | **6,009** |
| After Stage 2 QA-eval refinement | 5,721 (288 dropped as ungrounded) |
| After exact dedup | 5,017 (-704 — high repetition across endpoint pages) |
| After MinHash fuzzy dedup | 4,956 |
| After length filter (Q ≥ 8 tok, A ≥ 25 tok) | 4,636 |
| After answer-subset-of-question filter | **4,627** |

The large exact-dedup drop (12%, vs ~1% for NIM) reflects the NeMo Microservices
docs' repeated setup/auth sections across multiple endpoint pages — the synthetic
Q&A pairs converged on similar wording for those repeated steps.

### Train / validation split

- **Training**: 4,162 pairs
- **Validation**: 465 pairs
- Ratio: 89.95 / 10.05 (target 90 / 10)

### Top domain-area metadata by KVP count

| Family | KVPs |
|---|---:|
| `NeMo Microservices` | 5,696 |
| `unknown` | 25 |

Single-scope crawl: every NeMo Microservices doc URL maps to the
one broad source-area label in `CRAWLER_PRODUCT_URL_MAP`. The 25 `unknown`
chunks are URLs that fell outside the crawler's prefix map.

### Validation gate (independent external judge, 100-pair stratified sample)

| Criterion | Pass |
|---|---:|
| Grounded | 98 / 100 |
| Answer fidelity | **100 / 100** (perfect) |
| No hallucination | 97 / 100 |
| **All three** | **95 / 100 = 95.0%** ✅ |
| Threshold | 90% |

**Verdict**: PASSED. 5 pairs flagged for spot-check.
