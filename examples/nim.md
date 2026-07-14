# Example: NVIDIA Inference Microservices (NIM)

A worked example of [Stage 1 methodology](../docs/stage-1-curated-crawl.md)
applied to `docs.nvidia.com/nim` — NVIDIA's catalog of containerized AI
inference services.

> **Snapshot note**: numbers in this example were captured against the NIM
> sitemap as published in May 2026. The sitemap refreshes frequently (we
> observed a 1500→419 URL change overnight during this project). Treat the
> specific counts as illustrative; the methodology and resulting prefix list
> are durable.

## Step 1 — Find the sitemap

`docs.nvidia.com/robots.txt` lists per-product sitemaps. For NIM:

```
$ curl -s https://docs.nvidia.com/s3-sitemap-index.xml \
    | grep -oE 'https://[^<]+sitemap[^<]+' \
    | grep -i nim
https://docs.nvidia.com/nim/nim-s3-sitemap.xml
https://docs.nvidia.com/nim-operator/nim-operator-s3-sitemap.xml
```

The first is the NIM catalog itself. The second (`nim-operator`) is a
separate concern — exclude unless you want it as a sibling collection.

## Step 2 — Inventory the URL list

```bash
python scripts/sitemap_to_inventory.py \
    --sitemap https://docs.nvidia.com/nim/nim-s3-sitemap.xml \
    --path-prefix /nim/ \
    --output nim_inventory.csv
```

Summary output:

```
Distinct products: 50
  with /latest/:      44
  no /latest/:        6   (need pinned-version curation)
  no version segment: 0
```

50 products — broader than you might expect from a "single product family"
sitemap. Includes LLM NIMs, bioscience NIMs (bionemo/*), audio/video NIMs
(maxine/*), medical imaging (medical/*), and several others.

## Step 3 — Curate per-product versions

44 of 50 products have a `/latest/` symlink. For those, use `/latest/`
directly. The remaining 6 need manual version pinning:

| Product | Available versions | Pinned to | Rationale |
|---|---|---|---|
| `large-language-models/early-access` | (no version segment) | `early-access` | 2 URLs only; treat the path segment as a version |
| `llama-3-1-nemoguard-8b-contentsafety` | 1.0.0, 1.10.1 | `1.10.1` | newest |
| `maxine/active-speaker-detection` | 1.0.0 | `1.0.0` | only version |
| `nemo-retriever/text-embedding` | 1.7.0, 1.9.0, 1.10.0, 1.10.1, 1.11.0, 1.12.0 | `1.12.0` | newest |
| `speech` | 26.02.0 | `26.02.0` | only version |
| `visual-genai` | 1.0.0, 1.1.0, 1.3.0, 1.3.1 | `1.3.1` | newest |

## Step 4 — Curated prefix list (48 entries)

The 44 `/latest/` products + the 6 pinned versions + the `early-access`
entry. Pass this as `allowed_url_prefixes` to the crawler:

```json
[
  "https://docs.nvidia.com/nim/alchemi/alchemi-bgr/latest",
  "https://docs.nvidia.com/nim/alchemi/alchemi-bmd/latest",
  "https://docs.nvidia.com/nim/benchmarking/llm/latest",
  "https://docs.nvidia.com/nim/bionemo/alphafold2/latest",
  "https://docs.nvidia.com/nim/bionemo/alphafold2-multimer/latest",
  "https://docs.nvidia.com/nim/bionemo/boltz2/latest",
  "https://docs.nvidia.com/nim/bionemo/diffdock/latest",
  "https://docs.nvidia.com/nim/bionemo/evo2/latest",
  "https://docs.nvidia.com/nim/bionemo/genmol/latest",
  "https://docs.nvidia.com/nim/bionemo/molmim/latest",
  "https://docs.nvidia.com/nim/bionemo/msa-search/latest",
  "https://docs.nvidia.com/nim/bionemo/openfold2/latest",
  "https://docs.nvidia.com/nim/bionemo/openfold3/latest",
  "https://docs.nvidia.com/nim/bionemo/proteinmpnn/latest",
  "https://docs.nvidia.com/nim/bionemo/rfdiffusion/latest",
  "https://docs.nvidia.com/nim/cloud/gke/latest",
  "https://docs.nvidia.com/nim/cosmos/latest",
  "https://docs.nvidia.com/nim/cosmos-embed1/latest",
  "https://docs.nvidia.com/nim/digital-human/a2f-3d/latest",
  "https://docs.nvidia.com/nim/earth-2/corrdiff/latest",
  "https://docs.nvidia.com/nim/earth-2/fourcastnet/latest",
  "https://docs.nvidia.com/nim/financial-fraud-training/latest",
  "https://docs.nvidia.com/nim/ingestion/image-ocr/latest",
  "https://docs.nvidia.com/nim/ingestion/object-detection/latest",
  "https://docs.nvidia.com/nim/ingestion/table-extraction/latest",
  "https://docs.nvidia.com/nim/large-language-models/latest",
  "https://docs.nvidia.com/nim/large-language-models/early-access",
  "https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest",
  "https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest",
  "https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-multilingual-8b-v1/latest",
  "https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/1.10.1",
  "https://docs.nvidia.com/nim/maxine/active-speaker-detection/1.0.0",
  "https://docs.nvidia.com/nim/maxine/audio2face-2d/latest",
  "https://docs.nvidia.com/nim/maxine/bnr/latest",
  "https://docs.nvidia.com/nim/maxine/eye-contact/latest",
  "https://docs.nvidia.com/nim/maxine/lipsync/latest",
  "https://docs.nvidia.com/nim/maxine/relighting/latest",
  "https://docs.nvidia.com/nim/maxine/studio-voice/latest",
  "https://docs.nvidia.com/nim/maxine/synthetic-video-detector/latest",
  "https://docs.nvidia.com/nim/medical/maisi/latest",
  "https://docs.nvidia.com/nim/medical/vista3d/latest",
  "https://docs.nvidia.com/nim/multimodal-safety/latest",
  "https://docs.nvidia.com/nim/nemoguard-jailbreakdetect/latest",
  "https://docs.nvidia.com/nim/nemo-retriever/text-embedding/1.12.0",
  "https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest",
  "https://docs.nvidia.com/nim/nvclip/latest",
  "https://docs.nvidia.com/nim/physicsnemo/domino-automotive-aero/latest",
  "https://docs.nvidia.com/nim/speech/26.02.0",
  "https://docs.nvidia.com/nim/vision-language-models/latest"
]
```

Resulting corpus size when this list was first crawled: **~351 URLs**
(vs. ~1500 in the unfiltered sitemap = a ~4.3x reduction by dropping
historical versions).

## Step 5 — Crawl

Sample `POST /crawl` body (rag-crawler-compatible shape):

```json
{
  "start_url": "https://docs.nvidia.com/nim/",
  "collection_name": "nim_curated",
  "use_product_url_map": false,
  "max_depth": null,
  "use_sitemap": false,
  "max_pages": null,
  "extract_linked_files": true,
  "allowed_url_prefixes": [ ... 48 entries above ... ],
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

`use_sitemap: false` is intentional — sitemap mode would re-introduce
historical-version URLs, even when filtered through `allowed_url_prefixes`,
and pollute the queue.

`use_product_url_map: false` is also intentional. By default the crawler
consults a URL-prefix → product-collection map to route binaries (PDFs,
DOCX) into per-product directories under `pdf-repo/`. For a curated SFT
crawl, you want all binaries to land in *one* directory keyed to your
explicit collection name — so that the binary manifest and the
downloaded files stay grouped together and don't commingle with prior
crawls of overlapping URL prefixes. Setting this to `false` forces all
binaries to route to `collection_name`.

## Step 6 — Phase 3 binary parse

After the HTML crawl completes, the binary manifest CSV at
`/crawl-exports/<task-id>-binaries.csv` lists every `.pdf`/`.docx`/`.pptx`
that was downloaded but not yet parsed. Trigger Phase 3 to extract text
from these files and ingest into the same collection:

```bash
curl -sf -X POST http://<ingestor-host>:8082/batch-ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_name": "nim_curated",
    "method": "auto",
    "force_reingest": false,
    "max_files": 500,
    "max_concurrent": 4
  }'
```

`method: "auto"` routes `.pdf` to nemotron-parse and `.docx`/`.pptx` to
nv-ingest's fallback path. Monitor via `GET /status?task_id=<id>` and
expect runtime to depend on PDF page counts (typically 5-30s per PDF).

## Expected linked-file shape (NIM docs)

What kinds of files to expect from NIM-area pages, based on typical NVIDIA
docs patterns:

| Source | Typical files | Notes |
|---|---|---|
| `docs.nvidia.com/nim/...` HTML pages | inline `.md` snippets, occasional linked PDFs (deployment guides, performance whitepapers) | Most product technical content is in HTML; PDFs are supplementary |
| `images.nvidia.com/aem-dam/...` | Marketing PDFs, datasheets | Mixed quality for SFT — filter at extraction |
| `developer.download.nvidia.com/...` | SDK release notes, large technical guides | Usually high-quality for SFT |
| `github.com/NVIDIA/<repo>/...` | READMEs, contributing guides, code-sample explanations | Per-product repos vary; high-signal when present |

Exact counts will be known after the crawl completes — add to this table
post-crawl with actual binary-manifest totals.

## Corpus bias to know about

`/nim/large-language-models/latest/` alone is **~88% of the curated corpus**
(311 of 351 URLs). The remaining ~40 URLs cover all 47 other products
combined.

If your downstream SFT use case needs balanced coverage of all NIM products,
plan to **stratify at the entailment-extraction step**: cap per-product
chunk count to, say, 50× the median per-product count, then sample. Don't
try to "fix" this at crawl time — the docs are what they are.

If your use case is specifically LLM-NIM deployment knowledge, the bias is
actually correct and stratification would hurt.

## Verification

Once the crawl finishes:

```bash
# Count distinct URLs in the resulting collection (Elasticsearch example):
curl -sk -u "$ES_USER:$ES_PASS" \
  "$ES_URL/nim_curated/_search?size=0" \
  -H 'Content-Type: application/json' \
  -d '{"aggs":{"distinct_urls":{"cardinality":{"field":"content_url.keyword"}}}}'

# Expected: distinct_urls ≈ 351 (small drift OK; large drift means
# something went wrong with prefix matching or content-hash dedup).
```

## Lessons for other docs.nvidia.com product families

1. **Sitemap-listed product count is approximate.** What looks like one
   product (NIM) is actually a portfolio of 50 internal products with
   varied versioning conventions.
2. **A single product can dominate the corpus.** `large-language-models`
   here, but you'll see similar imbalances in other portfolios (e.g.,
   one flagship product with 10x the docs of its siblings).
3. **Versioning is not uniform within a single sitemap.** Mix of semver
   (`1.0.0`), CalVer (`26.02.0`), and unversioned paths (`early-access`)
   all appear in one sitemap. The classifier in `sitemap_to_inventory.py`
   handles all three.

## Stage 2 dataset

Built via the Stage 2 pipeline against the `nim_curated` ES collection
(2,086 chunks → 498 passages after Stage 0 grouping + noise filter).

**Runtime**: ~13h 14min end-to-end on the in-cluster super-120b NIM.

### Yield through each stage

| Stage | KVPs |
|---|---:|
| Stage 1A (LE → KVP, multi-entailment) | ~4,750 |
| Stage 1B (kNN bridging + contrastive) | ~840 |
| Stage 1C (instruction diversity) | ~470 |
| Stage 1.5 (gap-fill) | 0 (no under-represented products flagged) |
| **Pre-Curator total** | **6,057** |
| After Stage 2 QA-eval refinement | 5,752 (305 dropped as ungrounded) |
| After exact dedup | 5,697 |
| After MinHash fuzzy dedup | 5,695 |
| After length filter (Q ≥ 8 tok, A ≥ 25 tok) | 5,415 |
| After answer-subset-of-question filter | **5,413** |

### Train / validation split

- **Training**: 4,870 pairs
- **Validation**: 543 pairs
- Ratio: 89.97 / 10.03 (target 90 / 10)

### Top product_family by KVP count

| Family | KVPs |
|---|---:|
| `NIM` | 4,990 |
| `unknown` | 762 |

Coarse `product_family` granularity reflects the rag-crawler's
`CRAWLER_PRODUCT_URL_MAP` setting for this crawl — every NIM doc URL maps to
the single `NIM` family. Finer per-product analysis would require either
extending the URL map or a post-hoc classifier.

### Validation gate (independent external judge, 99-pair stratified sample)

| Criterion | Pass |
|---|---:|
| Grounded | 96 / 99 |
| Answer fidelity | 97 / 99 |
| No hallucination | 95 / 99 |
| **All three** | **93 / 99 = 93.9%** ✅ |
| Threshold | 90% |

**Verdict**: PASSED. 6 pairs flagged for spot-check.
