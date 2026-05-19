# Stage 1: Curated Crawl

A small-but-clean documentation corpus is more useful for SFT training than a
large noisy one. This stage covers how to turn a vendor's documentation site
into a focused training corpus by curating *which* pages get ingested instead
of relying on a broad crawl.

## Why curate

| Broad crawl | Curated crawl |
|---|---|
| BFS from a seed URL, scope by host or path prefix | Explicit per-product URL prefix list |
| Includes every version of every product page | One version per product (`/latest/` or highest-pinned) |
| Same content ingested N times across versions | Each canonical page once |
| Crawl time scales with link graph | Crawl time scales with curated list |
| Retrieval has to disambiguate near-duplicates | Cleaner retrieval, cleaner SFT signal |

For SFT specifically, version duplicates inject contradictions into
ground-truth extraction ("In v1.2 X, in v1.5 Y — which is true?") that the
LoRA may memorize incorrectly. `/latest/`-only is therefore not just a
nice-to-have; it's a quality requirement.

## Methodology

A four-step process that works for any docs site with a published sitemap.

### Step 1: Find the authoritative sitemap

Most vendor docs sites publish at least one sitemap. Discovery order:

1. **`robots.txt`** at the site root — typically lists all sitemaps the
   vendor wants crawlers to find. `curl -s https://<host>/robots.txt | grep -i sitemap`.
2. **Conventional locations** if `robots.txt` is silent: `/sitemap.xml`,
   `/sitemap_index.xml`, `/s3-sitemap-index.xml`.
3. **Sitemap index files** that aggregate per-product sitemaps —
   `grep -oE 'https://[^<]+sitemap[^<]+' <index>.xml | grep -i <product>`.

The sitemap is the vendor's own statement of "which pages should be indexed."
It is orders of magnitude faster to parse than a Selenium-rendered BFS that
re-discovers the same content, and it surfaces pages BFS would miss (pages
with no inbound link from the seed).

### Step 2: Inventory the URL list

Parse the sitemap into a CSV with `url, product, version, page, path_depth,
lastmod` columns. The provided script does this:

```bash
python scripts/sitemap_to_inventory.py \
  --sitemap https://docs.example.com/foo/sitemap.xml \
  --path-prefix /foo/ \
  --output inventory.csv
```

The classifier splits each URL on the first segment that looks like a version
(`/\d+\.\d+/` or `latest|stable|dev|main`). Everything before is the product
path; the segment itself is the version; everything after is the page.

### Step 3: Curate per-product versions

For each distinct product:

- If `/latest/` exists, use it.
- If `/latest/` is missing or appears to be a stub (very few URLs vs. older
  versions — see Anomalies below), pick the highest non-deprecated version.
- If a product only has versioned directories, pick the newest semver.

Output: one prefix per product, where each prefix is everything up to and
including the version segment. Example:

```
https://docs.example.com/foo/widget-a/latest
https://docs.example.com/foo/widget-b/1.4.0   # no /latest/, pinned to highest
https://docs.example.com/foo/widget-c/latest
```

### Step 4: Run the crawl with the curated prefix list

Most BFS crawlers accept a list of allowed URL prefixes. With the curated
list as the allowlist, `max_depth` becomes irrelevant — the crawler can't
escape the allowlist regardless of depth.

```jsonc
// example shape, varies by crawler
{
  "start_url": "https://docs.example.com/foo/",
  "max_depth": null,
  "allowed_url_prefixes": [ ... per-product /latest/ or pinned prefixes ... ],
  "extract_linked_files": true,                            // capture linked binaries
  "binary_extensions": [".pdf", ".docx", ".pptx", ".txt"], // see scope table below
  "binary_host_allowlist": ["docs.example.com", "images.example.com"]
}
```

### Step 5: Capture linked files (binaries + inline text)

HTML pages routinely link to files containing technical content not
duplicated in the HTML — whitepapers, deployment guides, datasheets,
architecture decks, plain-text READMEs on GitHub. For an SFT corpus
covering "everything the vendor has documented about product X," you want
these too.

**Two paths, depending on file type:**

1. **Inline-text path** for `.txt`, `.md`, `.rst`. The crawler fetches the
   file during the HTML crawl, runs it through the same element-based
   chunker as HTML, embeds inline, and writes to the collection. No
   separate parse phase, no GPU parser required.
2. **Binary parse path** for `.pdf`, `.docx`, `.pptx`. The crawler records
   the binary into a manifest CSV (URL, source page, filename,
   content-type) during the HTML crawl and downloads it. A separate
   Phase 3 step reads the manifest, parses each binary via a
   document-extraction model (e.g., nemotron-parse), and ingests the
   resulting chunks into the same collection.

The split exists because PDF/DOCX/PPTX parsing requires different compute
(a parse NIM, several GB of GPU each) than HTML or plain-text chunking.
Running them sequentially lets the GPU profile adapt to the workload.

#### File-type scope

| File type | Path | In default scope | Notes |
|---|---|---|---|
| `.pdf` | binary parse (Phase 3) | yes | Most vendor whitepapers and deployment guides |
| `.docx` | binary parse (Phase 3) | yes | Less common but worth capturing when present |
| `.pptx` | binary parse (Phase 3) | yes | Architecture decks, technical webinar slides. Parser support varies by tool — nemotron-parse handles `.pptx` partially as of May 2026; confirm before relying on it |
| `.txt` | inline text (HTML crawl) | yes | READMEs, plain config samples, license texts |
| `.md` | inline text (HTML crawl) | yes | GitHub-hosted READMEs, common in vendor docs links |
| `.rst` | inline text (HTML crawl) | yes | Sphinx source occasionally linked directly |
| `.zip`, `.tar.gz` | — | **no** | Code archives; ingest the docs *about* them, not the code itself |
| `.json`, `.yaml` | — | **no** | Config/schema files; usually noise for SFT |
| `.iso`, `.img` | — | **no** | OS/container images |

#### Host scope for linked files

Binaries and linked text often live outside the docs host:

- **Vendor CDNs** for PDFs (e.g., `images.<vendor>.com`,
  `developer.download.<vendor>.com`). NVIDIA, for example, ships many PDFs
  from `images.nvidia.com` rather than `docs.nvidia.com`.
- **GitHub** for READMEs and source-of-truth Markdown docs that vendor
  HTML pages link out to. Most BFS crawlers block `github.com` and
  `gitlab.com` by default (they're trip-wires for unbounded crawl scope);
  you need to explicitly override the block.

Default to a small allowlist per crawl:

```
docs.<vendor>.com
images.<vendor>.com
developer.download.<vendor>.com
github.com/<vendor-org>/                    ← e.g., github.com/NVIDIA/
raw.githubusercontent.com/<vendor-org>/     ← raw file content lives here
```

Avoid wildcards — `*.github.com` or any-github allowlist pulls in
community examples, partner repos, third-party tutorials. Restrict to the
vendor's GitHub organization only.

#### GitHub URL handling

GitHub links in vendor docs typically point at `github.com/<org>/<repo>/blob/<branch>/<path>.md`,
which returns the *rendered* HTML view (with navigation chrome) rather
than the raw Markdown. To get clean inline-text content, the crawler must
either:

- Rewrite `github.com/.../blob/.../X.md` → `raw.githubusercontent.com/.../X.md`
  before fetching, or
- Follow the link, detect the GitHub HTML wrapper, and strip it to find
  the raw text.

The reference rag-crawler handles this via URL rewriting; verify your
crawler does too before relying on GitHub content.

#### Per-page binary caps

No cap by default — capture everything linked from in-scope pages. Override
if you find one page (e.g., a release-notes index) linking to 100+ PDFs;
cap at ~20 per page to keep one outlier from dominating the corpus.

#### What you can't filter at capture time

Files that are technically in-scope (correct extension, correct host) but
low-quality for SFT:

- One-page datasheets with mostly tables and product photos
- Pre-rendered slide decks with sparse prose and heavy branding
- Generated API reference dumps converted to PDF
- Marketing whitepapers with little technical content
- GitHub READMEs that are stubs or boilerplate

Filter these at the entailment-extraction step in
[Stage 2](stage-2-dataset-creation.md), not during the crawl. Easier to
inspect rejected files (their parsed text is in your collection) than to
re-crawl.

## Anomalies to watch for

These show up often enough that they're worth checking before committing to a
prefix list. All four were observed in the NIM and NeMo case studies (see
`examples/`).

1. **`/latest/` may be a stub.** Some products migrate to a Fern-style
   docs portal and leave `/latest/` as a minimal landing page while older
   numbered versions retain the full doc tree. Compare URL counts at
   `/latest/` vs. the previous version. If `/latest/` has <10% of the URLs
   of the previous version, treat it as suspect.
2. **Newest version may be incomplete.** Vendors sometimes ship a new
   version dir before all docs are migrated. If the newest semver has far
   fewer URLs than the previous one, pin to the previous version.
3. **Sitemap may omit currently-served pages.** Live URLs returning 200 are
   not always in the sitemap. Common gaps: product landing pages
   (`/product/index.html`), recently-migrated portals (Fern-hosted sites
   that don't auto-publish). For these, a small follow-up BFS pass with
   `max_depth=1` from the seed will fill the gaps.
4. **Same-name product, different audience.** A "Foo SDK" portal and a
   "Foo Microservice" portal often share a name but document different
   audiences (library users vs. platform operators). Read the index pages
   before deciding which to include — they typically overlap less than the
   names suggest.

## Crawler requirements

This methodology assumes a BFS crawler that supports:

- **Sitemap-mode seeding** that respects the allowed-prefix list (so the
  sitemap pre-populates the queue, then BFS fills any gaps).
- **Literal-prefix matching** for the allowlist (regex/glob is nice-to-have
  but not required — the curated list expresses the same scope explicitly).
- **Persistent URL registry** across restarts, so a stalled crawl can resume
  without re-fetching everything.
- **Linked-file capture** with a configurable host allowlist (so the
  crawler can download files hosted on CDNs or GitHub even when
  link-following is host-restricted).
- **Inline-text handling** for `.txt`/`.md`/`.rst` (different code path
  than binary parsing — these are chunked and embedded without a separate
  parse phase).
- **Separate binary parse phase** keyed off the manifest the HTML crawl
  produces, for `.pdf`/`.docx`/`.pptx`.
- **GitHub URL rewriting** (or equivalent) so that
  `github.com/<org>/<repo>/blob/<branch>/X.md` is fetched as the raw text,
  not the rendered HTML wrapper.

The reference implementation used in this repo is `rag-crawler`
(https://github.com/joncoons/rag-crawler — check the deployment/ guide).
Any equivalent crawler with the same primitives will work.

## Filter at extraction, not at crawl

Don't try to filter low-quality pages at crawl time. Crawl the full curated
list, then drop weak chunks at the entailment-extraction step
([stage 2](stage-2-dataset-creation.md)). It's easier to inspect rejections
than to re-crawl. Typical filters to apply *after* the crawl, not before:

- Pages with `<main>` text length under ~500 characters
- Paths matching `_static|genindex|webpack|404\.html`
- Auto-generated API stubs with only signature + type info
- Multi-product comparison tables (entailment claims span products in ways
  that confuse single-product adapter training)

## Pre-flight checklist

Before running the crawl:

1. Confirm the target collection name does not collide with any existing
   index in your vector store.
2. Verify the embedding model used by the crawler matches the dimensions
   you plan to use in stage 2 (the SFT pipeline will retrieve from this
   collection; mismatched dimensions silently produce garbage similarity).
3. Sample 5–10 pages from your curated prefix list in a browser. Read them.
   If they don't look like the content you want to train on, your curation
   logic is wrong; fix it before running a 500-page crawl.
4. Record the crawl-start timestamp. The `/latest/` snapshot is the
   ground-truth of record for the SFT dataset; if the vendor refreshes docs
   mid-training, you need to know which snapshot trained which adapter.

## Post-crawl verification

After the crawl completes:

1. Count URLs in the resulting collection. Compare against the curated
   prefix list's expected size. Significant under-count usually means a
   prefix mismatch (e.g., trailing slash mismatch); over-count means the
   prefix matcher accepted unintended URLs.
2. Sample 10–20 random chunks. They should all be informative prose, not
   navigation boilerplate or empty stubs.
3. Inspect the crawler's error CSV for systematic failures (a single page
   timing out is normal; an entire product directory failing means a
   crawler/network issue worth investigating).

## Worked examples

The two case studies in `examples/` apply this methodology end-to-end:

- [`examples/nim.md`](../examples/nim.md) — 50-product NVIDIA Inference
  Microservices catalog with mixed `/latest/` and pinned-version curation.
- [`examples/nemo-microservices.md`](../examples/nemo-microservices.md) —
  Narrowing a sprawling product umbrella to a single-prefix scope.

## Optional integrations

Three drop-in modules that improve corpus quality for documentation-heavy
PDFs and long technical guides. None are required for the methodology
above to work; they're enhancements for cases where naive chunking and
per-page parsing fall short.

- [`integrations/01-semantic-chunking.md`](integrations/01-semantic-chunking.md) —
  Element-aware chunker for HTML, Markdown, RST, plain text, and
  PDF-parser output. Conditional overlap only when single elements
  exceed `max_tokens`; tables and formulas stay atomic.
- [`integrations/02-cross-page-text-stitching.md`](integrations/02-cross-page-text-stitching.md) —
  Detects sentences that span PDF page boundaries and stitches them back
  together before chunking. ~50 lines, parser-agnostic.
- [`integrations/03-visual-stitching-and-routing.md`](integrations/03-visual-stitching-and-routing.md) —
  Detects tables/figures split across PDF pages, re-rasterizes both
  pages as one image, re-submits to Nemotron-Parse v1.2 to reassemble
  the structure. Includes a routing decision: skip the expensive parser
  entirely for documents with no complex elements.

See the [integrations README](integrations/README.md) for a decision
table on when to use which module.

## Next stage

Once the curated collection exists in your vector store, proceed to
[Stage 2: Dataset Creation](stage-2-dataset-creation.md).
