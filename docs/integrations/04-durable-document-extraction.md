# Durable Document Extraction

Use this path when the raw corpus is file-based rather than crawl-first: PDF archives, DOCX exports, policy manuals, SOP libraries, field-service binders, research packs, knowledge-base exports, or any source where documents arrive as files and need reliable extraction before dataset generation.

The recommended external reference implementation is [`nv-ingest-265-durable-orchestration`](https://github.com/joncoons/nv-ingest-265-durable-orchestration). It is the NVAIE/NV-Ingest-oriented companion path for making raw-document extraction durable enough for enterprise use. Keep it as a sibling ingestion component rather than a hard dependency of this repo. Its job is to produce durable, auditable chunks; this repo's job starts once those chunks are available as a scoped collection.

## Contract With This Repo

A durable extraction path should emit chunks with enough metadata for downstream dataset creation and evaluation:

- stable `source_id` or document ID
- source URI or original file path
- source kind such as `pdf`, `docx`, `pptx`, `html`, `markdown`, or `export`
- page, section, or element location when available
- chunk index and chunk text
- parser/extractor version
- extraction timestamp or source snapshot ID
- checksum or content hash for the raw source when feasible
- retry/failure records for documents or pages that did not parse cleanly

After extraction, embed the chunks and load them into the same vector-store shape used by the curated crawl path. Stage 2 can then operate on the collection without caring whether the original material came from HTML, PDFs, DOCX files, Markdown, or an internal export.

## Why Keep This Separate

PDF and raw-document extraction have different operational failure modes than web crawling:

- parsers can fail per page or per element
- tables and figures can span pages
- scanned pages may require OCR
- large files need checkpointing and resumable processing
- document batches often need retry queues and provenance sidecars

Keeping durable extraction as a sibling ingestion implementation lets this repo stay focused on dataset generation, LoRA training, and evaluation while still showing how NVAIE building blocks can be extended with reliability, retry, and provenance controls for non-web corpora.

## When To Use It

Use this path when:

- the source corpus is mostly PDFs or office documents
- the source is offline or behind enterprise access controls
- crawlable HTML is incomplete compared with the document archive
- extraction failures need to be replayed without rerunning the whole batch
- provenance must identify exact document/page/element origins

Skip it when a documentation site is already clean, crawlable, and has a sitemap. In that case, the curated crawl path is simpler and faster.

## Relationship To Optional Chunking Modules

Durable extraction produces raw elements or text chunks. The other Stage 1 optional integrations can still be applied after extraction:

- semantic chunking for paragraph-aware chunk boundaries
- cross-page text stitching for sentences split across PDF page boundaries
- visual stitching and routing for tables or figures split across pages

For PDF-heavy corpora, the natural order is durable extraction, visual/text repair where needed, semantic chunking, embedding, then Stage 2 dataset generation.
