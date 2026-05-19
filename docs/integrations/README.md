# Stage 1 Optional Integrations

Three drop-in modules that improve corpus quality for documentation-heavy
PDFs and long technical guides. None of them are required for the
methodology in [`stage-1-curated-crawl.md`](../stage-1-curated-crawl.md) to
work — they're optional enhancements for cases where your documents fight
naive chunking and per-page parsing.

## When to use which

| Module | Adds | Use it when… | Skip it when… |
|---|---|---|---|
| [01 — Semantic chunking](01-semantic-chunking.md) | Paragraph-aware chunking with overlap only at boundaries that exceed max-token | Naïve fixed-size chunking is fragmenting paragraphs mid-sentence and the resulting chunks duplicate too much content via overlap | Your text is already paragraph-bounded (Markdown READMEs) and chunk size > average paragraph length |
| [02 — Cross-page text stitching](02-cross-page-text-stitching.md) | Detects sentences/paragraphs that span PDF page boundaries and stitches them back together before chunking | You're parsing multi-page PDFs and noticing chunks that start mid-sentence (a tell-tale "Page 3 of 47" boundary artifact) | You're only ingesting HTML and Markdown (no page boundaries) |
| [03 — Visual stitching + classifier routing](03-visual-stitching-and-routing.md) | Detects tables/figures split across PDF pages, re-rasterizes both pages into one image, and re-parses via Nemotron-Parse to reconstruct the structure | Your PDFs have tables/figures that span pages and naïve per-page parsing produces broken tables | You're not using a vision-language parser like Nemotron-Parse, or your PDFs don't have multi-page tables |

Modules compose: 03 → 02 → 01 is a natural pipeline (visual stitching
produces clean per-page elements → text stitching merges adjacent
text/atomic elements → semantic chunker emits final chunks). You can also
use 01 standalone with any upstream parser.

## Implementation effort

- **01** is ~50 lines, pure Python, no GPU. Easy lift.
- **02** is ~80 lines if you bring your own parser output (a list of
  `(class_name, text)` tuples per page). Easy lift assuming your parser
  emits element classifications.
- **03** is ~400 lines plus a Nemotron-Parse endpoint (or another VLM
  parser with similar capabilities), PDF rasterization (PyMuPDF or
  pdf2image), and image compositing (Pillow). Significant lift.

## Origin

These modules were extracted from a local fork of the
[NVIDIA AI Blueprints RAG](https://github.com/NVIDIA-AI-Blueprints/rag)
project where they're tightly woven into the ingest pipeline. The docs
here present them as portable units with the fork's couplings (specific
ES schema, specific embedding model, specific config layout) abstracted
out. Refer to the canonical implementations for the full versions:

| Module | Canonical implementation |
|---|---|
| 01 | `src/nvidia_rag/storage/embed_store.py:chunk_text` |
| 02 | `src/nvidia_rag/tools/parse_document.py:DocumentClassifierRouter._stitch_page_boundaries` |
| 03 | `src/nvidia_rag/tools/parse_document.py:DocumentClassifierRouter` (class) + `_repair_visual_splits` + `_call_nemo_parse` |

## What none of these modules do

- **Row-level table matching across pages.** Module 03 reassembles tables
  visually (re-rasterize → re-parse) rather than by matching column
  headers, cell counts, or row spans. If you need structural row-matching
  logic (e.g., for tables that span 10+ pages and can't be re-rasterized
  feasibly), you'll need to build it separately.
- **Cross-document deduplication.** These modules operate within a single
  document. If the same passage appears in five different PDFs (a common
  vendor pattern), you'll need a separate dedup step downstream.
- **OCR.** All three modules assume the parser already gave you text. If
  your PDFs are scans, run OCR upstream (Tesseract, PaddleOCR, or
  Nemotron-Parse's own text-extraction path).
