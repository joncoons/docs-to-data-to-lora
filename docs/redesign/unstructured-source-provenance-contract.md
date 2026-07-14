# Unstructured Source Provenance Contract

This contract keeps the dataset pipeline source-agnostic. Webcrawl is one
ingestion adapter, but the same ES and provenance fields must also support
document capture/OCR, dense image captioning, audio transcription, and video
summarization text.

## Common ES Fields

Every text-bearing ES chunk should keep the existing RAG-compatible shape:

```text
text
vector
metadata.source
metadata.content_metadata
```

The integration adds a `metadata.provenance` object and duplicates the most
important join keys into `metadata.source` and `metadata.content_metadata` for
query and aggregation ergonomics.

Required common fields:

| Field | Meaning |
|---|---|
| `metadata.content_metadata.provenance_schema_version` | Contract version for common source provenance |
| `metadata.content_metadata.ingestion_run_id` | Acquisition/processing run ID, source-agnostic |
| `metadata.content_metadata.source_revision_id` | Immutable observed source revision |
| `metadata.content_metadata.source_chunk_id` | Stable text chunk ID derived from source revision and text hash |
| `metadata.content_metadata.source_system` | Adapter or producer, such as `web_crawl`, `document_capture`, `image_dense_caption`, `video_summary` |
| `metadata.content_metadata.source_kind` | Producer-specific kind, such as `web_page`, `inline_text`, `downloaded_asset`, `dense_caption`, `video_summary_text` |
| `metadata.content_metadata.modality` | `text`, `document`, `image`, `audio`, or `video` |
| `metadata.content_metadata.canonical_uri` | Durable source URI or asset URI |
| `metadata.content_metadata.final_uri` | Final source URI after redirects or storage indirection |
| `metadata.content_metadata.raw_sha256` | Hash of observed source bytes/content when available |
| `metadata.content_metadata.text_sha256` | Hash of the actual text chunk inserted into ES |
| `metadata.content_metadata.retrieved_at` | Source observation timestamp |

Optional fields should be present when available: `http_status_code`,
`http_etag`, `http_last_modified`, `content_type`, `parser_version`,
`chunker_version`, page number, bounding box, timestamp range, frame range,
heading path, and product metadata.

## Source Adapter Mapping

| Source | `source_system` | Expected anchors | Required transform metadata |
|---|---|---|---|
| Web HTML | `web_crawl` | heading path, H1, crawl depth | HTML parser/chunker version, HTTP metadata |
| Inline docs | `web_crawl` | heading path, crawl depth | Markdown/RST/TXT parser version |
| Captured documents | `document_capture` | page number, bbox, section path | OCR/parser model, extraction method, parser version |
| Dense image captions | `image_dense_caption` | bbox or image region ID | caption model, prompt hash/config hash |
| Audio/video summaries | `video_summary` or `audio_transcript` | timestamp range, frame range | ASR/summarizer model, prompt hash/config hash |

## Current Implementation

The editable webcrawler copy lives at `external/rag-crawler`. Its ES writer now
stamps source-agnostic provenance for any text chunk produced through
`crawler.es_client._build_doc`. The web crawler path supplies web-specific
HTTP, registry, and crawl metadata, but the ES writer also accepts non-web
metadata for future capture/captioning/summarization adapters.

Stage 0 now consumes the ES provenance contract directly. It prefers upstream
`source_revision_id` and `source_chunk_id` values from `metadata.provenance`,
`metadata.content_metadata`, or `metadata.source` before deriving local fallback
IDs. When a retained passage maps to one ES chunk, Stage 0 keeps the upstream
`source_chunk_id`; when web HTML chunks are grouped into a larger passage, Stage
0 emits a deterministic aggregate chunk ID and preserves all upstream chunk IDs
and provenance records in `source_chunks.jsonl` metadata.

The current crawler URL registry remains a webcrawl adapter artifact. It
normalizes legacy bare hashes to `sha256:<hex>` and records common source
revision fields where available, but downstream stages should treat ES
`metadata.provenance`, `source_revision_id`, and `source_chunk_id` as the
durable cross-source join points. The registry is now fallback/enrichment for
webcrawl HTTP metadata, historical hashes, and registry coverage metrics.

The legacy `Passage.doc_kind` field still only allows `html` or `pdf`; Stage 0
uses the `pdf` value as the per-chunk grouping mode for captured documents,
dense image captions, audio transcripts, and video summaries until the passage
model grows a source-agnostic kind field.
