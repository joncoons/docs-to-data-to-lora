# Module 03 — Visual Stitching + Classifier Routing

The largest of the three modules. Orchestrates a three-phase PDF parse
pipeline:

1. **Rasterize + extract** each page in parallel via Nemotron-Parse v1.2
2. **Visually re-stitch** page pairs where an atomic structure (table,
   formula) spans the boundary — re-rasterize, vertically composite, and
   re-parse the combined image
3. **Text-stitch + chunk** using [Module 02](02-cross-page-text-stitching.md)
   and [Module 01](01-semantic-chunking.md)

Plus a routing decision: if the document contains no complex elements
(no tables, no pictures), bypass Nemotron-Parse entirely and let a simpler
upstream parser handle it. The complex-element path is GPU-expensive;
skipping it for simple text documents saves significant compute.

This module **does not do row-level table matching across pages**. The
table-reassembly work is delegated to Nemotron-Parse's vision model by
re-rendering the two adjacent pages as one image. This is the right
trade-off for tables that span 2 pages; it does not generalize to tables
that span 10+ pages.

## When to use

You need all of:

- PDFs with tables, figures, or formulas that frequently split across
  pages.
- Access to a Nemotron-Parse v1.2 endpoint (or another vision-language
  document parser that produces element-class-tagged output and accepts
  arbitrary rendered images).
- A few seconds per PDF page of GPU compute budget.

If any of those is missing, use a lighter parser (PyMuPDF, pdfplumber)
and Module 02 alone for text continuation. You'll lose the visual table
reassembly, but the text path still works.

## The pipeline

```
                  ┌───────────────────────────────────────────────┐
                  │ PDF (filepath, page_count)                    │
                  └─────────────────┬─────────────────────────────┘
                                    │
                  ┌─────────────────▼─────────────────────────────┐
                  │ Phase 1: rasterize + parallel parse           │
                  │   pdf2image at 300 DPI, batch_size pages      │
                  │   ThreadPoolExecutor(max_parallel_pages)      │
                  │   each page → POST Nemotron-Parse → elements  │
                  │   → list[list[(class, text)]]  (per-page)     │
                  └─────────────────┬─────────────────────────────┘
                                    │
                  ┌─────────────────▼─────────────────────────────┐
                  │ Phase 2: visual re-stitch                     │
                  │   detect page pairs where (last on page N,    │
                  │     first on page N+1) are same atomic class  │
                  │   for each: rasterize both at half-DPI,       │
                  │     vertically composite, re-call Nemotron-   │
                  │     Parse on the combined image               │
                  │   replace page N's elements with unified out, │
                  │     page N+1 elements → []  (consumed)        │
                  └─────────────────┬─────────────────────────────┘
                                    │
                  ┌─────────────────▼─────────────────────────────┐
                  │ Routing decision                              │
                  │   detected = {class} ∩ {table, picture}       │
                  │   if empty and not forced: return None        │
                  │     → caller falls back to lighter parser     │
                  └─────────────────┬─────────────────────────────┘
                                    │  (only if complex elements present)
                  ┌─────────────────▼─────────────────────────────┐
                  │ Phase 3: text stitch + chunk                  │
                  │   Module 02: stitch_page_boundaries(...)      │
                  │   Module 01: split_by_semantic_elements(...)  │
                  │   → list[(chunk_text, section_path)]          │
                  └───────────────────────────────────────────────┘
```

## Constants

| Name | Default | Purpose |
|---|---|---|
| `_ATOMIC_CLASSES` | `{"table", "formula"}` | Classes that trigger visual re-stitch when adjacent across page boundary |
| `COMPLEX_ELEMENT_CLASSES` | `{"table", "picture"}` | Classes whose presence triggers routing to this parser (vs. lighter fallback) |
| `dpi` | 300 | Rasterization DPI for the initial per-page pass |
| `stitch_dpi` | `max(150, dpi // 2)` | Lower DPI for the two-page composite (keeps memory manageable) |
| `page_batch_size` | 32 | Number of pages rasterized per batch (memory bound; tune for your host RAM) |
| `max_parallel_pages` | 8 | ThreadPoolExecutor workers per document |
| `max_parallel_docs` | 4 | Concurrent documents through the orchestrator |
| `parse_max_tokens` | 8990 | `max_tokens` on each Nemotron-Parse call |
| `max_tokens` | 2048 | Semantic chunker ceiling (passed to Module 01) |

The parallelism defaults assume the Nemotron-Parse backend can handle
`max_parallel_docs × max_parallel_pages` = 32 concurrent requests
(e.g., 7 replicas × ~4 in-flight each). Size for your replica count.

## Nemotron-Parse v1.2 HTTP shape

The endpoint accepts OpenAI-style chat-completions with a base64 image
in the message content:

```python
def call_nemo_parse(
    endpoint_url: str,
    model_name: str,
    api_key: str,
    image_b64: str,
    image_mime: str,
    extraction_prompt: str,
    max_tokens: int = 8990,
) -> str:
    """Issue one Nemotron-Parse v1.2 request; return raw model output text."""
    import requests
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": extraction_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "repetition_penalty": 1.1,
        "top_k": 1,
        "skip_special_tokens": False,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
```

`temperature=0.0`, `top_k=1`, `repetition_penalty=1.1` are tuned for
deterministic structured output. Don't change these without understanding
the parser's behavior — higher temperatures produce element-ordering
variation that breaks downstream stitching.

The response content is a markdown-ish string with element tags. The
local fork's `_parse_model_output_elements` parser converts that into
`(class_name, text)` tuples; consult the canonical implementation for
the exact tag format (it follows Nemotron-Parse's published schema).

## Detecting split boundaries

```python
_ATOMIC_CLASSES = frozenset({"table", "formula"})


def detect_split_boundaries(
    page_element_lists: list[list[tuple[str, str]]],
) -> list[int]:
    """Return page indices (0-based) where an atomic element spans the boundary.

    Index i is returned when the last element on page i and the first
    element on page i+1 share the same atomic class — indicating the
    structure was split across the page break.
    """
    boundaries: list[int] = []
    for i in range(len(page_element_lists) - 1):
        cur = page_element_lists[i]
        nxt = page_element_lists[i + 1]
        if not cur or not nxt:
            continue
        last_cls = cur[-1][0].lower()
        first_cls = nxt[0][0].lower()
        if last_cls in _ATOMIC_CLASSES and last_cls == first_cls:
            boundaries.append(i)
    return boundaries
```

Tables and formulas are the only candidates because they're the only
element classes where a structural break is recoverable by re-parsing
the visual. A "text" split is handled by [Module 02](02-cross-page-text-stitching.md);
no re-parse needed.

## Vertical image composition

```python
from PIL import Image


def stitch_page_images(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    """Concatenate two page images vertically into one tall image."""
    width = max(img_a.width, img_b.width)
    total_height = img_a.height + img_b.height
    out = Image.new("RGB", (width, total_height), "white")
    out.paste(img_a.convert("RGB"), (0, 0))
    out.paste(img_b.convert("RGB"), (0, img_a.height))
    return out
```

Simple enough that any image-manipulation library will work; PIL/Pillow
is the lowest-friction default. The white background prevents transparency
artifacts if either source page has an alpha channel.

## Visual re-stitch orchestrator

```python
def repair_visual_splits(
    filepath: str,
    page_element_lists: list[list[tuple[str, str]]],
    *,
    initial_dpi: int,
    parse_max_tokens: int,
    endpoint_url: str,
    model_name: str,
    api_key: str,
    extraction_prompt: str,
    parse_model_output_elements,  # callable: raw_text -> [(class, text), ...]
) -> list[list[tuple[str, str]]]:
    """Re-render split page pairs and replace with unified element lists.

    For each detected split boundary, rasterizes the two adjacent pages at
    half DPI, stitches vertically, re-submits to Nemotron-Parse. The
    unified output replaces page N's element list; page N+1 is set to []
    so downstream stitching skips it cleanly.
    """
    from pdf2image import convert_from_path
    import base64
    from io import BytesIO

    boundaries = detect_split_boundaries(page_element_lists)
    if not boundaries:
        return page_element_lists

    stitch_dpi = max(150, initial_dpi // 2)
    result = list(page_element_lists)
    consumed: set[int] = set()

    for i in boundaries:
        if i in consumed or (i + 1) in consumed:
            continue
        try:
            pages = convert_from_path(
                filepath, dpi=stitch_dpi,
                first_page=i + 1, last_page=i + 2,
            )
        except Exception:
            continue
        if len(pages) < 2:
            continue

        stitched_img = stitch_page_images(pages[0], pages[1])

        buf = BytesIO()
        stitched_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        try:
            raw = call_nemo_parse(
                endpoint_url, model_name, api_key,
                image_b64=b64, image_mime="image/png",
                extraction_prompt=extraction_prompt,
                max_tokens=parse_max_tokens,
            )
        except Exception:
            continue

        unified_elements = parse_model_output_elements(raw)
        result[i] = unified_elements
        result[i + 1] = []
        consumed.add(i)
        consumed.add(i + 1)

    return result
```

Important details from the canonical implementation:

- **`consumed` set** prevents the same page from being part of two
  separate re-stitches. If pages 5–6 stitched, page 6 is locked from
  participating in a 6–7 stitch even if one was detected.
- **Half-DPI for the composite** keeps memory bounded. A 300-DPI letter
  page is ~9 MB as an RGB array; doubling it to 18 MB and re-parsing is
  fine, but quadrupling would matter at scale.
- **Empty list on the consumed page** is the signal to Module 02's
  stitcher that it has nothing to merge there.

## Routing decision

```python
COMPLEX_ELEMENT_CLASSES = frozenset({"table", "picture"})


def should_route_to_parser(
    all_elements: list[tuple[str, str]],
    *,
    force: bool = False,
) -> bool:
    """Return True if the document warrants this expensive parser path.

    If the document has no tables and no pictures, a lighter text-only
    parser will produce equivalent output at a fraction of the cost.
    `force=True` bypasses the check.
    """
    if force:
        return True
    classes = {cls.lower() for cls, _ in all_elements}
    return bool(classes & COMPLEX_ELEMENT_CLASSES)
```

In the canonical implementation, returning `False` causes
`route_document()` to return `None`, signalling the caller to fall back
to standard NV-Ingest. Adapt this to your pipeline's fallback semantics
(returning an `Optional` is one option; an exception, or a sentinel value,
are others).

## End-to-end orchestrator

```python
def route_document(
    filepath: str,
    *,
    endpoint_url: str,
    model_name: str,
    api_key: str,
    extraction_prompt: str,
    parse_model_output_elements,
    dpi: int = 300,
    parse_max_tokens: int = 8990,
    max_tokens: int = 2048,
    chunk_overlap: int = 150,
    max_parallel_pages: int = 8,
    page_batch_size: int = 32,
    force: bool = False,
) -> list[tuple[str, str]] | None:
    """Three-phase orchestrator. Returns chunk pairs or None if not routed."""
    from pdf2image import convert_from_path
    from concurrent.futures import ThreadPoolExecutor

    # ---- Phase 1: rasterize + extract per-page ------------------------
    page_count = _get_page_count(filepath)
    all_page_elements: list[list[tuple[str, str]]] = []

    for batch_start in range(0, page_count, page_batch_size):
        batch_end = min(batch_start + page_batch_size, page_count)
        pages = convert_from_path(
            filepath, dpi=dpi,
            first_page=batch_start + 1, last_page=batch_end,
        )
        with ThreadPoolExecutor(max_workers=max_parallel_pages) as pool:
            results = list(pool.map(
                lambda p: _process_page(p, endpoint_url, model_name, api_key,
                                        extraction_prompt, parse_max_tokens,
                                        parse_model_output_elements),
                pages,
            ))
        all_page_elements.extend(results)

    # ---- Phase 2: visual re-stitch ------------------------------------
    all_page_elements = repair_visual_splits(
        filepath, all_page_elements,
        initial_dpi=dpi,
        parse_max_tokens=parse_max_tokens,
        endpoint_url=endpoint_url,
        model_name=model_name,
        api_key=api_key,
        extraction_prompt=extraction_prompt,
        parse_model_output_elements=parse_model_output_elements,
    )

    # ---- Phase 3: text-level stitch + chunk ---------------------------
    flat = stitch_page_boundaries(all_page_elements)  # from Module 02

    if not should_route_to_parser(flat, force=force):
        return None  # caller falls back to lighter parser

    return split_by_semantic_elements(  # from Module 01
        flat, max_tokens=max_tokens, chunk_overlap=chunk_overlap,
    )
```

## Operational notes

- **GPU memory.** Nemotron-Parse v1.2 needs ~20 GB per replica. For
  long documents (50+ pages), Phase 1 saturates the parser quickly. Size
  replicas accordingly.
- **Thread safety.** `requests.Session` is not thread-safe for
  concurrent posts. The canonical implementation uses `threading.local`
  to give each worker thread its own session. If you use `httpx` instead,
  `AsyncClient` handles this automatically.
- **`Connection: close` per request.** The canonical implementation
  forces a new TCP connection per request to encourage the upstream load
  balancer to round-robin across replicas. Without this, an HTTP/2
  client can pin all requests to a single replica.
- **Failure isolation.** Per-page exceptions degrade to empty element
  lists rather than crashing the whole document. Track them via logs;
  don't silently absorb failures across many documents or you'll get
  systematic gaps in your corpus.

## What this does NOT do

- **Row-level table matching across more than two pages.** A table that
  spans pages 3, 4, and 5 will be visually re-stitched as 3–4 (or
  consumed by 3–4 and 4–5 conflict if both are atomic-spanning
  boundaries), but the resulting markdown may still have an artifact
  between the 4–5 portion. For long tables, a separate structural
  extraction pipeline is required.
- **Cross-page OCR for scanned PDFs.** This module assumes Nemotron-Parse
  produces text from the rendered image. For scanned (image-only) PDFs,
  Nemotron-Parse's OCR is competent but not its specialty — a dedicated
  OCR pipeline (Tesseract, PaddleOCR) upstream of this module may be
  more reliable.
- **Footnote re-association.** Footnotes split across pages are detected
  as their own class and don't re-stitch. The chunker (Module 01) emits
  them as `> blockquote` units in semantic order, which is usually
  sufficient for retrieval but won't reconstruct numbered footnote
  references.
- **Per-document content hash deduplication.** Two PDFs containing the
  same content will produce two independent chunk sets. Handle dedup
  upstream of this module (the crawler) or downstream (your vector
  store's bulk-write logic).

## Origin

Extracted from `/home/joncoons/claude/rag/src/nvidia_rag/tools/parse_document.py`:

- `DocumentClassifierRouter` class — lines 285-357
- `route_document` main entry point — lines 392-540
- `_call_nemo_parse` HTTP shape — lines 600-625
- `_detect_split_boundaries` — lines 847-866
- `_stitch_page_images` — lines 868-878
- `_repair_visual_splits` — lines 880-945

Refer to the canonical implementation for the tag-parser that converts
Nemotron-Parse's raw output into `(class, text)` tuples, and for the
optional VLM picture-description path (which describes embedded figures
via a second VLM call) not documented here.
