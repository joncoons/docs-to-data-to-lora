# Module 01 — Semantic Chunking

The chunking strategy used across HTML, Markdown, RST, plain-text, and
PDF (post-parser) input. Two layers:

1. **Format-specific element extraction** — converts the raw source into
   a uniform list of `(class_name, text)` tuples using a small taxonomy
   (title, section-header, text, list-item, table, formula, picture,
   caption, page-header, page-footer, toc).
2. **Element-aware chunker** — walks the element list and emits chunks
   that respect semantic boundaries (headings flush, tables stay whole,
   captions follow their parent figure), with token overlap applied only
   when a single text element would exceed `max_tokens`.

For raw text without semantic structure (a generic `.txt` file with no
markup), a simpler paragraph-based chunker is the fallback. Both are
documented below.

## Why this beats naive fixed-size chunking

Fixed-size character / token chunkers fragment paragraphs mid-sentence,
split tables across chunks, lose heading context, and waste storage on
omnipresent overlap. Element-aware chunking:

- **No mid-sentence splits.** Chunk boundaries always fall at paragraph
  or element boundaries.
- **Tables and formulas stay whole.** Atomic elements (table, formula)
  are emitted as their own chunks if oversized rather than fragmented.
- **Headings carry forward.** When a chunk overflows mid-section, the
  next chunk is seeded with the enclosing H1/H2/H3 so retrieval sees
  context.
- **Overlap only at hard boundaries.** Most chunks have zero overlap;
  only chunks created by a single oversized text element use overlap.
  This dramatically reduces storage cost and removes duplicate-retrieval
  noise.

## The element taxonomy

This is the contract between the format-specific extractors and the
chunker:

| Class | Behavior in chunker |
|---|---|
| `title`, `section-header` | Triggers a flush + becomes part of the running header context for subsequent chunks |
| `text` | Standard paragraph; concatenated up to `max_tokens` |
| `list-item` | Formatted with leading `-`; concatenated like text |
| `table` | **Atomic** — emitted as own chunk if oversized, otherwise grouped |
| `formula` | **Atomic** — wrapped in code fence if not already; otherwise like table |
| `picture` | Combined with its caption (if present) into one chunk-unit |
| `caption` | Bound to preceding picture / table / formula |
| `footnote` | Formatted as a blockquote |
| `page-header`, `page-footer`, `toc` | **Skipped** — never emitted |

The format-specific extractors (`html_to_elements`, `md_to_elements`,
`rst_to_elements`) all produce tuples conforming to this taxonomy.
Nemotron-Parse output already maps cleanly (its native classification
uses the same names).

## Element-aware chunker — example code

Portable version, no project-specific dependencies. The function takes
the element list and returns `(chunk_text, section_path)` pairs. The
section path is a human-readable breadcrumb like
`"Setup > Configuration > Auth"` — useful as chunk metadata for
retrieval filtering.

```python
"""semantic_chunker.py — element-aware chunker with conditional overlap."""

_SECTION_STARTERS = frozenset({"title", "section-header"})
_ATOMIC_CLASSES   = frozenset({"table", "formula"})
_SKIP_CLASSES     = frozenset({"page-header", "page-footer", "toc"})
_CAPTION_CLASS    = "caption"


def split_by_semantic_elements(
    elements: list[tuple[str, str]],
    max_tokens: int,
    chunk_overlap: int = 150,
) -> list[tuple[str, str]]:
    """
    Chunk a list of (class_name, text) tuples respecting semantic boundaries.

    Returns: list of (chunk_text, section_path) pairs.

    Token approximation: ~4 chars per token (sufficient for English; for
    other languages substitute a real tokenizer in the char/token math).
    """
    max_chars = max_tokens * 4
    overlap_chars = chunk_overlap * 4

    chunks: list[tuple[str, str]] = []
    parts: list[str] = []
    chars = 0
    headers: dict[int, str] = {}
    section_path = ""

    def flush() -> None:
        nonlocal parts, chars
        body = "\n\n".join(p for p in parts if p.strip()).strip()
        if body:
            chunks.append((body, section_path))
        parts = []
        chars = 0

    def add_unit(unit: str) -> None:
        nonlocal parts, chars
        sep = 2 if parts else 0
        if parts and chars + sep + len(unit) > max_chars:
            flush()
            # seed next chunk with H1-H3 header context if any
            ctx = "\n".join(headers[lvl] for lvl in sorted(headers) if lvl <= 3)
            if ctx:
                parts.append(ctx)
                chars = len(ctx)
                sep = 2
            parts.append(unit)
            chars += sep + len(unit)
        else:
            parts.append(unit)
            chars += sep + len(unit)

    def fmt(cls: str, text: str) -> str:
        if cls == "table":
            return text   # already markdown
        if cls == "formula":
            return text if text.startswith(("```", "$$")) else f"```\n{text}\n```"
        if cls == "list-item":
            lines = [
                f"- {ln}" if not ln.startswith(("- ", "* ", "• ")) else ln
                for ln in text.splitlines() if ln.strip()
            ]
            return "\n".join(lines) if lines else f"- {text}"
        if cls == "footnote":
            return f"> {text}"
        return text

    i = 0
    while i < len(elements):
        cls, text = elements[i]
        cls = cls.lower()
        text = text.strip()
        i += 1

        if cls in _SKIP_CLASSES or (not text and cls not in _ATOMIC_CLASSES):
            continue

        # Section header: flush current chunk, record header for context
        if cls in _SECTION_STARTERS:
            level = 1 if cls == "title" else 2
            if cls == "title" or chars > max_chars // 4:
                flush()
            headers[level] = text
            # drop deeper headers when a shallower one resets
            for k in list(headers):
                if k > level:
                    del headers[k]
            section_path = " > ".join(
                headers[lvl].lstrip("#").strip()
                for lvl in sorted(headers) if lvl <= 3
            )
            header_md = "#" * level + " " + text
            parts.append(header_md)
            chars += len(header_md)
            continue

        # Picture + optional following caption
        if cls == "picture":
            cap = ""
            if i < len(elements) and elements[i][0].lower() == _CAPTION_CLASS:
                cap = elements[i][1].strip()
                i += 1
            block = (f"**[Figure]** *{cap}*\n\n{text}" if cap and text
                     else text or (f"[Image: {cap}]" if cap else ""))
            if block:
                add_unit(block)
            continue

        # Atomic element (table / formula) + optional caption
        if cls in _ATOMIC_CLASSES:
            formatted = fmt(cls, text)
            block_parts = [formatted]
            if i < len(elements) and elements[i][0].lower() == _CAPTION_CLASS:
                cap = elements[i][1].strip()
                if cap:
                    block_parts.append(f"*{cap}*")
                i += 1
            block = "\n\n".join(block_parts)
            if len(block) >= max_chars:
                # oversized atomic — emit as its own chunk, do NOT split
                flush()
                chunks.append((block, section_path))
            else:
                add_unit(block)
            continue

        # Regular text / list-item / etc.
        unit = f"*{text}*" if cls == _CAPTION_CLASS else fmt(cls, text)
        if overlap_chars > 0 and len(unit) > max_chars:
            # Only here is overlap actually applied — when a SINGLE text
            # element is too large to fit and must be sub-divided.
            for sub in _split_oversized_text(unit, max_chars, overlap_chars):
                add_unit(sub)
        else:
            add_unit(unit)

    flush()
    return chunks if chunks else [("", "")]


def _split_oversized_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split a single oversized text element with overlap at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = start + max_chars
        if end >= n:
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break
        # prefer to split at sentence-end punctuation
        split_at = text.rfind(". ", start + overlap_chars, end)
        if split_at > start:
            split_at += 1
        else:
            split_at = text.rfind(" ", start + overlap_chars, end)
            if split_at <= start:
                split_at = end
        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, split_at - overlap_chars)
    return [c for c in chunks if c.strip()]
```

## Format-specific extractors

Each extractor converts one input format into the element-tuple list the
chunker expects:

| Source | Extractor | Library used |
|---|---|---|
| HTML | `html_to_elements(html_str)` | BeautifulSoup; walks the DOM, mapping `<h1>`/`<h2>` → section-header, `<table>` → table (via `html_table_to_markdown`), etc. Strips nav/footer/aside/script. |
| Markdown | `md_to_elements(md_str)` | Pure regex; handles ATX + Setext headings, fenced code, GFM tables, ordered/unordered lists, blockquotes |
| RST | `rst_to_elements(rst_str)` | Pure regex; section adornments → headings, code blocks, tables |
| PDF | (use Nemotron-Parse output directly) | Nemotron-Parse emits the same `(class, text)` shape natively |
| Plain `.txt` | Paragraph-fallback chunker (see below) | None — no semantic structure available |

The canonical implementations live in
`rag-crawler/src/crawler/chunker.py` (HTML/MD/RST) and
`rag/src/nvidia_rag/tools/parse_document.py:_split_by_semantic_elements`
(PDF post-parser).

## Paragraph fallback for raw text

When the input is unstructured text (a `.txt` file with no markup), use
this simpler greedy-paragraph chunker. Same conditional-overlap principle.

```python
def chunk_text(
    text: str,
    chunk_size: int = 8192,   # chars ≈ 2048 tokens (4:1 approx)
    chunk_overlap: int = 600, # chars ≈ 150 tokens
) -> list[str]:
    """Greedy paragraph-based chunker with conditional overlap."""
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    chunks, buf, buf_len = [], [], 0
    for para in paragraphs:
        plen = len(para) + 2  # include separator
        if buf and buf_len + plen > chunk_size:
            chunks.append("\n\n".join(buf))
            # seed next buffer with tail-overlap
            overlap_buf, overlap_len = [], 0
            for p in reversed(buf):
                pl = len(p) + 2
                if overlap_len + pl > chunk_overlap:
                    break
                overlap_buf.insert(0, p)
                overlap_len += pl
            buf = overlap_buf
            buf_len = overlap_len
        buf.append(para)
        buf_len += plen
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks
```

## Tunable parameters

| Parameter | Default | When to change |
|---|---|---|
| `max_tokens` | 512–2048 (your choice; depends on retrieval pipeline) | Match your embedding model's context (most are 512 or 8192) |
| `chunk_overlap` | 150 tokens | Higher (250+) for short queries / sparse retrieval; 0 for high-recall scenarios where you want zero duplicate content |
| char/token ratio | 4:1 | Use a real tokenizer (tiktoken, HuggingFace) if you need precision; 4:1 is conservative for English |

## Not for

- **Tables larger than `max_chars`.** The chunker emits these as a single
  oversized chunk rather than splitting them — your embedding model
  needs to handle that or you need to pre-shrink tables (cell trimming,
  column dropping) before chunking.
- **Languages where 4:1 char/token doesn't hold.** CJK languages need
  closer to 1.5:1; pass your own tokenizer if accuracy matters.
- **Cross-document context.** This module operates within a single
  document. Don't expect it to recognize that two PDFs cover the same
  topic.

## Wiring into Stage 1

Stage 1's crawl produces either parsed HTML/MD elements directly or
PDF-extracted elements (from Module 03). In both cases, feed the
element list into `split_by_semantic_elements` and write the resulting
chunks to your vector store. For plain-text linked files, use the
paragraph fallback.
