# Module 02 — Cross-Page Text Stitching

When a PDF parser emits one list of elements per page, a sentence that
spans page N and page N+1 will appear as two adjacent text elements with
the page-break artifact in between. Naïve concatenation produces broken
chunks like `"...the configuration is also" / "responsible for routing
inbound..."` — fragments that retrieval and embedding both treat as
incomplete claims.

This module detects these page-spanning fragments and stitches them back
together **before** the [semantic chunker (Module 01)](01-semantic-chunking.md)
runs. It's small (~50 lines), parser-agnostic given the right input
shape, and composes naturally with the rest of Stage 1.

## The input contract

`stitch_page_boundaries` takes a **list of element lists**, one per page,
where each element is the same `(class_name, text)` tuple used by
Module 01:

```python
page_element_lists = [
    [("text", "Setup begins with..."), ("table", "| col | ... |")],     # page 1
    [("text", "routing inbound traffic to..."), ("section-header", "Auth")],  # page 2
    # ...
]
```

It returns a single flat `[(class, text), ...]` list with the page
boundaries either stitched (when warranted) or preserved as-is.

## The decision rules

The stitching decision depends on the class of the boundary elements and
the punctuation at the seam.

| Last element of page N | First element of page N+1 | Stitch? |
|---|---|---|
| `text` or `list-item`, ending in `.`, `!`, `?`, `:`, `;` | (any) | **No** — the sentence completed on page N |
| `text` or `list-item`, ending in any other character | **same class** | **Yes** — clear mid-sentence continuation |
| `text` or `list-item` | **different class** | **No** — class change marks a semantic break |
| `table` | `table` | **Yes** — almost always a table row split |
| `formula` | `formula` | **Yes** — multi-line formula split |
| `title` or `section-header` | (any) | **No** — headers always start fresh |

The terminal-punctuation set is the key signal. English sentences end in
`.`/`!`/`?`. Adding `:` catches list-introducers ("the following:" → list
on next page is its own element, no need to stitch). Adding `;` catches
semicolon-separated clauses where stitching would produce odd compound
sentences.

## What gets dropped along the way

The class taxonomy from Module 01 applies. Specifically:

- `page-header` and `page-footer` elements are filtered out by the
  upstream extractor (or by your own filter pass) **before** stitching.
  If you don't filter them, you'll stitch headers/footers into the
  surrounding text.
- `toc` (table-of-contents) elements should also be filtered, for the
  same reason.

If your parser doesn't classify headers/footers reliably, a position-based
filter (drop elements whose y-coordinate is in the top/bottom 5% of the
page) is a cheap fallback.

## Example code

Portable, no project dependencies. The output is the same flat element
list that Module 01's `split_by_semantic_elements` expects.

```python
"""cross_page_text_stitching.py — stitch elements spanning PDF page boundaries."""

_STITCHABLE_CLASSES = frozenset({"text", "list-item"})
_ATOMIC_CLASSES     = frozenset({"table", "formula"})
_TERMINAL_PUNCT     = frozenset({".", "!", "?", ":", ";"})


def stitch_page_boundaries(
    page_element_lists: list[list[tuple[str, str]]],
) -> list[tuple[str, str]]:
    """
    Merge cross-page element pairs into a single flat element list.

    Args:
        page_element_lists: One element list per PDF page. Each element is
            a (class_name, text) tuple. Page-header/footer/toc elements
            should already be filtered out upstream.

    Returns:
        Flat list of (class, text) tuples with page-break artifacts
        repaired. Pass directly into Module 01's chunker.
    """
    all_elements: list[tuple[str, str]] = []

    for elements in page_element_lists:
        if not elements:
            continue

        if all_elements:
            prev_cls, prev_text = all_elements[-1]
            next_cls, next_text = elements[0]
            prev_l = prev_cls.lower()
            next_l = next_cls.lower()

            should_stitch = False

            if prev_l in _STITCHABLE_CLASSES and prev_l == next_l:
                # Text / list-item: only stitch if previous text doesn't
                # end in terminal punctuation (i.e. mid-sentence break).
                if prev_text and prev_text.rstrip()[-1] not in _TERMINAL_PUNCT:
                    should_stitch = True
            elif prev_l in _ATOMIC_CLASSES and prev_l == next_l:
                # Tables and formulas: always stitch same-class adjacency.
                # The chunker treats these as atomic, so concatenation
                # produces a single coherent unit downstream.
                should_stitch = True

            if should_stitch:
                # Text uses single-space separator; atomic uses newline.
                sep = " " if prev_l in _STITCHABLE_CLASSES else "\n"
                stitched = prev_text.rstrip() + sep + next_text.lstrip()
                all_elements[-1] = (prev_cls, stitched)
                elements = elements[1:]  # consume the first element

        all_elements.extend(elements)

    return all_elements
```

## Wiring into Stage 1

```python
# After per-page parse, before chunking:
page_lists = [parser.extract_page(pdf, page_num) for page_num in range(n_pages)]
flat_elements = stitch_page_boundaries(page_lists)
chunks = split_by_semantic_elements(flat_elements, max_tokens=512)
# → write chunks to vector store
```

## What this does NOT do

- **Page-header / page-footer filtering.** That's upstream of this module.
  The decision rules assume those have already been removed.
- **Visual table reassembly.** This module concatenates two adjacent
  `(table, "...")` elements as text. If the underlying parser produced
  fragmented table markdown (e.g., partial header row on page 1, the
  rest of the rows on page 2), this module's concatenation may still
  yield broken markdown. For visual-level table reassembly, use
  [Module 03](03-visual-stitching-and-routing.md).
- **Multi-page list continuation.** A numbered list that resets numbering
  on each page (1, 2, 3 on page 1; then 1, 2, 3 on page 2 representing
  items 4, 5, 6) will look like two separate lists to this module.
  Re-numbering would require list-context tracking, which this module
  intentionally avoids.
- **Cross-document stitching.** Operates within one document.

## Limitations of the punctuation heuristic

The terminal-punctuation rule is intentionally simple. It will:

- **Miss stitch** when a sentence ends with an abbreviation
  (`U.S.` followed by mid-sentence continuation looks like a sentence
  end). Acceptable — most retrieval can handle the resulting fragment.
- **Wrong-stitch** when a sentence legitimately ends without punctuation
  (rare in technical docs; common in marketing PDFs).

For documents where stitch decisions matter heavily, you can swap the
punctuation rule for a small classifier (predict "incomplete sentence?"
from the last 50 chars) without changing the function signature. Most
SFT corpora don't need this — the punctuation rule is sufficient.

## Origin

Extracted from a local fork of the NVIDIA AI Blueprints RAG ingest pipeline's
`DocumentClassifierRouter._stitch_page_boundaries` helper. The local fork wires this into
`DocumentClassifierRouter.route_document()` between the per-page parse
loop and the chunker (see [Module 03](03-visual-stitching-and-routing.md)
for the full orchestration).
