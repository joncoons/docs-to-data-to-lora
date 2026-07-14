"""
chunker.py — HTML semantic chunking and XML preprocessing.

Self-contained: no nvidia_rag imports.  Extracted from:
  - nvidia_rag.tools.crawl.SimpleWebCrawler._html_to_elements
  - nvidia_rag.tools.crawl.SimpleWebCrawler._html_table_to_markdown
  - nvidia_rag.tools.parse_document.DocumentClassifierRouter._split_by_semantic_elements
  - nvidia_rag.utils.xml_preprocessor.xml_to_markdown
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

logger = logging.getLogger(__name__)

# ── Chunker constants (mirrors parse_document.py) ─────────────────────────────
_SECTION_STARTERS: frozenset[str] = frozenset({"title", "section-header"})
_ATOMIC_CLASSES: frozenset[str] = frozenset({"table", "formula"})
_SKIP_CLASSES: frozenset[str] = frozenset({"page-header", "page-footer", "toc"})
_CAPTION_CLASS: str = "caption"


def html_table_to_markdown(table_tag: Any) -> str:
    """Convert a BeautifulSoup <table> tag to GitHub-Flavored Markdown."""
    rows: list[str] = []
    for tr in table_tag.find_all("tr"):
        cells = [
            cell.get_text(separator=" ", strip=True).replace("|", "\\|")
            for cell in tr.find_all(["th", "td"])
        ]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    header = rows[0]
    col_count = max(1, header.count("|") - 1)
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    return "\n".join([header, separator] + rows[1:])


def html_to_elements(html_str: str) -> list[tuple[str, str]]:
    """
    Convert raw HTML to (class_name, text) element pairs using the nemoretriever-parse
    taxonomy: Title, Section-header, Text, List-item, Table, Formula, Caption.

    Navigation containers (nav, header, footer, aside, script, style) are stripped
    before extraction.
    """
    try:
        from bs4 import BeautifulSoup, Tag
    except ImportError:
        logger.warning("beautifulsoup4 not available; HTML semantic chunking skipped")
        return []

    soup = BeautifulSoup(html_str, "html.parser")

    for noise_tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style"]):
        noise_tag.decompose()

    main: Any = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", id="content")
        or soup.find("body")
        or soup
    )

    elements: list[tuple[str, str]] = []

    def _walk(node: Any, in_list: bool = False) -> None:
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            name = child.name

            if name == "h1":
                t = child.get_text(separator=" ", strip=True)
                if t:
                    elements.append(("Title", t))
            elif name in ("h2", "h3", "h4", "h5", "h6"):
                t = child.get_text(separator=" ", strip=True)
                if t:
                    elements.append(("Section-header", t))
            elif name == "p":
                t = child.get_text(separator=" ", strip=True)
                if t:
                    elements.append(("Text", t))
            elif name in ("ul", "ol"):
                _walk(child, in_list=True)
            elif name == "li":
                t = child.get_text(separator=" ", strip=True)
                if t:
                    elements.append(("List-item", t))
            elif name == "table":
                md = html_table_to_markdown(child)
                if md:
                    elements.append(("Table", md))
            elif name == "pre":
                t = child.get_text(strip=True)
                if t:
                    elements.append(("Formula", t))
            elif name == "figcaption":
                t = child.get_text(separator=" ", strip=True)
                if t:
                    elements.append(("Caption", t))
            elif name == "blockquote":
                t = child.get_text(separator=" ", strip=True)
                if t:
                    elements.append(("Text", t))
            elif name in (
                "div", "section", "article", "main", "figure",
                "details", "summary", "form",
            ):
                _walk(child, in_list=in_list)

    _walk(main)
    return elements


def _build_header_context(headers: dict[int, str]) -> str:
    """Return H1–H3 header lines as context prefix for overflow chunks."""
    lines = [headers[lvl] for lvl in sorted(headers) if lvl <= 3]
    return "\n".join(lines)


def _section_path_string(headers: dict[int, str]) -> str:
    """Return human-readable section breadcrumb, e.g. 'Overview > Architecture'."""
    parts = [headers[lvl].lstrip("#").strip() for lvl in sorted(headers) if lvl <= 3]
    return " > ".join(p for p in parts if p)


def _split_oversized_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split text into sub-chunks when it exceeds max_chars, with overlap."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + max_chars
        if end >= text_len:
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

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


def split_by_semantic_elements(
    elements: list[tuple[str, str]],
    max_tokens: int,
    chunk_overlap: int = 150,
) -> list[tuple[str, str]]:
    """
    Build non-homogeneous semantic chunks from (class_name, text) element pairs.

    Returns list of (chunk_text, section_path) pairs.
    """
    max_chars = max_tokens * 4
    overlap_chars = chunk_overlap * 4

    chunks: list[tuple[str, str]] = []
    parts: list[str] = []
    chars: int = 0
    headers: dict[int, str] = {}
    section_path: str = ""

    def flush() -> None:
        nonlocal parts, chars
        body = "\n\n".join(p for p in parts if p.strip()).strip()
        if body:
            chunks.append((body, section_path))
        parts = []
        chars = 0

    def _add(unit: str) -> None:
        nonlocal parts, chars
        sep = 2 if parts else 0
        if parts and chars + sep + len(unit) > max_chars:
            flush()
            ctx = _build_header_context(headers)
            if ctx:
                parts.append(ctx)
                chars = len(ctx)
                sep = 2
            parts.append(unit)
            chars += sep + len(unit)
        else:
            parts.append(unit)
            chars += sep + len(unit)

    def _fmt(cls_l: str, text: str) -> str:
        if cls_l == "table":
            return text
        if cls_l == "formula":
            if not text.startswith(("```", "$$")):
                return f"```\n{text}\n```"
            return text
        if cls_l == "list-item":
            lines = [
                f"- {ln}" if not ln.startswith(("- ", "* ", "• ")) else ln
                for ln in text.splitlines()
                if ln.strip()
            ]
            return "\n".join(lines) if lines else f"- {text}"
        if cls_l == "footnote":
            return f"> {text}"
        return text

    i = 0
    while i < len(elements):
        cls, text = elements[i]
        cls_l = cls.lower()
        text = text.strip()
        i += 1

        if cls_l in _SKIP_CLASSES or (not text and cls_l not in _ATOMIC_CLASSES):
            continue

        if cls_l in _SECTION_STARTERS:
            level = 1 if cls_l == "title" else 2
            if cls_l == "title" or chars > max_chars // 4:
                flush()
            headers[level] = text
            for k in list(headers):
                if k > level:
                    del headers[k]
            section_path = _section_path_string(headers)
            header_md = "#" * level + " " + text
            parts.append(header_md)
            chars += len(header_md)
            continue

        if cls_l == "picture":
            cap = ""
            if i < len(elements) and elements[i][0].lower() == _CAPTION_CLASS:
                cap = elements[i][1].strip()
                i += 1
            if text:
                parts_list = []
                if cap:
                    parts_list.append(f"**[Figure]** *{cap}*")
                parts_list.append(text)
                _add("\n\n".join(parts_list))
            elif cap:
                _add(f"[Image: {cap}]")
            continue

        if cls_l in _ATOMIC_CLASSES:
            formatted = _fmt(cls_l, text)
            atom_parts_list = [formatted]
            atom_chars = len(formatted)
            if i < len(elements) and elements[i][0].lower() == _CAPTION_CLASS:
                cap = elements[i][1].strip()
                if cap:
                    cap_md = f"*{cap}*"
                    atom_parts_list.append(cap_md)
                    atom_chars += 2 + len(cap_md)
                i += 1
            atom_block = "\n\n".join(atom_parts_list)
            if atom_chars >= max_chars:
                flush()
                chunks.append((atom_block, section_path))
            else:
                _add(atom_block)
            continue

        unit = f"*{text}*" if cls_l == _CAPTION_CLASS else _fmt(cls_l, text)
        if overlap_chars > 0 and len(unit) > max_chars:
            for sub in _split_oversized_text(unit, max_chars, overlap_chars):
                _add(sub)
        else:
            _add(unit)

    flush()
    return chunks if chunks else [("", "")]


# ── Markdown → elements ────────────────────────────────────────────────────────

# Characters that RST uses as section-title adornments
_RST_ADORNMENT_RE = re.compile(r"^([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])\1{2,}$")


def md_to_elements(md_str: str) -> list[tuple[str, str]]:
    """
    Convert Markdown text to (class_name, text) element pairs using the same
    taxonomy as html_to_elements: Title, Section-header, Text, List-item,
    Table, Formula.

    Handles: ATX headings, setext headings, fenced code blocks (``` / ~~~),
    GFM tables, ordered/unordered lists, blockquotes, and paragraphs.
    Inline markup (bold, italic, links) is stripped from paragraph text.
    """
    elements: list[tuple[str, str]] = []
    lines = md_str.splitlines()
    n = len(lines)
    i = 0

    def _strip_inline(text: str) -> str:
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[Image: \1]", text)  # images
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)             # links
        text = re.sub(r"\*{2}([^*]+)\*{2}", r"\1", text)                 # **bold**
        text = re.sub(r"__([^_]+)__", r"\1", text)                       # __bold__
        text = re.sub(r"\*([^*\n]+)\*", r"\1", text)                     # *italic*
        text = re.sub(r"_([^_\n]+)_", r"\1", text)                       # _italic_
        return text.strip()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Fenced code block  (``` or ~~~)
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            code_lines: list[str] = []
            i += 1
            while i < n:
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                code_lines.append(lines[i])
                i += 1
            code = "\n".join(code_lines).strip()
            if code:
                elements.append(("Formula", code))
            continue

        # Setext heading: current line + next line is ==+ or ---+ (≥2 dashes)
        if i + 1 < n:
            nxt = lines[i + 1].strip()
            if nxt and re.match(r"^=+$", nxt):
                elements.append(("Title", stripped))
                i += 2
                continue
            # Avoid confusing `- item` bullet lines with setext underlines
            if nxt and re.match(r"^-{2,}$", nxt) and not re.match(r"^[-*+]\s", line):
                elements.append(("Section-header", stripped))
                i += 2
                continue

        # ATX heading  (# … ######)
        m = re.match(r"^(#{1,6})\s+(.*?)(?:\s+#+\s*)?$", line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            if text:
                elements.append(("Title" if level == 1 else "Section-header", text))
            i += 1
            continue

        # GFM table: first line contains "|" and the very next is a separator row
        if "|" in line and i + 1 < n:
            sep = lines[i + 1].strip()
            if re.match(r"^\|?[\s\-:|]+\|[\s\-:|]*$", sep):
                table_lines = [stripped]
                j = i + 1
                while j < n and "|" in lines[j] and lines[j].strip():
                    table_lines.append(lines[j].strip())
                    j += 1
                if len(table_lines) >= 2:
                    elements.append(("Table", "\n".join(table_lines)))
                    i = j
                    continue

        # Unordered list item
        m = re.match(r"^\s{0,3}[-*+]\s+(.*)", line)
        if m:
            text = m.group(1).strip()
            if text:
                elements.append(("List-item", _strip_inline(text)))
            i += 1
            continue

        # Ordered list item
        m = re.match(r"^\s{0,3}\d+[.)]\s+(.*)", line)
        if m:
            text = m.group(1).strip()
            if text:
                elements.append(("List-item", _strip_inline(text)))
            i += 1
            continue

        # Blockquote
        if line.startswith(">"):
            text = line.lstrip(">").strip()
            if text:
                elements.append(("Text", _strip_inline(text)))
            i += 1
            continue

        # Horizontal rule — skip
        if re.match(r"^[-*_]{3,}\s*$", stripped):
            i += 1
            continue

        # Paragraph — accumulate until blank line or block-level starter
        para_lines = [line]
        i += 1
        while i < n:
            cur = lines[i]
            cur_s = cur.strip()
            if not cur_s:
                break
            if (cur_s.startswith("#") or
                    cur_s.startswith("```") or cur_s.startswith("~~~") or
                    re.match(r"^\s{0,3}[-*+]\s", cur) or
                    re.match(r"^\s{0,3}\d+[.)]\s", cur) or
                    cur_s.startswith(">")):
                break
            # Peek ahead for setext heading
            if i + 1 < n:
                nn = lines[i + 1].strip()
                if re.match(r"^[=\-]{2,}$", nn) and cur_s:
                    break
            para_lines.append(cur)
            i += 1

        text = " ".join(ln.strip() for ln in para_lines if ln.strip())
        text = _strip_inline(text)
        if text:
            elements.append(("Text", text))

    return elements


def rst_to_elements(rst_str: str) -> list[tuple[str, str]]:
    """
    Convert reStructuredText to (class_name, text) element pairs (simplified).

    Handles: section titles (overline/underline adornments), ``.. code-block::``
    and ``.. code::`` directives, bullet/enumerated lists, and paragraphs.
    Other directives (note, warning, etc.) have their body collected as Text.
    Inline RST markup (emphasis, strong, hyperlinks) is stripped.
    """
    elements: list[tuple[str, str]] = []
    lines = rst_str.splitlines()
    n = len(lines)
    i = 0
    heading_chars: list[str] = []  # first encounter = title, subsequent = section-header

    def _heading_cls(ch: str) -> str:
        if ch not in heading_chars:
            heading_chars.append(ch)
        return "Title" if heading_chars.index(ch) == 0 else "Section-header"

    def _strip_rst(text: str) -> str:
        text = re.sub(r"`([^`]+)`__?", r"\1", text)        # hyperlinks
        text = re.sub(r"\*{2}([^*]+)\*{2}", r"\1", text)   # **strong**
        text = re.sub(r"\*([^*]+)\*", r"\1", text)          # *emphasis*
        text = re.sub(r"``([^`]+)``", r"\1", text)          # ``literal``
        return text.strip()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Overline + text + underline  (all three lines use same adornment char)
        if _RST_ADORNMENT_RE.match(stripped) and i + 2 < n:
            title_text = lines[i + 1].strip()
            under = lines[i + 2].strip()
            if title_text and _RST_ADORNMENT_RE.match(under) and under[0] == stripped[0]:
                elements.append((_heading_cls(stripped[0]), title_text))
                i += 3
                continue

        # Text + underline  (setext-style)
        if i + 1 < n:
            nxt = lines[i + 1].strip()
            if nxt and _RST_ADORNMENT_RE.match(nxt) and not _RST_ADORNMENT_RE.match(stripped):
                elements.append((_heading_cls(nxt[0]), stripped))
                i += 2
                continue

        # Directive
        if re.match(r"^\.\.\s+\w", stripped):
            is_code = bool(re.match(r"^\.\.\s+code(-block)?::", stripped))
            i += 1
            # skip option lines (:linenos: etc.) and blank lines
            while i < n and (not lines[i].strip() or re.match(r"^\s+:\w", lines[i])):
                i += 1
            # collect indented body
            body_lines: list[str] = []
            while i < n and (not lines[i].strip() or lines[i].startswith(" ")):
                body_lines.append(lines[i])
                i += 1
            body = "\n".join(body_lines).strip()
            if body:
                elements.append(("Formula" if is_code else "Text", body))
            continue

        # Bullet list  (-, *, +, •)
        m = re.match(r"^\s{0,3}[-*+•]\s+(.*)", line)
        if m:
            text = _strip_rst(m.group(1).strip())
            if text:
                elements.append(("List-item", text))
            i += 1
            continue

        # Enumerated list  (1. / a. / #.)
        m = re.match(r"^\s{0,3}(?:\d+|[a-zA-Z]|#)[.)]\s+(.*)", line)
        if m:
            text = _strip_rst(m.group(1).strip())
            if text:
                elements.append(("List-item", text))
            i += 1
            continue

        # Paragraph
        para_lines = [line]
        i += 1
        while i < n:
            cur = lines[i]
            cur_s = cur.strip()
            if not cur_s:
                break
            if re.match(r"^\.\.\s+\w", cur_s):
                break
            if re.match(r"^\s{0,3}[-*+•]\s", cur) or re.match(r"^\s{0,3}(?:\d+|[a-zA-Z]|#)[.)]\s", cur):
                break
            # Peek: setext heading coming up
            if i + 1 < n and _RST_ADORNMENT_RE.match(lines[i + 1].strip()) and cur_s:
                break
            para_lines.append(cur)
            i += 1

        text = " ".join(ln.strip() for ln in para_lines if ln.strip())
        text = _strip_rst(text)
        if text:
            elements.append(("Text", text))

    return elements


# ── XML → Markdown ─────────────────────────────────────────────────────────────

def xml_to_markdown(xml_bytes: bytes) -> str:
    """
    Convert RSS/Atom/Sitemap/generic XML bytes to Markdown text.

    Returns empty string when the input is not parseable XML or contains
    no extractable text content.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.debug("xml_to_markdown: XML parse error: %s", exc)
        return ""

    tag = root.tag
    ns = tag.split("}")[0].strip("{") if "}" in tag else ""

    def q(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    lines: list[str] = []

    # ── RSS 2.0 ──────────────────────────────────────────────────────────────
    channel = root.find("channel")
    if channel is not None:
        title_el = channel.find("title")
        if title_el is not None and title_el.text:
            lines.append(f"# {title_el.text.strip()}")
        for item in channel.findall("item"):
            item_title = item.findtext("title", "").strip()
            item_link = item.findtext("link", "").strip()
            item_desc = item.findtext("description", "").strip()
            if item_title:
                lines.append(f"\n## {item_title}")
            if item_link:
                lines.append(f"Link: {item_link}")
            if item_desc:
                lines.append(item_desc)
        if lines:
            return "\n".join(lines)

    # ── Atom 1.0 ─────────────────────────────────────────────────────────────
    feed_title = root.findtext(q("title"), "").strip()
    entries = root.findall(q("entry"))
    if entries:
        if feed_title:
            lines.append(f"# {feed_title}")
        for entry in entries:
            entry_title = entry.findtext(q("title"), "").strip()
            entry_summary = entry.findtext(q("summary"), "").strip()
            entry_content = entry.findtext(q("content"), "").strip()
            link_el = entry.find(q("link"))
            entry_link = link_el.get("href", "") if link_el is not None else ""
            if entry_title:
                lines.append(f"\n## {entry_title}")
            if entry_link:
                lines.append(f"Link: {entry_link}")
            if entry_summary:
                lines.append(entry_summary)
            elif entry_content:
                lines.append(entry_content)
        if lines:
            return "\n".join(lines)

    # ── Generic XML: extract all text nodes ──────────────────────────────────
    def _extract_text(node: ET.Element, depth: int = 0) -> None:
        tag_local = node.tag.split("}")[-1] if "}" in node.tag else node.tag
        text = (node.text or "").strip()
        if text:
            prefix = "#" * min(depth + 1, 6) + " " if depth == 0 else ""
            lines.append(f"{prefix}{tag_local}: {text}")
        for child in node:
            _extract_text(child, depth + 1)
        tail = (node.tail or "").strip()
        if tail:
            lines.append(tail)

    _extract_text(root)
    return "\n".join(lines)
