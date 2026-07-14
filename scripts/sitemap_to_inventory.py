#!/usr/bin/env python3
"""Parse a sitemap.xml into a CSV inventory keyed by source_area / version / page.

Identifies the first segment matching a version-like pattern
(`/\\d+\\.\\d+/`, `latest`, `stable`, `dev`, `main`, `nightly`, `vN`,
`YYYY-MM-DD`) and splits the URL into source_area / version / page around it.

Usage:
    python sitemap_to_inventory.py \\
        --sitemap https://docs.example.com/foo/sitemap.xml \\
        [--path-prefix /foo/] \\
        --output inventory.csv

Output CSV columns: url, source_area, version, page, path_depth, lastmod
"""
import argparse
import csv
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from urllib.parse import urlparse

VERSION_LIKE = re.compile(
    r"^("
    r"latest|stable|dev|main|nightly|"
    r"v?\d+(\.\d+)+([a-zA-Z0-9.\-]*)?|"
    r"\d{4}-\d{2}-\d{2}"
    r")$"
)
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def fetch_sitemap(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sitemap-to-inventory/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def classify(url: str, path_prefix: str | None) -> tuple[str, str, str]:
    """Split URL into (source_area, version, page).

    If `path_prefix` is given, strip it before classification. The first
    segment matching VERSION_LIKE marks the boundary between source_area and
    page; everything before is source_area, the segment itself is version,
    everything after is page.
    """
    path = urlparse(url).path.strip("/")
    if path_prefix:
        normalized = path_prefix.strip("/")
        if path.startswith(normalized + "/"):
            path = path[len(normalized) + 1:]
        elif path == normalized:
            path = ""

    parts = path.split("/") if path else []

    version_idx = None
    for i, seg in enumerate(parts):
        if VERSION_LIKE.match(seg):
            version_idx = i
            break

    if version_idx is None:
        if not parts:
            return ("", "", "")
        if len(parts) == 1:
            return (parts[0], "", "")
        return ("/".join(parts[:-1]), "", parts[-1])

    source_area = "/".join(parts[:version_idx])
    version = parts[version_idx]
    page = "/".join(parts[version_idx + 1:])
    return (source_area, version, page)


def parse_urlset(xml_bytes: bytes) -> list[tuple[str, str]]:
    """Return list of (loc, lastmod) tuples from a sitemap urlset."""
    root = ET.fromstring(xml_bytes)
    out = []
    for url_el in root.findall("sm:url", SITEMAP_NS):
        loc = (url_el.findtext("sm:loc", default="", namespaces=SITEMAP_NS) or "").strip()
        lastmod = (url_el.findtext("sm:lastmod", default="", namespaces=SITEMAP_NS) or "").strip()
        if loc:
            out.append((loc, lastmod))
    return out


def parse_sitemap_index(xml_bytes: bytes) -> list[str]:
    """Return list of sitemap URLs from a sitemapindex (None if not an index)."""
    root = ET.fromstring(xml_bytes)
    locs = []
    for sm_el in root.findall("sm:sitemap", SITEMAP_NS):
        loc = (sm_el.findtext("sm:loc", default="", namespaces=SITEMAP_NS) or "").strip()
        if loc:
            locs.append(loc)
    return locs


def fetch_all_urls(sitemap_url: str, max_depth: int = 3) -> list[tuple[str, str]]:
    """Follow sitemap-indexes recursively, returning (loc, lastmod) pairs."""
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(sitemap_url, 0)]
    urls: list[tuple[str, str]] = []

    while queue:
        url, depth = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        try:
            xml = fetch_sitemap(url)
        except Exception as exc:
            print(f"  WARN: failed to fetch {url}: {exc}", file=sys.stderr)
            continue
        # Try urlset first, then sitemapindex
        try:
            page_urls = parse_urlset(xml)
            if page_urls:
                urls.extend(page_urls)
                continue
        except ET.ParseError:
            pass
        try:
            child_sitemaps = parse_sitemap_index(xml)
            for c in child_sitemaps:
                queue.append((c, depth + 1))
        except ET.ParseError as exc:
            print(f"  WARN: could not parse {url}: {exc}", file=sys.stderr)
    return urls


def summarize(rows: list[dict]) -> None:
    by_source_area: dict[str, dict] = defaultdict(
        lambda: {"versions": set(), "url_count": 0, "lastmods": []}
    )
    for r in rows:
        p = by_source_area[r["source_area"]]
        if r["version"]:
            p["versions"].add(r["version"])
        p["url_count"] += 1
        if r["lastmod"]:
            p["lastmods"].append(r["lastmod"])

    print(f"\nDistinct source areas: {len(by_source_area)}")
    has_latest = [p for p, d in by_source_area.items() if "latest" in d["versions"]]
    no_latest = [p for p, d in by_source_area.items() if "latest" not in d["versions"] and d["versions"]]
    no_version = [p for p, d in by_source_area.items() if not d["versions"]]
    print(f"  with /latest/:      {len(has_latest)}")
    print(f"  no /latest/:        {len(no_latest)}  (need pinned-version curation)")
    print(f"  no version segment: {len(no_version)}")

    if no_latest:
        print("\n=== Source areas WITHOUT /latest/ (need version pinning) ===")
        print(f"{'source_area':50s} {'urls':>5s}  {'last-modified':14s}  versions")
        print("-" * 110)
        for p in sorted(no_latest):
            d = by_source_area[p]
            latest_mod = max(d["lastmods"]) if d["lastmods"] else ""
            vers = ",".join(sorted(d["versions"]))
            print(f"{p:50s} {d['url_count']:>5d}  {latest_mod:14s}  {vers}")

    if has_latest:
        print("\n=== Source areas WITH /latest/ (auto-curated) ===")
        print(f"{'source_area':50s} {'urls':>5s}  {'last-modified':14s}")
        print("-" * 80)
        for p in sorted(has_latest):
            d = by_source_area[p]
            latest_mod = max(d["lastmods"]) if d["lastmods"] else ""
            print(f"{p:50s} {d['url_count']:>5d}  {latest_mod:14s}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sitemap", required=True, help="Sitemap URL (urlset or sitemapindex)")
    parser.add_argument("--path-prefix", default=None,
                        help="Optional path-prefix filter (e.g., /nim/). URLs not matching are excluded.")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-source-area summary")
    args = parser.parse_args()

    print(f"Fetching sitemap: {args.sitemap}", file=sys.stderr)
    raw_urls = fetch_all_urls(args.sitemap)
    print(f"  raw URLs: {len(raw_urls)}", file=sys.stderr)

    rows = []
    for loc, lastmod in raw_urls:
        if args.path_prefix and args.path_prefix not in urlparse(loc).path:
            continue
        source_area, version, page = classify(loc, args.path_prefix)
        path_depth = len([p for p in urlparse(loc).path.strip("/").split("/") if p])
        rows.append({
            "url": loc,
            "source_area": source_area,
            "version": version,
            "page": page,
            "path_depth": path_depth,
            "lastmod": lastmod,
        })

    if not rows:
        print("ERROR: no URLs matched after filtering", file=sys.stderr)
        return 1

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "source_area", "version", "page", "path_depth", "lastmod"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} URLs to {args.output}", file=sys.stderr)
    if not args.quiet:
        summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
