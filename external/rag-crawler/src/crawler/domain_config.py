"""
domain_config.py — per-domain crawl profile management.

Each profile is stored as a YAML file in CONFIGS_DIR:
  docs_nvidia_com.yaml, medtronic_com.yaml, etc.

On first startup, seed_default_configs() writes the NVIDIA default
profile if it does not already exist.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import aiohttp
import yaml
from pydantic import BaseModel, Field

from . import config

logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────

class CollectionRoutingConfig(BaseModel):
    mode: Literal["fixed", "product_url_map"] = "fixed"
    default_collection: str = ""


class SitemapConfig(BaseModel):
    enabled: bool = False


class CrawlDefaults(BaseModel):
    max_pages: int | None = None
    max_depth: int | None = None
    batch_ingest_size: int = 50
    extract_linked_files: bool = False
    allowed_url_prefixes: list[str] = Field(default_factory=list)


class DomainConfig(BaseModel):
    domain: str
    label: str = ""
    collection_routing: CollectionRoutingConfig = Field(default_factory=CollectionRoutingConfig)
    sitemap: SitemapConfig = Field(default_factory=SitemapConfig)
    crawl_defaults: CrawlDefaults = Field(default_factory=CrawlDefaults)


# ── Helpers ───────────────────────────────────────────────────────────────────

def domain_to_slug(domain: str) -> str:
    """Convert a domain name to a filesystem-safe slug.

    "docs.nvidia.com" → "docs_nvidia_com"
    """
    return re.sub(r"[^a-z0-9]", "_", domain.lower()).strip("_")


def _configs_dir() -> Path:
    d = Path(config.CONFIGS_DIR)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


# ── CRUD ──────────────────────────────────────────────────────────────────────

def list_configs() -> list[DomainConfig]:
    """Return all domain configs sorted by domain name."""
    cfgs: list[DomainConfig] = []
    for p in sorted(_configs_dir().glob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text())
            if raw and isinstance(raw, dict):
                cfgs.append(DomainConfig(**raw))
        except Exception as exc:
            logger.warning("domain_config: could not load %s: %s", p, exc)
    return cfgs


def load_config(slug: str) -> DomainConfig | None:
    """Load a single config by slug."""
    p = _configs_dir() / f"{slug}.yaml"
    if not p.exists():
        return None
    try:
        raw = yaml.safe_load(p.read_text())
        return DomainConfig(**raw) if raw else None
    except Exception as exc:
        logger.warning("domain_config: could not load %s: %s", p, exc)
        return None


def save_config(cfg: DomainConfig) -> None:
    """Write a DomainConfig to YAML (create or overwrite)."""
    slug = domain_to_slug(cfg.domain)
    p = _configs_dir() / f"{slug}.yaml"
    try:
        p.write_text(
            yaml.safe_dump(cfg.model_dump(), default_flow_style=False, sort_keys=False)
        )
    except Exception as exc:
        logger.error("domain_config: could not save %s: %s", p, exc)
        raise


def delete_config(slug: str) -> bool:
    """Delete a config file by slug. Returns False if not found."""
    p = _configs_dir() / f"{slug}.yaml"
    if not p.exists():
        return False
    try:
        p.unlink()
        return True
    except Exception as exc:
        logger.error("domain_config: could not delete %s: %s", p, exc)
        return False


def match_config(url: str) -> DomainConfig | None:
    """Find the best-matching config for a URL using longest-domain-suffix matching."""
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return None
    if not hostname:
        return None

    cfgs = list_configs()
    # Longest domain first so more-specific entries win
    cfgs.sort(key=lambda c: len(c.domain), reverse=True)
    for cfg in cfgs:
        d = cfg.domain.lower().lstrip(".")
        if hostname == d or hostname.endswith("." + d):
            return cfg
    return None


# ── Default config seeding ────────────────────────────────────────────────────

_NVIDIA_DEFAULT: dict = {
    "domain": "docs.nvidia.com",
    "label": "NVIDIA Documentation",
    "collection_routing": {
        "mode": "product_url_map",
        "default_collection": "nvidia_docs",
    },
    "sitemap": {"enabled": False},
    "crawl_defaults": {
        "max_pages": None,
        "max_depth": 4,
        "batch_ingest_size": 50,
        "extract_linked_files": False,
        "allowed_url_prefixes": [],
    },
}


def seed_default_configs() -> None:
    """Write the NVIDIA default config on first startup if it does not yet exist."""
    slug = domain_to_slug("docs.nvidia.com")
    p = _configs_dir() / f"{slug}.yaml"
    if p.exists():
        return
    try:
        p.write_text(
            yaml.safe_dump(_NVIDIA_DEFAULT, default_flow_style=False, sort_keys=False)
        )
        logger.info("domain_config: seeded default NVIDIA config → %s", p)
    except Exception as exc:
        logger.warning("domain_config: could not seed NVIDIA config: %s", exc)


# ── Generate from sitemap ─────────────────────────────────────────────────────

async def generate_from_sitemap(start_url: str) -> DomainConfig:
    """
    Draft a DomainConfig by inspecting the site's robots.txt and sitemaps.

    1. Parse hostname from start_url
    2. Fetch robots.txt → extract Sitemap: lines
    3. Fetch up to 3 sitemaps, expand sitemapindex one level, collect <loc> URLs
    4. Derive allowed_url_prefixes from start_url path (if non-trivial)
    """
    parsed = urlparse(start_url)
    hostname = (parsed.hostname or "").lower()
    base = f"{parsed.scheme}://{parsed.netloc}"

    sitemap_urls: list[str] = []

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15),
        headers={"User-Agent": "NVIDIA-RAG-Crawler/1.0"},
    ) as session:
        # Fetch robots.txt
        try:
            async with session.get(f"{base}/robots.txt") as r:
                if r.status == 200:
                    for line in (await r.text()).splitlines():
                        if line.lower().startswith("sitemap:"):
                            sm = line.split(":", 1)[1].strip()
                            if sm:
                                sitemap_urls.append(sm)
        except Exception as exc:
            logger.debug("generate_from_sitemap: robots.txt error: %s", exc)

        if not sitemap_urls:
            sitemap_urls.append(f"{base}/sitemap.xml")

        # Expand sitemapindex one level
        expanded: list[str] = []
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for sm_url in sitemap_urls[:3]:
            try:
                async with session.get(sm_url) as r:
                    if r.status != 200:
                        continue
                    root = ET.fromstring(await r.text())
                    # sitemapindex → collect nested sitemap locs
                    for loc in root.findall("sm:sitemap/sm:loc", ns):
                        u = (loc.text or "").strip()
                        if u:
                            expanded.append(u)
            except Exception as exc:
                logger.debug("generate_from_sitemap: sitemap fetch error %s: %s", sm_url, exc)
        sitemap_found = bool(sitemap_urls) and sitemap_urls != [f"{base}/sitemap.xml"]

    # Derive allowed_url_prefixes from start_url path
    allowed_prefixes: list[str] = []
    path = parsed.path.rstrip("/")
    if path and path not in ("", "/"):
        allowed_prefixes = [start_url.rstrip("/")]

    return DomainConfig(
        domain=hostname,
        label=hostname,
        collection_routing=CollectionRoutingConfig(mode="fixed", default_collection=""),
        sitemap=SitemapConfig(enabled=sitemap_found),
        crawl_defaults=CrawlDefaults(
            max_pages=None,
            max_depth=None,
            batch_ingest_size=50,
            extract_linked_files=False,
            allowed_url_prefixes=allowed_prefixes,
        ),
    )
