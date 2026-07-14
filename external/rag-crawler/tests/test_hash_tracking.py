"""Tests for NFS hash tracking: registry load/save, error matrix pre-load,
binary skip logic, and registry update after ingest."""
from __future__ import annotations

import csv
import json
import os
import tempfile

import pytest


@pytest.fixture
def registry_dir(tmp_path):
    """Temporary directory simulating /mnt/nvme2/crawler-registry."""
    return str(tmp_path)


import sys
from pathlib import Path

# Add src to path so we can import crawler
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler.crawl import load_url_registry, save_url_registry


def test_load_url_registry_missing_file(registry_dir):
    """Returns empty dict when no registry file exists."""
    result = load_url_registry("my-collection", registry_dir)
    assert result == {}


def test_save_and_load_url_registry(registry_dir):
    """Round-trips a registry through save then load."""
    data = {
        "https://example.com/page": {
            "content_hash": "sha256:abc123",
            "last_ingested": "2026-04-10T00:00:00+00:00",
            "last_ingested_hash": "sha256:abc123",
        }
    }
    save_url_registry("my-collection", registry_dir, data)
    loaded = load_url_registry("my-collection", registry_dir)
    assert loaded == data


def test_save_url_registry_creates_correct_filename(registry_dir):
    """File is named <slug>_url_registry.json."""
    save_url_registry("My Collection", registry_dir, {"k": "v"})
    expected = os.path.join(registry_dir, "my_collection_url_registry.json")
    assert os.path.exists(expected)


from crawler.crawl import _collection_slug_for


def _write_error_matrix(registry_dir: str, collection: str, broken_urls: list[str]) -> None:
    slug = _collection_slug_for(collection)
    path = os.path.join(registry_dir, f"{slug}_error_matrix.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "url", "status_code", "error"])
        writer.writeheader()
        for url in broken_urls:
            writer.writerow({"category": "broken_links", "url": url, "status_code": 404, "error": "Not Found"})


def test_load_known_errors_from_error_matrix(registry_dir):
    """URLs with category=broken_links in the error matrix are returned as a set."""
    _write_error_matrix(registry_dir, "test-coll", [
        "https://example.com/gone",
        "https://example.com/also-gone",
    ])
    from crawler.crawl import _load_known_permanent_errors
    result = _load_known_permanent_errors("test-coll", registry_dir)
    assert "https://example.com/gone" in result
    assert "https://example.com/also-gone" in result


def test_load_known_errors_missing_file(registry_dir):
    """Returns empty set when no error matrix exists."""
    from crawler.crawl import _load_known_permanent_errors
    result = _load_known_permanent_errors("no-such-coll", registry_dir)
    assert result == set()


def test_load_known_errors_ignores_transient(registry_dir):
    """5xx and timeout errors are not included (only broken_links category)."""
    slug = _collection_slug_for("test-coll")
    path = os.path.join(registry_dir, f"{slug}_error_matrix.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "url", "status_code", "error"])
        writer.writeheader()
        writer.writerow({"category": "other", "url": "https://example.com/flaky", "status_code": 503, "error": "Service Unavailable"})
        writer.writerow({"category": "broken_links", "url": "https://example.com/gone", "status_code": 404, "error": "Not Found"})
    from crawler.crawl import _load_known_permanent_errors
    result = _load_known_permanent_errors("test-coll", registry_dir)
    assert "https://example.com/flaky" not in result
    assert "https://example.com/gone" in result


def test_should_skip_binary_unchanged():
    """Returns True when last_ingested_hash matches stored content_hash."""
    from crawler.crawl import _binary_unchanged
    file_reg = {
        "content_hash": "sha256:abc123",
        "last_ingested_hash": "sha256:abc123",
    }
    assert _binary_unchanged(file_reg, force_recrawl=False) is True


def test_should_not_skip_binary_no_ingested_hash():
    """Returns False when last_ingested_hash is empty (never ingested)."""
    from crawler.crawl import _binary_unchanged
    file_reg = {
        "content_hash": "sha256:abc123",
        "last_ingested_hash": "",
    }
    assert _binary_unchanged(file_reg, force_recrawl=False) is False


def test_should_not_skip_binary_hash_mismatch():
    """Returns False when hashes differ (file changed since last ingest)."""
    from crawler.crawl import _binary_unchanged
    file_reg = {
        "content_hash": "sha256:abc123",
        "last_ingested_hash": "sha256:old456",
    }
    assert _binary_unchanged(file_reg, force_recrawl=False) is False


def test_force_recrawl_bypasses_skip():
    """Returns False even when hashes match if force_recrawl=True."""
    from crawler.crawl import _binary_unchanged
    file_reg = {
        "content_hash": "sha256:abc123",
        "last_ingested_hash": "sha256:abc123",
    }
    assert _binary_unchanged(file_reg, force_recrawl=True) is False


def test_ingest_updates_url_registry(registry_dir):
    """After _update_registries_after_ingest runs, the URL registry for each
    collection has last_ingested and last_ingested_hash set."""
    pytest.importorskip("starlette")
    from crawler.server import _update_registries_after_ingest
    from crawler.crawl import load_url_registry, save_url_registry

    # Seed registry with a URL that has content_hash but no last_ingested
    save_url_registry("cuda", registry_dir, {
        "https://docs.nvidia.com/cuda/some.pdf": {
            "content_hash": "sha256:abc",
            "last_ingested": None,
            "last_ingested_hash": "",
        }
    })

    ingested_urls = {
        "https://docs.nvidia.com/cuda/some.pdf": {
            "collection_name": "cuda",
            "content_hash": "sha256:abc",
        }
    }
    _update_registries_after_ingest(ingested_urls, registry_dir)

    reg = load_url_registry("cuda", registry_dir)
    entry = reg["https://docs.nvidia.com/cuda/some.pdf"]
    assert entry["last_ingested_hash"] == "sha256:abc"
    assert entry["last_ingested"] is not None
    assert entry["last_ingested"] != ""


def test_load_url_registry_normalizes_legacy_bare_hash(registry_dir):
    """Legacy bare hex content hashes are normalized on load."""
    bare_hash = "d" * 64
    path = Path(registry_dir) / "cuda_url_registry.json"
    path.write_text(json.dumps({
        "https://docs.nvidia.com/cuda/page": {
            "content_hash": bare_hash,
            "last_ingested_hash": bare_hash,
        }
    }))

    reg = load_url_registry("cuda", registry_dir)
    entry = reg["https://docs.nvidia.com/cuda/page"]
    assert entry["content_hash"] == "sha256:" + bare_hash
    assert entry["last_ingested_hash"] == "sha256:" + bare_hash


def test_should_skip_binary_with_legacy_bare_hash():
    """Bare and prefixed hashes compare equal during binary skip checks."""
    from crawler.crawl import _binary_unchanged

    file_reg = {
        "content_hash": "sha256:" + "e" * 64,
        "last_ingested_hash": "e" * 64,
    }
    assert _binary_unchanged(file_reg, force_recrawl=False) is True
