"""Tests for deterministic provenance IDs and hashes."""

from scripts.pipeline.provenance import (
    sha256_text,
    source_chunk_id_for_text,
    source_revision_id_for_text,
    stable_id,
)


def test_sha256_text_has_prefix_and_is_deterministic():
    first = sha256_text("same text")
    second = sha256_text("same text")
    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64


def test_stable_id_is_order_sensitive_and_prefixed():
    one = stable_id("sample", "a", "b")
    two = stable_id("sample", "a", "b")
    different = stable_id("sample", "b", "a")
    assert one == two
    assert one != different
    assert one.startswith("sample_")


def test_source_ids_change_when_content_changes():
    url = "https://docs.example.com/page"
    rev_a = source_revision_id_for_text(url, "old")
    rev_b = source_revision_id_for_text(url, "new")
    assert rev_a != rev_b

    chunk_a = source_chunk_id_for_text(rev_a, f"{url}#p0", "old")
    chunk_b = source_chunk_id_for_text(rev_a, f"{url}#p0", "new")
    assert chunk_a != chunk_b
