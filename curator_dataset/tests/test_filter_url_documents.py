from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from filter_url_documents import filter_rows  # noqa: E402


def test_filter_rows_excludes_acknowledgements_and_eula():
    rows = [
        {"document_id": "a", "url": "https://example.test/acknowledgements.html"},
        {"document_id": "b", "url": "https://example.test/EULA.html"},
        {"document_id": "c", "url": "https://example.test/deploy.html"},
    ]
    retained, excluded = filter_rows(rows)
    assert [row["document_id"] for row in retained] == ["c"]
    assert [row["document_id"] for row in excluded] == ["a", "b"]
