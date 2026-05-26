"""Tests for the ES client wrapper."""
from unittest.mock import MagicMock
from scripts.pipeline.es_client import scroll_all_chunks, knn_search


def test_scroll_all_chunks_yields_each_hit():
    es = MagicMock()
    es.search.return_value = {
        "_scroll_id": "sid",
        "hits": {"hits": [
            {"_id": "1", "_source": {"text": "a"}},
            {"_id": "2", "_source": {"text": "b"}},
        ]},
    }
    es.scroll.side_effect = [
        {"_scroll_id": "sid", "hits": {"hits": []}}
    ]
    es.clear_scroll = MagicMock()

    hits = list(scroll_all_chunks(es, index="nim_curated"))
    assert len(hits) == 2
    assert hits[0]["_id"] == "1"
    es.clear_scroll.assert_called_once_with(scroll_id="sid")


def test_knn_search_excludes_seed_url():
    es = MagicMock()
    es.search.return_value = {"hits": {"hits": [
        {"_id": "n1", "_source": {"text": "neighbor1"}, "_score": 0.9},
    ]}}

    neighbors = knn_search(
        es, index="nim_curated",
        vector=[0.1, 0.2, 0.3, 0.4],
        exclude_url="https://x.com/seed",
        k=6, num_candidates=50,
    )

    assert len(neighbors) == 1
    call_kwargs = es.search.call_args.kwargs
    body = call_kwargs["body"]
    # Confirm the exclude_url is in the must_not filter
    must_not = body["query"]["bool"]["must_not"]
    assert any("https://x.com/seed" in str(m) for m in must_not)


def test_knn_search_filter_extra_merges_with_must_not():
    """Caller can pass filter_extra={"must_not": [...]} without overwriting exclude_url."""
    from scripts.pipeline.es_client import knn_search
    es = MagicMock()
    es.search.return_value = {"hits": {"hits": []}}

    knn_search(es, index="x", vector=[0.1], exclude_url="https://seed.com/p",
                k=6, num_candidates=50,
                filter_extra={"must_not": [{"term": {"metadata.bad_field": "bad"}}]})

    body = es.search.call_args.kwargs["body"]
    must_not = body["query"]["bool"]["must_not"]
    # Should contain BOTH the exclude_url and the caller's extra must_not
    assert len(must_not) == 2
    assert any("https://seed.com/p" in str(m) for m in must_not)
    assert any("bad_field" in str(m) for m in must_not)
