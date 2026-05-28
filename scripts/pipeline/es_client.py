"""ES client wrapper: scroll + kNN."""
from collections.abc import Iterable
from typing import Any

try:
    from elasticsearch import Elasticsearch
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test envs
    Elasticsearch = Any  # type: ignore[misc, assignment]
    _MISSING_ELASTICSEARCH = True
else:
    _MISSING_ELASTICSEARCH = False


def make_es_client(host: str, password: str) -> Elasticsearch:
    if _MISSING_ELASTICSEARCH:
        raise RuntimeError(
            "The 'elasticsearch' package is required for live ES access. "
            "Install the project dependencies before running pipeline stages against ES."
        )
    return Elasticsearch(
        host,
        basic_auth=("elastic", password),
        verify_certs=False,
        ssl_show_warn=False,
        request_timeout=60,
    )


def scroll_all_chunks(es: Elasticsearch, index: str, page_size: int = 1000) -> Iterable[dict]:
    """Yield every hit in the index."""
    page = es.search(
        index=index,
        body={"query": {"match_all": {}}, "size": page_size,
              "_source": ["text", "metadata", "vector"]},
        scroll="5m",
    )
    scroll_id = page["_scroll_id"]
    try:
        while True:
            hits = page["hits"]["hits"]
            if not hits:
                break
            yield from hits
            page = es.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = page["_scroll_id"]
    finally:
        try:
            es.clear_scroll(scroll_id=scroll_id)
        except Exception:
            pass


def knn_search(es: Elasticsearch, index: str, vector: list[float],
               exclude_url: str, k: int = 6, num_candidates: int = 50,
               filter_extra: dict | None = None) -> list[dict]:
    """kNN over `vector` field; exclude chunks whose content_url matches exclude_url.

    `filter_extra` may contain `must`, `filter`, `should`, or `must_not` keys; they
    are merged into the bool clause. The exclude_url must_not entry is always
    preserved (caller's must_not entries are appended, not overwriting).
    """
    must_not = [{"term": {"metadata.content_metadata.content_url": exclude_url}}]
    extra = dict(filter_extra or {})
    # If caller passes their own must_not, merge with ours instead of overwriting
    if "must_not" in extra:
        must_not.extend(extra.pop("must_not"))
    body = {
        "knn": {
            "field": "vector",
            "query_vector": vector,
            "k": k,
            "num_candidates": num_candidates,
        },
        "query": {
            "bool": {
                "must_not": must_not,
                **extra,
            }
        },
        "_source": ["text", "metadata"],
    }
    res = es.search(index=index, body=body)
    return res["hits"]["hits"]
