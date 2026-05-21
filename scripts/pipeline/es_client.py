"""ES client wrapper: scroll + kNN."""
from collections.abc import Iterable

from elasticsearch import Elasticsearch


def make_es_client(host: str, password: str) -> Elasticsearch:
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
    """kNN over `vector` field; exclude chunks whose content_url matches exclude_url."""
    must_not = [{"term": {"metadata.content_metadata.content_url": exclude_url}}]
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
                **(filter_extra or {}),
            }
        },
        "_source": ["text", "metadata"],
    }
    res = es.search(index=index, body=body)
    return res["hits"]["hits"]
