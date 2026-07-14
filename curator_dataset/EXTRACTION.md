# NIM HTML URL-document extraction

The experiment-local extractor reads the `nim_curated` Elasticsearch index,
keeps HTML-like text chunks only, sorts them by `(chunk_index, _id)`, and emits
one concatenated document per resolved content URL.

```bash
.venv/bin/python curator_dataset/src/build_url_documents.py \
  --es-host https://10.43.233.46:9200
```

The host may instead be supplied with `PIPELINE_ES_HOST`. If
`PIPELINE_ES_PASSWORD` is unset, the command reads the existing ECK password
from Kubernetes Secret `rag-eck-elasticsearch-es-elastic-user` in `runai-rag`.
It never prints or stores that credential.

## 2026-06-29 result

- Raw ES hits: 2,086
- Accepted HTML chunks: 2,018
- Excluded non-HTML chunks: 68
- URL documents: 453
- Documents containing repeated exact chunk text: 225
- Repeated exact chunk-text instances: 777
- Output SHA-256: `97d40a1d2295177abc62f5f3d8987024339f8951f2216be532c20046ea337770`

Outputs:

```text
curator_dataset/data/nim_curated/url_documents.jsonl
curator_dataset/data/nim_curated/url_documents.manifest.json
```

Exact duplicate chunk text is retained and reported so reconstruction remains
auditable. The largest records include acknowledgement pages over 100,000
words; duplicate handling, legal-page filtering, minimum length, and Curator
segmentation therefore remain explicit preprocessing decisions for the next
phase. No Curator generation workload has been run yet.
