# Deployment Reference: Kubernetes + Run.ai

One concrete deployment target. The methodology in this repo is
infrastructure-agnostic — anywhere you can run `rag-crawler` against an
Elasticsearch cluster and a NIM-compatible embedding endpoint will work.
This guide is included as a reference for users whose existing setup is
already RKE2 + Run.ai.

## Cluster assumptions

- RKE2 or k3s (any recent version) with the NIM Operator installed.
- Run.ai installed for GPU scheduling, or a similar fractional-GPU manager.
- An Elasticsearch cluster reachable from the cluster (ECK-managed or
  external).

## Components

| Component | Role in pipeline | GPU? |
|---|---|---|
| `rag-crawler` | hosts the BFS crawl loop, renders SPAs via Selenium | no |
| `nemoretriever-embedding-ms` (NIM) | embeds chunks for ES bulk write | yes (~4 GB) |
| Elasticsearch | stores crawled chunks for retrieval | no |
| `nim-llm`, `nim-vlm` (NIM) | not needed during crawl | – |
| `nemotron-parse` (NIM) | only needed for PDF crawls | – |

## Two GPU profiles

Profile selection depends on whether the crawl captures linked binaries
(`.pdf`/`.docx`/`.pptx`):

### Profile B — Full ingest (default)

Use whenever the curated scope includes any binary file types. This is the
**recommended default** because most vendor docs link to PDFs and the
SFT corpus is more complete with them included.

- `nim-llm` replicas: 0
- `nim-vlm` replicas: 1
- `nemotron-parse-v12` replicas: 7 (spread via Run.ai GPU memory)
- `nemoretriever-embedding-ms` replicas: 1
- `rag-crawler` replicas: 1

The 7-replica nemotron-parse setup is sized for batch PDF parsing during
Phase 3. If your corpus is small (e.g., < 50 PDFs total), drop to 2-3
replicas to free GPU memory.

### Profile A — HTML-only (optional)

Use only when you've explicitly excluded binary file types from the crawl
configuration. Minimal GPU footprint: just the embedding NIM (~4 GB).

- `nim-llm` replicas: 0
- `nim-vlm` replicas: 0
- `nemotron-parse-v12` replicas: 0
- `nemoretriever-embedding-ms` replicas: 1
- `rag-crawler` replicas: 1

For an SFT corpus, this profile is rarely the right choice — vendor PDFs
often contain detailed technical content (deployment guides, architecture
deep-dives) that won't be in the HTML.

Linked `.txt`/`.md`/`.rst` files (including GitHub-hosted READMEs) are
handled inline during the HTML crawl and do *not* require Profile B —
they go through the embedding NIM only, same as HTML chunks.

## Run.ai GPU placement

If your cluster uses Run.ai, place NIMs via `run.ai/gpu-memory` annotations
on the pod template. The memory request determines GPU selection (Run.ai
won't schedule onto a GPU that doesn't have the requested headroom):

| NIM | Suggested `run.ai/gpu-memory` |
|---|---|
| `nemoretriever-embedding-ms` | `"4000M"` |
| `nemoretriever-ranking-ms` (query-time only) | `"16000M"` |
| `nim-llm` (large LLM) | `"90000M"` |
| `nim-vlm` | `"32000M"` |
| `nemotron-parse-v12` | `"20000M"` per replica |

Without Run.ai, use standard `resources.limits["nvidia.com/gpu"]` or
device-plugin-specific scheduling.

## Switching profiles

Profile A activation (HTML-only):

```bash
NAMESPACE=<your-namespace>

# Bring down inference-time NIMs (NIM Operator manages these, so patch
# the NIMService not the Deployment)
kubectl patch nimservice nim-llm -n "$NAMESPACE" --type=merge \
  -p '{"spec":{"replicas":0}}'
kubectl patch nimservice nim-vlm -n "$NAMESPACE" --type=merge \
  -p '{"spec":{"replicas":0}}'

# Bring down PDF parser (standard Deployment)
kubectl scale deploy/nemotron-parse-v12 -n "$NAMESPACE" --replicas=0

# Ensure embedding NIM is up
kubectl get deploy -n "$NAMESPACE" nemoretriever-embedding-ms \
  -o jsonpath='{.status.readyReplicas}'
# Expected: 1

# Bring the crawler up
kubectl scale deploy/rag-crawler -n "$NAMESPACE" --replicas=1
kubectl rollout status -n "$NAMESPACE" deploy/rag-crawler --timeout=120s

# Verify the crawler can reach the embedding NIM
kubectl exec -n "$NAMESPACE" deploy/rag-crawler -- \
  curl -sf http://nemoretriever-embedding-ms:8000/v1/health/ready
```

Restore inference mode after the crawl completes (or before someone needs
the chat NIM):

```bash
# Scale embedding sources back to inference layout; order matters because
# Run.ai memory annotations cause GPU placement races if you bring nim-llm
# up before clearing nemotron-parse.
kubectl scale deploy/nemotron-parse-v12 -n "$NAMESPACE" --replicas=0
kubectl rollout status -n "$NAMESPACE" deploy/nemotron-parse-v12 --timeout=120s
kubectl patch nimservice nim-llm -n "$NAMESPACE" --type=merge \
  -p '{"spec":{"replicas":1}}'
# wait for nim-llm Running, then:
kubectl scale deploy/nemotron-parse-v12 -n "$NAMESPACE" --replicas=1
```

## Caveats

- **NIM Operator overrides.** NIMService `spec.replicas` is authoritative;
  scaling the underlying Deployment directly will be reverted. Always patch
  the NIMService.
- **Helm upgrades may conflict with `kubectl set` env overrides.** If your
  Helm chart sets OTEL env vars on a Deployment that you've patched
  directly, a subsequent `helm upgrade` may fail with a strategic-merge
  conflict. Workaround: apply env changes via `kubectl set env` or
  ConfigMap overlays, not via Helm chart values, on long-lived deployments.
- **Network topology matters.** rag-crawler talks to the embedding NIM over
  the cluster's pod network. Cross-node VXLAN issues (Flannel, Calico)
  can silently produce empty embeddings without erroring. If you see
  zero-vector chunks in ES, check `ping` between rag-crawler and the
  embedding NIM pod.

## Alternatives to this profile

- **Single-node Docker Compose.** For a small corpus (< few thousand URLs)
  on a workstation, run rag-crawler + a single embedding NIM + ES via
  `docker compose`. See the rag-crawler repo for a sample compose file.
- **Cloud-managed alternatives.** Replace ECK with managed Elasticsearch
  (e.g., Elastic Cloud); replace local NIMs with NVIDIA's hosted inference
  endpoints (`api.nvcf.nvidia.com`). The crawler config just points to
  different URLs.

The methodology and curation logic do not change across deployment
targets — only the operational mechanics.
