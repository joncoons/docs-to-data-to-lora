# Local Findings

These notes capture local experiment observations that should remain in this
workspace until explicitly promoted.

## Kimi Judge Endpoint Comparison

Date: 2026-06-02

Context:

- Previous judge endpoint: local MAAS-hosted Kimi K2.6 endpoint.
- Current judge endpoint: NVIDIA Inference API chat-completions endpoint.
- Current model: `nvidia/moonshotai/kimi-k2.6`.
- Do not store or commit ephemeral API keys. The Kubernetes Secret
  `runai-rag/kimi-judge-api` was updated locally.

Observed results:

| Workload | Previous endpoint | NVIDIA endpoint | Delta |
|---|---:|---:|---:|
| Pairwise rows/min at 8 judge slots | ~5.4 | ~8.3 | +54% |
| Pairwise judge calls/min at 8 judge slots | ~10.8 | ~16.6 | +54% |
| Pairwise judge tokens/min at 8 judge slots | ~46k | ~67k | +45% |
| Pairwise failed-row behavior | High retry exhaustion under load | 0 failed rows in initial Stage 2 sample | Improved |

49B single-axis result:

- Previous endpoint returned immediate `503 Service Unavailable` responses and
  produced no usable 49B scoring progress.
- NVIDIA endpoint completed the 49B single-axis run successfully at
  `--concurrency 8`.
- Final 49B score coverage: 1,001 / 1,002 rows scored.
- Failed rows: 1.
- Retry-exhausted rows: 0.
- Recorded total tokens: 7,944,139.

Operational finding:

- The NVIDIA endpoint is materially faster and more reliable for this workload.
- For pairwise scoring, use clean run IDs when switching judge endpoints so old
  and new judge behavior is not mixed in one result set.
- Stage 2 pairwise should run sequentially by corpus at `--concurrency 8`:
  start NIM first, release NeMo only after NIM succeeds.
- Continue tracking failed rows, retry counts, and token usage in each
  `summary.json` before promoting results into final reports.

