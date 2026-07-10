# NeMo Microservices LE Stage 1A Recovery 2 Prep

Prepared at: 2026-07-10T10:57:59.228484+00:00

- Recovery pass 1 checkpoint commit: `dd641a6`
- Local archive: `.local-archive/20260710-nemo-usvcs-super-stage1a-recovery1`

## Latest Status Before Filtering

- complete: 485
- partial: 1

## Filtering

- Retryable passage IDs: 1
- KVP rows before filtering: 9612
- KVP rows removed: 7
- KVP rows after filtering: 9605
- Entailment rows after regeneration: 3047

| removed_rows | passage |
| ---: | --- |
| 7 | `https://docs.nvidia.com/nemo/microservices/latest/evaluate/tutorials/index.html#p0` |

## Resume Behavior

The next durable resume should select only the remaining partial passage and append replacement rows if it completes.
