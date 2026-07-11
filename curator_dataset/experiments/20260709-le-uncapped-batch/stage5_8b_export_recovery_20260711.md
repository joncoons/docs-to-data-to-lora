# Stage 5 8B r32 Export Recovery - 2026-07-11

## Issue

Both 8B r32 LoRA SFT jobs completed their training loops, but the Customizer
post-training entity handler failed during adapter publication because the
handler image pull hit an `nvcr.io` DNS timeout:

`nvcr.io/nvidia/nemo-microservices/nds-v2-huggingface-cli:25.12`

Customizer marked the r32 jobs `cancelled` after the entity handler exceeded its
pending deadline, even though training had already reached 5/5 epochs and 100%.

## Recovery

The export image was pre-pulled with short-lived pods on both GPU nodes:

- `ubuntu-local-dev`
- `ubuntu2`

The completed trained adapter directories were still present on
`peft-workspace-pvc`:

- `/pvc/cust-bbbuteyxhnw2zjzgzhqmpy/trained/`
- `/pvc/cust-8ge3t81ypq76fhk3b4b7ts/trained/`

Two export-only retry Jobs were run. They did not retrain and did not request
GPUs:

| Adapter | Retry Job | Result |
| --- | --- | --- |
| NIM r32 | `export-r32-nim-8b-retry-20260711` | uploaded |
| NeMo Microservices r32 | `export-r32-nemo-8b-retry-20260711` | uploaded |

## Final Artifact Status

| Adapter | Customizer Job | Customizer terminal status | Entity artifact status | Files URL |
| --- | --- | --- | --- | --- |
| NIM r16 | `cust-VM3mbWVx7FcPdtTG84UiJs` | `completed` | `upload_completed` | `hf://default/lora-nim-le-super-v3-e5-llama-3.1-8b-r16-20260711@cust-VM3mbWVx7FcPdtTG84UiJs` |
| NeMo r16 | `cust-5HqsCLjwYzyy2AyiW3EiNW` | `completed` | `upload_completed` | `hf://default/lora-nemo-usvcs-le-super-v3-e5-llama-3.1-8b-r16-20260711@cust-5HqsCLjwYzyy2AyiW3EiNW` |
| NIM r32 | `cust-BBbUtEYXHNW2zjzGZhqMpY` | `cancelled` after training completed | `upload_completed` | `hf://default/lora-nim-le-super-v3-e5-llama-3.1-8b-r32-20260711@cust-BBbUtEYXHNW2zjzGZhqMpY` |
| NeMo r32 | `cust-8GE3t81yPq76FhK3b4b7Ts` | `cancelled` after training completed | `upload_completed` | `hf://default/lora-nemo-usvcs-le-super-v3-e5-llama-3.1-8b-r32-20260711@cust-8GE3t81yPq76FhK3b4b7Ts` |

Conclusion: no 8B r32 retraining is required. Treat the Customizer r32 terminal
status as an export-handler failure, not a training failure; use Entity Store
artifact status for adapter availability.
