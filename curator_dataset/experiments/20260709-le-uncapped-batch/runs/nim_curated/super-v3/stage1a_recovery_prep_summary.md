# NIM LE Stage 1A Recovery Prep

Prepared at: 2026-07-10T11:38:52.875296+00:00

- Source checkpoint commit: `94927f3`
- Local archive: `.local-archive/20260710-nim-super-stage1a-pre-recovery`

## Latest Status Before Filtering

- complete: 414
- le_parse_failed: 44
- partial: 40

## Filtering

- Retryable passage IDs: 84
- KVP rows before filtering: 8169
- KVP rows removed: 520
- KVP rows after filtering: 7649
- Entailment rows after regeneration: 2459

## Removed Partial Rows

| removed_rows | passage |
| ---: | --- |
| 30 | `2024.findings-emnlp.713.pdf#c15` |
| 9 | `2024.findings-emnlp.713.pdf#c3` |
| 9 | `2309.06180.pdf#c16` |
| 12 | `Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf#c2` |
| 12 | `Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf#c8` |
| 15 | `https://docs.nvidia.com/nim/alchemi/alchemi-bgr/latest/custom-models.html#p0` |
| 12 | `https://docs.nvidia.com/nim/alchemi/alchemi-bgr/latest/release-notes.html#p0` |
| 22 | `https://docs.nvidia.com/nim/alchemi/alchemi-bgr/latest/security.html#p0` |
| 10 | `https://docs.nvidia.com/nim/alchemi/alchemi-bmd/latest/quickstart-guide.html#p0` |
| 19 | `https://docs.nvidia.com/nim/alchemi/alchemi-bmd/latest/security.html#p0` |
| 9 | `https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html#p0` |
| 5 | `https://docs.nvidia.com/nim/bionemo/alphafold2/latest/overview.html#p0` |
| 23 | `https://docs.nvidia.com/nim/bionemo/alphafold2/latest/prerequisites.html#p0` |
| 15 | `https://docs.nvidia.com/nim/bionemo/alphafold2/latest/quickstart-guide.html#p0` |
| 25 | `https://docs.nvidia.com/nim/bionemo/diffdock/latest/getting-started.html#p0` |
| 13 | `https://docs.nvidia.com/nim/bionemo/evo2/latest/index.html#p0` |
| 7 | `https://docs.nvidia.com/nim/bionemo/evo2/latest/overview.html#p0` |
| 7 | `https://docs.nvidia.com/nim/bionemo/proteinmpnn/latest/overview.html#p0` |
| 6 | `https://docs.nvidia.com/nim/bionemo/rfdiffusion/latest/overview.html#p0` |
| 9 | `https://docs.nvidia.com/nim/cosmos/latest/release-notes.html#p0` |
| 19 | `https://docs.nvidia.com/nim/ingestion/image-ocr/latest/security.html#p0` |
| 2 | `https://docs.nvidia.com/nim/ingestion/image-ocr/latest/troubleshoot.html#p0` |
| 27 | `https://docs.nvidia.com/nim/ingestion/object-detection/latest/getting-started.html#p0` |
| 7 | `https://docs.nvidia.com/nim/large-language-models/latest/turbo/support-matrix-turbo.html#p0` |
| 6 | `https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/index.html#p0` |
| 4 | `https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/prompt-template.html#p0` |
| 1 | `https://docs.nvidia.com/nim/maxine/eye-contact/latest/eula.html#p0` |
| 11 | `https://docs.nvidia.com/nim/maxine/eye-contact/latest/overview.html#p0` |
| 15 | `https://docs.nvidia.com/nim/medical/maisi/latest/performance.html#p0` |
| 38 | `https://docs.nvidia.com/nim/medical/vista3d/latest/getting-started.html#p0` |
| 12 | `https://docs.nvidia.com/nim/medical/vista3d/latest/support-matrix.html#p0` |
| 12 | `https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/security.html#p0` |
| 1 | `https://docs.nvidia.com/nim/nvclip/latest/EULA.html#p0` |
| 17 | `https://docs.nvidia.com/nim/vision-language-models/latest/api-reference.html#p0` |
| 11 | `https://docs.nvidia.com/nim/vision-language-models/latest/deploy-air-gap.html#p0` |
| 6 | `https://docs.nvidia.com/nim/vision-language-models/latest/get-started/index.html#p0` |
| 36 | `https://raw.githubusercontent.com/NVIDIA-AI-Blueprints/rag/main/README.md#p0` |
| 12 | `https://raw.githubusercontent.com/NVIDIA/nim-deploy/main/cloud-service-providers/google-cloud/gke/gcloud/README.md#p0` |
| 14 | `https://raw.githubusercontent.com/open-telemetry/opentelemetry-collector-contrib/main/exporter/zipkinexporter/README.md#p0` |

## Resume Behavior

The durable runner will retry all non-complete latest statuses. Rows for retryable passages were removed first so successful retries append replacement rows without duplicating prior partial rows.
