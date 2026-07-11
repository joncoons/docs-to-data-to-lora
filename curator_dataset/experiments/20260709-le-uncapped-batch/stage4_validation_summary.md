# Stage 4 External-Judge Validation

Stage 4 validates finalized Stage 3 `training.jsonl` rows. Each sampled prompt/completion is joined back to `stage2_eval.jsonl` so Claude sees the original source context.

Judge: `azure/anthropic/claude-sonnet-4-6` via `https://inference-api.nvidia.com/v1`.

## Results

| Corpus | Source Rows | Sample | Grounded | Answer Fidelity | No Hallucination | All Three | Pass Rate | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NIM | 7905 | 100 | 99 | 100 | 100 | 99 | 0.99 | passed |
| NeMo Microservices | 7171 | 100 | 99 | 100 | 99 | 99 | 0.99 | passed |

## Failure Notes

### NIM

- Stage `1a` row from `https://docs.nvidia.com/nim/vision-language-models/latest/observability.html`
  - Question: What specific metrics are included under the KV Cache Count category in the NIM metrics table?
  - Reason: The answer incorrectly groups Running Count, Waiting Count, and Max Request Count under the 'KV Cache Count' category, when the source table only explicitly lists GPU Cache Usage under that category; the other metrics appear as separate unlabeled rows.

### NeMo Microservices

- Stage `1c` row from `https://docs.nvidia.com/nemo/microservices/latest/fine-tune/models/data-format.html`
  - Question: Summarize the key points of dataset format requirements for NVIDIA NeMo Microservices from the following documentation.
  - Reason: The answer incorrectly states that DPO formats require 'rejected_response' as a field name, when the source text uses 'rejected_response' correctly, but more critically the answer omits the 'OpenAI Chat With Tool' format (which requires both 'messages' and 'tools' fields) and incorrectly simplifies the DPO OpenAI Message Format by not distinguishing it from DPO Raw Request (the prompt field is an array of objects in one and a string in the other), and also incorrectly describes neg_doc as simply requiring 'neg_doc fields' without noting it is an array of strings.

## Artifacts

- NIM: `runs/nim_curated/super-v3/validation_report.json`, `validation_sample.jsonl`, `validation_judgments.jsonl`
- NeMo Microservices: `runs/nemo_usvcs_curated/super-v3/validation_report.json`, `validation_sample.jsonl`, `validation_judgments.jsonl`
