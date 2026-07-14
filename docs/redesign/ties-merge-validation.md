# TIES Merge Checkpoint Validation

This note records the current structural validation status for the Stage 3
Nemotron Nano TIES-merged LoRA adapters.

The merge script requires `torch` and `safetensors`. It did not run under the
system Python. It ran under the `nat` conda environment used by the MoE
orchestrators:

```text
<USER_HOME>/anaconda3/envs/nat/bin/python3
torch: 2.12.0
safetensors: 0.7.0
```

The current system Python does not have those dependencies:

```text
/usr/bin/python3
torch: missing
safetensors: missing
```

## Structural Inspection

Use the torch-free inspector:

```bash
python3 scripts/stage3/inspect_adapter_checkpoint.py \
  <ARTIFACT_ROOT>/checkpoints/lora/lora-nim-nemotron-nano-30b-r16

python3 scripts/stage3/inspect_adapter_checkpoint.py \
  <ARTIFACT_ROOT>/checkpoints/lora/lora-nemo-usvcs-nemotron-nano-30b-r16
```

### `lora-nim-nemotron-nano-30b-r16`

Status: `warn`

Required files:

- `adapter_config.json`: present
- `adapter_model.safetensors`: present

Tensor summary:

- tensor count: 12,008
- LoRA A tensors: 6,004
- LoRA B tensors: 6,004
- non-LoRA tensors: 0
- dtype counts: `BF16: 12008`
- total elements: 441,936,896

Warnings:

- `adapter_config.json` has no `base_model_name_or_path`
- `adapter_config.json` has no `task_type`
- `target_modules` is a small generic list of 8 module names

Interpretation: the safetensors file is structurally consistent, but loadability
depends on NIM/PEFT resolving the generic target module names against the
Nemotron Nano base model.

### `lora-nemo-usvcs-nemotron-nano-30b-r16`

Status: `warn`

Required files:

- `adapter_config.json`: present
- `adapter_model.safetensors`: present

Auxiliary files:

- `automodel_peft_config.json`: present
- `tokenizer.json`: present
- `tokenizer_config.json`: present
- `special_tokens_map.json`: present
- `chat_template.jinja`: present

Tensor summary:

- tensor count: 12,031
- LoRA A tensors: 6,004
- LoRA B tensors: 6,004
- non-LoRA tensors: 23
- dtype counts: `BF16: 12008`, `F32: 23`
- total elements: 441,939,840

Warnings:

- safetensors contains 23 non-LoRA tensors

Interpretation: the adapter config is more complete and model-specific than the
NIM corpus adapter. The 23 non-LoRA F32 tensors appear to be MoE gate
correction-bias tensors; they may be expected for this architecture, but this
still needs a live NIM load test.

## Viability Definition

Structural inspection is not enough. A merged adapter is viable only after:

1. NIM starts successfully with the adapter path in `NIM_PEFT_SOURCE`.
2. `/v1/models` exposes the adapter model ID.
3. A one-prompt `/v1/chat/completions` request succeeds.
4. The response is not an immediate loader/runtime error.
5. The same base model can still answer without the adapter, if the target setup
   serves both base and adapter routes.

## Next Live-Load Test

Run one adapter at a time against the Nemotron Nano NIM deployment. Suggested
test prompts:

```text
For the NIM adapter:
What does NIM use LoRA adapters for, and how are they exposed to inference?

For the NeMo Microservices adapter:
How do NeMo Customizer, Entity Store, and Data Store relate during a LoRA customization job?
```

Record:

- NIM deployment/config used
- adapter path
- adapter model ID shown by `/v1/models`
- request payload
- HTTP status
- first response
- pod logs around adapter loading

Do not register either TIES-merged adapter as validated until this live-load
test passes.
