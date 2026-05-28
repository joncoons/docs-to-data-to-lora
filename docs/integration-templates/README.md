# Integration Templates

Planning templates for wiring this pipeline into external lifecycle systems.
These are not required for the core docs-to-data-to-lora flow; they are
implementation guides for teams that want a repeatable control-plane
integration around the existing Stage 2 and Stage 3 scripts.

## Available Templates

| Template | Purpose |
|---|---|
| [MLflow + NeMo orchestration](mlflow-nemo/README.md) | Use MLflow as the run, lineage, and promotion record while NeMo Microservices own dataset storage, customization, evaluation, and adapter artifacts. |
