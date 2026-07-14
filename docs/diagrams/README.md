# Diagrams

## Stage 2 pipeline architecture

`stage-2-pipeline.excalidraw` is the source; render to PNG via `render-diagram.sh`.

### Excalidraw+ API key

If you have an Excalidraw+ subscription and want to push directly to your workspace,
store the API key as a k8s secret in the `runai-rag` namespace:

```bash
kubectl create secret generic excalidraw-api-key \
  --from-literal=api-key="<YOUR_KEY>" \
  -n runai-rag
```

The render script reads it via:

```bash
kubectl get secret excalidraw-api-key -n runai-rag \
  -o jsonpath='{.data.api-key}' | base64 -d
```

(Same pattern as `llm-api-key` in the pipeline scripts.)
