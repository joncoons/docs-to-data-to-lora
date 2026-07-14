#!/usr/bin/env bash
# Deploy (or update) the rag-oai-proxy in the runai-rag namespace.
#
# Steps:
#   1. (Re)create the ConfigMap from deploy/rag-oai-proxy/proxy.py
#   2. Apply the Deployment + Service from manifests.yaml
#   3. Restart the rollout so the new ConfigMap is picked up
#
# Idempotent — safe to re-run after editing proxy.py.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
NS=runai-rag

echo "==> (re)creating ConfigMap rag-oai-proxy-script from $DIR/proxy.py"
kubectl create configmap rag-oai-proxy-script -n "$NS" \
  --from-file=proxy.py="$DIR/proxy.py" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "==> applying Deployment + Service from $DIR/manifests.yaml"
kubectl apply -f "$DIR/manifests.yaml"

echo "==> restarting rollout to pick up new ConfigMap (no-op if first deploy)"
kubectl rollout restart deployment/rag-oai-proxy -n "$NS" || true

echo "==> waiting for Pod Ready (timeout 120s)"
kubectl rollout status deployment/rag-oai-proxy -n "$NS" --timeout=120s

echo "==> done. service: http://rag-oai-proxy.${NS}:8080"
kubectl get pods -n "$NS" -l app=rag-oai-proxy
