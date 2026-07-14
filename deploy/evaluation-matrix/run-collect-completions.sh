#!/usr/bin/env bash
# Run durable Stage 3 completion collection in-cluster.
#
# This helper keeps the collector source in the repo while avoiding a rebuild for
# local iteration: it publishes scripts/eval/collect_completions.py as a
# ConfigMap, then creates a generated-name Job from collect-completions-job.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS="${NS:-runai-rag}"
JOB_MANIFEST="$ROOT/deploy/evaluation-matrix/collect-completions-job.yaml"
COLLECTOR="$ROOT/scripts/eval/collect_completions.py"

echo "==> creating/updating ConfigMap completion-collector-script from $COLLECTOR"
kubectl create configmap completion-collector-script -n "$NS" \
  --from-file=collect_completions.py="$COLLECTOR" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "==> creating completion collection Job from $JOB_MANIFEST"
kubectl create -f "$JOB_MANIFEST"

echo "==> recent completion collector jobs"
kubectl get jobs -n "$NS" \
  -l app.kubernetes.io/name=completion-collector \
  --sort-by=.metadata.creationTimestamp
