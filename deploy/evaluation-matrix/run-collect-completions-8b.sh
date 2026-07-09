#!/usr/bin/env bash
# Run durable Stage 3 completion collection for Llama 3.1 8B targets.
#
# Set RUN_ID to align this job with another active completion matrix:
#   RUN_ID=20260529T142723Z deploy/evaluation-matrix/run-collect-completions-8b.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS="${NS:-runai-rag}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
JOB_MANIFEST="$ROOT/deploy/evaluation-matrix/collect-completions-8b-job.yaml"
COLLECTOR="$ROOT/scripts/eval/collect_completions.py"
TMP_MANIFEST="$(mktemp)"
trap 'rm -f "$TMP_MANIFEST"' EXIT

echo "==> creating/updating ConfigMap completion-collector-script from $COLLECTOR"
kubectl create configmap completion-collector-script -n "$NS"   --from-file=collect_completions.py="$COLLECTOR"   --dry-run=client -o yaml | kubectl apply -f -

python3 - "$JOB_MANIFEST" "$TMP_MANIFEST" "$RUN_ID" <<'PYINJECT'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
run_id = sys.argv[3]
text = src.read_text(encoding='utf-8')
needle = """            - name: COMPLETIONS_OUTPUT_ROOT
              value: /mnt/nvme2/peft/evals/completions
"""
replacement = needle + f"""            - name: RUN_ID
              value: "{run_id}"
"""
if needle not in text:
    raise SystemExit('env insertion point not found')
dst.write_text(text.replace(needle, replacement, 1), encoding='utf-8')
PYINJECT

echo "==> creating 8B completion collection Job from $JOB_MANIFEST with RUN_ID=$RUN_ID"
kubectl create -f "$TMP_MANIFEST"

echo "==> recent 8B completion collector jobs"
kubectl get jobs -n "$NS"   -l app.kubernetes.io/name=completion-collector,app.kubernetes.io/component=llama-3.1-8b   --sort-by=.metadata.creationTimestamp
