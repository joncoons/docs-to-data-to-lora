#!/usr/bin/env bash
# End-to-end smoke test of the Stage 2 pipeline on a 50-passage subset.
#
# Prerequisites:
#   - super-120b NIM running and reachable
#   - ES nim_curated index populated
#   - K8s secret nvidia-inference-key present in runai-rag (for Stage 4 external judge)
#   - PIPELINE_ES_HOST and PIPELINE_NIM_ENDPOINTS env vars set if running from
#     a host that can't resolve cluster service DNS names.
#
# Runtime: ~30-60 min depending on super-120b throughput.

set -euo pipefail
OUT=<DATASET_ROOT>/smoke_nim
mkdir -p "$OUT"

<USER_HOME>/anaconda3/envs/nat/bin/python3 scripts/build_v2_dataset.py \
    --collection nim_curated \
    --output "$OUT" \
    --stage all \
    --max-passages 50

# Verify outputs
TRAIN_COUNT=$(wc -l < "$OUT/training.jsonl")
VAL_COUNT=$(wc -l < "$OUT/validation.jsonl")
echo "Smoke test complete: $TRAIN_COUNT train, $VAL_COUNT val"

RATIO=$(<USER_HOME>/anaconda3/envs/nat/bin/python3 -c "print(round($TRAIN_COUNT / max($VAL_COUNT, 1), 1))")
echo "train:val ratio = $RATIO  (expected ≈ 9.0)"

<USER_HOME>/anaconda3/envs/nat/bin/python3 -c "
import json, sys
r = json.load(open('$OUT/validation_report.json'))
print(f\"validation pass_rate={r['pass_rate']*100:.1f}% (threshold={r['threshold']*100:.0f}%) passed={r['passed']}\")
sys.exit(0 if r['passed'] else 1)
"
