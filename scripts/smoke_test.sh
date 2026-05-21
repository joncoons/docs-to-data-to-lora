#!/usr/bin/env bash
# End-to-end smoke test of the Stage 2 pipeline on a 50-passage subset.
#
# Prerequisites:
#   - super-120b NIM running at nim-llm-super-120b-bw.runai-rag:8000
#   - ES nim_curated index populated
#   - K8s secret nvidia-inference-key present in runai-rag (for Claude Sonnet judge)
#   - Optional: NeMo Data Designer service deployed in nemo-peft (gap-fill is
#     skipped if not — pipeline still produces a usable dataset)
#
# Runtime: ~30-90 min depending on super-120b throughput.

set -euo pipefail
OUT=/mnt/nvme2/peft/datasets/v2/smoke_nim
mkdir -p "$OUT"

# --- Stage 0: full corpus prep ---
python scripts/build_v2_dataset.py \
    --collection nim_curated \
    --output "$OUT" \
    --stage 0

# Sub-sample passages.jsonl to 50 lines for the rest of the smoke test
mv "$OUT/passages.jsonl" "$OUT/passages.full.jsonl"
shuf -n 50 "$OUT/passages.full.jsonl" > "$OUT/passages.jsonl"

# --- Stages 1a → 4 against the 50-passage subset ---
for stage in 1a 1b 1c 1.5 2 3 4; do
    echo "=== Smoke test: stage $stage ==="
    python scripts/build_v2_dataset.py \
        --collection nim_curated \
        --output "$OUT" \
        --stage "$stage" \
        --resume
done

# --- Verify outputs ---
TRAIN_COUNT=$(wc -l < "$OUT/training.jsonl")
VAL_COUNT=$(wc -l < "$OUT/validation.jsonl")
echo "Smoke test complete: $TRAIN_COUNT train, $VAL_COUNT val"

# Sanity: ratio should be ~9:1
RATIO=$(python -c "print(round($TRAIN_COUNT / max($VAL_COUNT, 1), 1))")
echo "train:val ratio = $RATIO  (expected ≈ 9.0)"

# Sanity: validation report passed?
python -c "
import json, sys
r = json.load(open('$OUT/validation_report.json'))
print(f\"validation pass_rate={r['pass_rate']*100:.1f}% (threshold={r['threshold']*100:.0f}%) passed={r['passed']}\")
sys.exit(0 if r['passed'] else 1)
"
