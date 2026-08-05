#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP="${TIMESTAMP:-$(date +%s)}"
N_INSTANCES="${N_INSTANCES:-1500}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-router_analysis/aws_eval_packages/swebench_train_real_label_subset_4route_${N_INSTANCES}_${TIMESTAMP}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" pipelinerl/swe/scripts/offline_router/export_real_label_subset_from_router_collection.py \
  --output-dir "$OUTPUT_DIR" \
  --n-instances "$N_INSTANCES" \
  --seed "$SEED" \
  --random-fraction 0.5 \
  --disagreement-fraction 0.3 \
  --make-tarball \
  "$@"

echo "Package: $OUTPUT_DIR"
echo "Tarball: ${OUTPUT_DIR}.tar.gz"
