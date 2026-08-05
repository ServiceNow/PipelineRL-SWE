#!/usr/bin/env bash
# Train an input-only abstention predictor on MATH CoT trajectories.
#
# This is the baseline that uses only the problem statement (no scout trace,
# no patch) to predict strong-model success — isolates task-difficulty signal
# from scout-quality signal.
#
# Required env vars:
#   MATH_OUTPUT_DIR  -- output dir from a launch_math_cot_collection.sh run,
#                       containing trajectories_train.jsonl, trajectories_eval.jsonl,
#                       labels_train.parquet, labels_eval.parquet
#
# Optional:
#   NUM_EPOCHS / LORA_R / BATCH_SIZE / etc.  -- passed through to launch_cot_abstention_predictor.sh
#
# Usage:
#   MATH_OUTPUT_DIR=/mnt/.../math_cot_trajectories_XYZ \
#   bash launch_math_input_only_abstention_predictor.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MATH_OUTPUT_DIR=${MATH_OUTPUT_DIR:?Need MATH_OUTPUT_DIR set to a launch_math_cot_collection.sh output dir}

echo "=== Launching MATH input-only abstention predictor ==="
echo "  Math output dir: ${MATH_OUTPUT_DIR}"
echo ""

INPUT_ONLY=true \
INCLUDE_THINKING=false \
LABEL_ROUTE_IDX=0 \
TRAJECTORIES_DIR="${MATH_OUTPUT_DIR}" \
TRAIN_PARQUET_DIR="${MATH_OUTPUT_DIR}/train" \
EVAL_PARQUET_DIR="${MATH_OUTPUT_DIR}/eval" \
  bash "${SCRIPT_DIR}/launch_cot_abstention_predictor.sh"
