#!/usr/bin/env bash
# Train a no-CoT abstention predictor using Qwen3-4B-Instruct patches.
#
# The Instruct patches already exist in the 4-route real-label parquet as
# primary_output_text (route 0). This script extracts them locally and then
# submits the training job.
#
# Optional env vars:
#   REAL_LABEL_DATASET_DIR  -- 4-route parquet collection dir (default below)
#   LABEL_ROUTE_IDX         -- target route for labels (default: 3 = 120B)
#   NUM_EPOCHS / LORA_R / BATCH_SIZE / etc.  -- passed through to launch_cot_abstention_predictor.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3

TIMESTAMP=$(date +%s)
REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
TRAJ_DIR=/mnt/llmd/results/exps/aristides/reason/instruct_patches_trajectories_${TIMESTAMP}

echo "=== Extracting Qwen3-4B-Instruct patches from parquet ==="
"${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/extract_instruct_patches_as_trajectories.py" \
  --train-parquet-dir "${REAL_LABEL_DATASET_DIR}/train" \
  --eval-parquet-dir  "${REAL_LABEL_DATASET_DIR}/eval" \
  --output-dir        "${TRAJ_DIR}"

echo ""
echo "Trajectories written to: ${TRAJ_DIR}"
echo ""
echo "=== Launching no-CoT predictor training ==="

INCLUDE_THINKING=false \
TRAJECTORIES_DIR="${TRAJ_DIR}" \
  bash "${SCRIPT_DIR}/launch_cot_abstention_predictor.sh"
