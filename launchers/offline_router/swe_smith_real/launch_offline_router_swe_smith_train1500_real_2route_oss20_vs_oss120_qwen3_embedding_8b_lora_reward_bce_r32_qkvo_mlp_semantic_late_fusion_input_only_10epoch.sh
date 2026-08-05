#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_semantic_late_fusion_input_only_10epoch}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
export TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_reward_bce_oss20_vs_oss120_r32_qkvo_mlp_semantic_late_fusion_input_only_10epoch}
export REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_file_context_1780639659/collect}
export INPUT_MODE=${INPUT_MODE:-input_only}
export EMBEDDING_INPUT_LAYOUT=${EMBEDDING_INPUT_LAYOUT:-semantic_late_fusion}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch.sh"
