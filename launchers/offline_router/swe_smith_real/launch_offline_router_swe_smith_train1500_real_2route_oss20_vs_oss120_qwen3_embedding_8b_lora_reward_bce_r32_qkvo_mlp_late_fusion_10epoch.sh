#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_late_fusion_10epoch}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
export TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_reward_bce_oss20_vs_oss120_r32_qkvo_mlp_late_fusion_10epoch}
export INPUT_MODE=${INPUT_MODE:-post_primary}
export EMBEDDING_INPUT_LAYOUT=${EMBEDDING_INPUT_LAYOUT:-late_fusion}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch.sh"
