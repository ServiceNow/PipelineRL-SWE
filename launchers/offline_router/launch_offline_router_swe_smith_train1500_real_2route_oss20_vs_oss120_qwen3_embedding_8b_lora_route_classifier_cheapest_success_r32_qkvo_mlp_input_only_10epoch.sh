#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_route_classifier_cheapest_success_r32_qkvo_mlp_input_only_10epoch}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
export TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_route_classifier_cheapest_success_oss20_vs_oss120_r32_qkvo_mlp_input_only_10epoch}

export INPUT_MODE=${INPUT_MODE:-input_only}
export OBJECTIVE=${OBJECTIVE:-route_classifier}
export CLASS_TARGET_MODE=${CLASS_TARGET_MODE:-cheapest_success}
export CLASS_SUCCESS_THRESHOLD=${CLASS_SUCCESS_THRESHOLD:-0.5}
export CLASS_WEIGHT_MODE=${CLASS_WEIGHT_MODE:-inverse_freq}
export DELTA_AUX_WEIGHT=${DELTA_AUX_WEIGHT:-0.0}
export LORA_R=${LORA_R:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_reward_bce_10epoch.sh"
