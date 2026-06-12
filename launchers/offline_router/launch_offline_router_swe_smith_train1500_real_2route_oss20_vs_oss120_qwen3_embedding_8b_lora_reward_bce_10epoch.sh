#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_reward_bce_10epoch}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_reward_bce_oss20_vs_oss120_10epoch}

export TIMESTAMP
export JOB_NAME
export OUTPUT_ROOT
export TRAIN_OUTPUT_DIR

# Original route 1 = OSS-20B, original route 3 = OSS-120B.
export TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-1,3}

# Plain BCE success prediction, matching the historical reward-BCE baseline.
export OBJECTIVE=${OBJECTIVE:-reward_bce}
export DELTA_AUX_WEIGHT=${DELTA_AUX_WEIGHT:-0.0}

# Historical reward-BCE capacity setting.
export LORA_R=${LORA_R:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}
export NUM_EPOCHS=${NUM_EPOCHS:-10}
export CHECKPOINT_EVERY_EPOCH=${CHECKPOINT_EVERY_EPOCH:-true}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_10epoch.sh"
