#!/usr/bin/env bash
set -euo pipefail

# Experiment: 120B full output (with thinking traces) as encoder input.
# Identical to the BCE baseline but input_mode=post_120b — uses the 120B
# model's full output (including <think> blocks where present) instead of
# the 4B scout output. Tests whether 120B thinking traces improve abstention
# ROC AUC vs the 4B-trace baseline (0.720).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_post120b_r32_qkvo_mlp_10epoch}
export OBJECTIVE=${OBJECTIVE:-reward_bce}
export INPUT_MODE=${INPUT_MODE:-post_120b}
export DELTA_AUX_WEIGHT=${DELTA_AUX_WEIGHT:-0.0}
export RANKING_AUX_WEIGHT=${RANKING_AUX_WEIGHT:-0.0}
export DISCRIM_UPWEIGHT=${DISCRIM_UPWEIGHT:-1.0}
export LORA_R=${LORA_R:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_10epoch.sh"
