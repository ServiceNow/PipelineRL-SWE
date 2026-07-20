#!/usr/bin/env bash
set -euo pipefail

# Experiment: 120B output with thinking traces stripped as encoder input.
# Identical to the post_120b variant but strips <think>...</think> blocks
# before encoding. The gap between post_120b and post_120b_no_think isolates
# the contribution of thinking trace content to abstention prediction quality.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_post120b_no_think_r32_qkvo_mlp_10epoch}
export OBJECTIVE=${OBJECTIVE:-reward_bce}
export INPUT_MODE=${INPUT_MODE:-post_120b_no_think}
export DELTA_AUX_WEIGHT=${DELTA_AUX_WEIGHT:-0.0}
export RANKING_AUX_WEIGHT=${RANKING_AUX_WEIGHT:-0.0}
export DISCRIM_UPWEIGHT=${DISCRIM_UPWEIGHT:-1.0}
export LORA_R=${LORA_R:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_10epoch.sh"
