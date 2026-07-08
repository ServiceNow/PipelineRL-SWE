#!/usr/bin/env bash
set -euo pipefail

# Experiment A: BCE + Bradley-Terry pairwise ranking auxiliary loss.
# Identical to the best baseline (r32, qkvo+mlp LoRA, 10 epochs) except:
#   --objective reward_bce_ranking
#   --ranking-aux-weight 1.0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_ranking_r32_qkvo_mlp_10epoch}
export OBJECTIVE=${OBJECTIVE:-reward_bce_ranking}
export DELTA_AUX_WEIGHT=${DELTA_AUX_WEIGHT:-0.0}
export RANKING_AUX_WEIGHT=${RANKING_AUX_WEIGHT:-1.0}
export DISCRIM_UPWEIGHT=${DISCRIM_UPWEIGHT:-1.0}
export LORA_R=${LORA_R:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_10epoch.sh"
