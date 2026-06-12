#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TIMESTAMP=${TIMESTAMP:-$(date +%s)}

bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_reward_bce_r128_qkvo_mlp_input_only_10epoch.sh" &
bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_reward_bce_r64_qkvo_mlp_h2048_input_only_10epoch.sh" &
bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_reward_bce_r128_qkvo_mlp_h2048_input_only_10epoch.sh" &
wait
