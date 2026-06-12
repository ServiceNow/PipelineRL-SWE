#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TIMESTAMP=${TIMESTAMP:-$(date +%s)}

bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_2route_capacity_reward_parallel.sh" &
bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_expert_cost_only_normalized_r128_qkvo_mlp_input_only_10epoch.sh" &
wait
