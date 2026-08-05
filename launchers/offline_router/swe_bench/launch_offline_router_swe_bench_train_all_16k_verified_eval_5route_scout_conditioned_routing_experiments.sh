#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
export TIMESTAMP

# Reward and expert-cost models for post-scout routing. Scout is a selectable stop route.
bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_7b_scout_qwen3_embedding_8b_lora_reward_mse_delta_5epoch.sh"
bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_4b_scout_qwen3_embedding_8b_lora_reward_mse_delta_5epoch.sh"
bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_7b_scout_qwen3_embedding_8b_lora_expert_cost_only_5epoch.sh"
bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_4b_scout_qwen3_embedding_8b_lora_expert_cost_only_5epoch.sh"
