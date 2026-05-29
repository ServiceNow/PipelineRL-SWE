#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
export TIMESTAMP

# Input-only reward and cost models. Because no scout attempt has run, this predicts scout cost too.
bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_7b_scout_qwen3_embedding_8b_lora_reward_mse_delta_input_only_5epoch.sh"
bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_4b_scout_qwen3_embedding_8b_lora_reward_mse_delta_input_only_5epoch.sh"
bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_7b_scout_qwen3_embedding_8b_lora_expert_cost_only_input_only_5epoch.sh"
bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_4b_scout_qwen3_embedding_8b_lora_expert_cost_only_input_only_5epoch.sh"
