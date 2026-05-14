#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export INPUT_MODE=${INPUT_MODE:-input_only}
export JOB_NAME=${JOB_NAME:-offline_router_swe_bench_train_all_16k_verified_eval_3route_qwen3_embedding_8b_lora_reward_mse_delta_input_only_5epoch}

bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_3route_qwen3_embedding_8b_lora_reward_mse_delta_5epoch.sh"
