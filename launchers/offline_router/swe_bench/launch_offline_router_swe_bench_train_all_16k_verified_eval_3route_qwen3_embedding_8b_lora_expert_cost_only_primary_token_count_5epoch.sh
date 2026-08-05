#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export INPUT_MODE=${INPUT_MODE:-post_primary}
export INCLUDE_PRIMARY_OUTPUT_TOKEN_COUNT=${INCLUDE_PRIMARY_OUTPUT_TOKEN_COUNT:-true}
export JOB_NAME=${JOB_NAME:-offline_router_swe_bench_train_all_16k_verified_eval_3route_qwen3_embedding_8b_lora_expert_cost_only_primary_token_count_5epoch}

bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_3route_qwen3_embedding_8b_lora_expert_cost_only_5epoch.sh"
