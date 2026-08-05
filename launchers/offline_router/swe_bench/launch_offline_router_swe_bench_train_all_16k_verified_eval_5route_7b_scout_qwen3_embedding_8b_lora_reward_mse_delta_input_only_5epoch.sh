#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INPUT_MODE=${INPUT_MODE:-input_only}
JOB_NAME=${JOB_NAME:-offline_router_swe_bench_train_all_16k_verified_eval_5route_7b_scout_qwen3_embedding_8b_lora_reward_mse_delta_input_only_5epoch}
TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-0,1,2,3,4}
INPUT_MODE="${INPUT_MODE}" JOB_NAME="${JOB_NAME}" TARGET_ROUTE_IDXS="${TARGET_ROUTE_IDXS}" \
  bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_7b_scout_qwen3_embedding_8b_lora_reward_mse_delta_5epoch.sh"
