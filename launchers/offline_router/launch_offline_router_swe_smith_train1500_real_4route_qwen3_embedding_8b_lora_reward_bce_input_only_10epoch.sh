#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export INPUT_MODE=${INPUT_MODE:-input_only}
export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_input_only_10epoch}
bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_10epoch.sh"
