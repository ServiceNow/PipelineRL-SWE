#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export INPUT_MODE=${INPUT_MODE:-input_only}
export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_expert_cost_only_input_only_10epoch}
# Input-only routing has not paid the scout yet, so it needs a cost estimate
# for every selectable route, including route 0/scout.
export COST_ROUTE_IDXS=${COST_ROUTE_IDXS:-0,1,2,3}

bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_expert_cost_only_10epoch.sh"
