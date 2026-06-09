#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_cheapest_tier_hier_abstain_input_only_10epoch}
export INPUT_MODE=${INPUT_MODE:-input_only}
export OBJECTIVE=${OBJECTIVE:-route_classifier_hierarchical}
export CLASS_TARGET_MODE=${CLASS_TARGET_MODE:-cheapest_success_or_abstain}
export CLASS_WEIGHT_MODE=${CLASS_WEIGHT_MODE:-none}
export APPEND_ABSTAIN_CLASS=${APPEND_ABSTAIN_CLASS:-true}
export HIERARCHICAL_ANY_SUCCESS_WEIGHT=${HIERARCHICAL_ANY_SUCCESS_WEIGHT:-1.0}
export HIERARCHICAL_ROUTE_WEIGHT=${HIERARCHICAL_ROUTE_WEIGHT:-1.0}
exec "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_cheapest_tier_ce_10epoch.sh" "$@"
