#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_expert_cost_only_normalized_r128_qkvo_mlp_input_only_10epoch}
export TRACE_COST_DATASET_DIR=${TRACE_COST_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_cost_4route_expanded_1781073985/collect}
export PREPARE_TRACE_DATASET=${PREPARE_TRACE_DATASET:-false}
export INPUT_MODE=${INPUT_MODE:-input_only}
export INCLUDE_PRIMARY_OUTPUT_TOKEN_COUNT=${INCLUDE_PRIMARY_OUTPUT_TOKEN_COUNT:-false}
export TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-0,1,2,3}
export COST_ROUTE_IDXS=${COST_ROUTE_IDXS:-0,1,2,3}
export LORA_R=${LORA_R:-128}
export LORA_ALPHA=${LORA_ALPHA:-256}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

if [[ ! -f "${TRACE_COST_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing expanded trace-cost dataset: ${TRACE_COST_DATASET_DIR}/metadata.json" >&2
  echo "Run: bash ${SCRIPT_DIR}/materialize_swe_smith_trace_cost_4route_expanded_1781073985.sh" >&2
  exit 1
fi

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_trace_4route_qwen3_embedding_8b_lora_expert_cost_only_normalized_10epoch.sh"
