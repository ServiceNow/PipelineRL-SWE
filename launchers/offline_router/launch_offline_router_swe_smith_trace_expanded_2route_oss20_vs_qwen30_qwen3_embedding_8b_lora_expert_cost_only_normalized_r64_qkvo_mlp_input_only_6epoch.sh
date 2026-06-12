#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export JOB_NAME=${JOB_NAME:-offline_router_swe_smith_trace_expanded_2route_oss20_vs_qwen30_qwen3_embedding_8b_lora_expert_cost_only_normalized_r64_qkvo_mlp_input_only_6epoch}
export TRACE_COST_DATASET_DIR=${TRACE_COST_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_cost_4route_expanded_1781073985/collect}
export PREPARE_TRACE_DATASET=${PREPARE_TRACE_DATASET:-false}

export INPUT_MODE=${INPUT_MODE:-input_only}
export INCLUDE_PRIMARY_OUTPUT_TOKEN_COUNT=${INCLUDE_PRIMARY_OUTPUT_TOKEN_COUNT:-false}

# Original route 1 = OSS-20B, original route 2 = Qwen3-Coder-30B-A3B.
export TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-1,2}
export COST_ROUTE_IDXS=${COST_ROUTE_IDXS:-1,2}
export NUM_EPOCHS=${NUM_EPOCHS:-6}

export LORA_R=${LORA_R:-64}
export LORA_ALPHA=${LORA_ALPHA:-128}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
export MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE:-1024}

if [[ ! -f "${TRACE_COST_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing expanded trace-cost dataset: ${TRACE_COST_DATASET_DIR}/metadata.json" >&2
  echo "Run: bash ${SCRIPT_DIR}/materialize_swe_smith_trace_cost_4route_expanded_1781073985.sh" >&2
  exit 1
fi

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_trace_4route_qwen3_embedding_8b_lora_expert_cost_only_normalized_10epoch.sh"
