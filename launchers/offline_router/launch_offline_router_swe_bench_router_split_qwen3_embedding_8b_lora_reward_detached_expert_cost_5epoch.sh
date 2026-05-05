#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
JOB_NAME=${JOB_NAME:-offline_router_swe_bench_router_split_qwen3_embedding_8b_lora_reward_detached_expert_cost_5epoch}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_reward_detached_expert_cost_5epoch}
COST_ROUTE_IDX=${COST_ROUTE_IDX:-1}
COST_GRADIENT_MODE=${COST_GRADIENT_MODE:-detached}

export JOB_NAME
export TIMESTAMP
export OUTPUT_ROOT
export TRAIN_OUTPUT_DIR
export COST_ROUTE_IDX
export COST_GRADIENT_MODE

bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_router_split_qwen3_embedding_8b_lora_reward_cost_5epoch.sh"
