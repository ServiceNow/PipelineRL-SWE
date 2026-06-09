#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_decision_aux_w10_5epoch}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_state_policy_decision_aux_w10_5epoch}

export TIMESTAMP
export JOB_NAME
export OUTPUT_ROOT
export TRAIN_OUTPUT_DIR
export DECISION_AUX_WEIGHT=${DECISION_AUX_WEIGHT:-1.0}
export DECISION_AUX_LAMBDAS=${DECISION_AUX_LAMBDAS:-${UTILITY_LAMBDAS:-0,5,10,15,20,25,30,35,40,50,75,100,150,200}}
export DECISION_AUX_TEMPERATURE=${DECISION_AUX_TEMPERATURE:-0.1}
export DECISION_AUX_COST_MODE=${DECISION_AUX_COST_MODE:-fixed_train_mean}
export DECISION_AUX_STOP_TIE_BONUS=${DECISION_AUX_STOP_TIE_BONUS:-1.0e-4}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_5epoch.sh"
