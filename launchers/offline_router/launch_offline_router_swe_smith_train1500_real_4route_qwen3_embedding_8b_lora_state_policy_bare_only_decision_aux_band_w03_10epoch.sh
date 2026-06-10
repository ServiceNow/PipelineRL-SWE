#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_bare_only_decision_aux_band_w03_10epoch}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_state_policy_bare_only_decision_aux_band_w03_10epoch}

export TIMESTAMP
export JOB_NAME
export OUTPUT_ROOT
export TRAIN_OUTPUT_DIR

# Direct-Q ablation: train only on the prompt-only state used by direct routing.
export ATTEMPTED_STATE_MODE=${ATTEMPTED_STATE_MODE:-none}
export INCLUDE_BARE_STATE=${INCLUDE_BARE_STATE:-true}
export MAX_POLICY_STEPS=${MAX_POLICY_STEPS:-1}

# Match the current all-subsets decision-aux band run except for state expansion.
export NUM_EPOCHS=${NUM_EPOCHS:-10}
export CHECKPOINT_EVERY_EPOCH=${CHECKPOINT_EVERY_EPOCH:-true}
export EPOCH_REPORT_EVERY=${EPOCH_REPORT_EVERY:-1}

export UTILITY_LAMBDAS=${UTILITY_LAMBDAS:-0,5,10,15,20,25,30,35,40,45,50,55,60,75,100,150,200}
export DECISION_AUX_WEIGHT=${DECISION_AUX_WEIGHT:-0.3}
export DECISION_AUX_LAMBDAS=${DECISION_AUX_LAMBDAS:-25,30,35,40,45,50,55,60}
export DECISION_AUX_TEMPERATURE=${DECISION_AUX_TEMPERATURE:-0.1}
export DECISION_AUX_COST_MODE=${DECISION_AUX_COST_MODE:-fixed_train_mean}
export DECISION_AUX_STOP_TIE_BONUS=${DECISION_AUX_STOP_TIE_BONUS:-1.0e-4}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_5epoch.sh"
