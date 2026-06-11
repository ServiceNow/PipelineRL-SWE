#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_3route_qwen3_embedding_8b_lora_marginal_utility_sampled_lambda_r32_qkvo_mlp_10epoch}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_marginal_utility_sampled_lambda_r32_qkvo_mlp_10epoch}

export TIMESTAMP
export JOB_NAME
export OUTPUT_ROOT
export TRAIN_OUTPUT_DIR

# Three-route low-budget setting: scout, OSS-20B, Qwen3-Coder-30B.
# Drop OSS-120B for this run because it loses the budget-random economics here.
export TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-0,1,2}
export ROUTE_OUTPUT_COST_WEIGHTS=${ROUTE_OUTPUT_COST_WEIGHTS:-2.78e-7,1.299e-6,4.64e-6}
export SCOUT_ROUTE_IDX=${SCOUT_ROUTE_IDX:-0}

# Direct decision setting: prompt-only state, no chained attempts.
export ATTEMPTED_STATE_MODE=${ATTEMPTED_STATE_MODE:-none}
export INCLUDE_BARE_STATE=${INCLUDE_BARE_STATE:-true}
export MAX_POLICY_STEPS=${MAX_POLICY_STEPS:-1}

# Keep cost out of the prompt so the head stays a reusable success predictor.
# Costs/lambdas are passed to the auxiliary loss and sampled at train time.
export INCLUDE_COSTS_IN_PROMPT=${INCLUDE_COSTS_IN_PROMPT:-false}

# Report broad lambdas, but train the decision auxiliary loss with sampled lambdas.
export UTILITY_LAMBDAS=${UTILITY_LAMBDAS:-0,5,8,10,12,15,18,20,25,30,35,40,45,50,60,75,100,150,200}
export DECISION_AUX_WEIGHT=${DECISION_AUX_WEIGHT:-1.0}
export DECISION_AUX_LAMBDAS=${DECISION_AUX_LAMBDAS:-25}
export DECISION_AUX_LAMBDA_SAMPLING=${DECISION_AUX_LAMBDA_SAMPLING:-log_uniform}
export DECISION_AUX_LAMBDA_SAMPLE_COUNT=${DECISION_AUX_LAMBDA_SAMPLE_COUNT:-2}
export DECISION_AUX_LAMBDA_MIN=${DECISION_AUX_LAMBDA_MIN:-15}
export DECISION_AUX_LAMBDA_MAX=${DECISION_AUX_LAMBDA_MAX:-120}
export DECISION_AUX_TEMPERATURE=${DECISION_AUX_TEMPERATURE:-0.1}
export DECISION_AUX_COST_MODE=${DECISION_AUX_COST_MODE:-actual}
export DECISION_AUX_STOP_TIE_BONUS=${DECISION_AUX_STOP_TIE_BONUS:-1.0e-4}

# Dynamic regret weighting inside the sampled-lambda CE: emphasize cases where
# the best route has real marginal utility over the scout/default action.
export DECISION_AUX_REGRET_WEIGHT_MODE=${DECISION_AUX_REGRET_WEIGHT_MODE:-default_action}
export DECISION_AUX_REGRET_WEIGHT_SCALE=${DECISION_AUX_REGRET_WEIGHT_SCALE:-6.0}
export DECISION_AUX_REGRET_WEIGHT_POWER=${DECISION_AUX_REGRET_WEIGHT_POWER:-1.0}
export DECISION_AUX_REGRET_WEIGHT_MIN=${DECISION_AUX_REGRET_WEIGHT_MIN:-1.0}
export DECISION_AUX_REGRET_WEIGHT_MAX=${DECISION_AUX_REGRET_WEIGHT_MAX:-8.0}
export SAMPLE_WEIGHTING=${SAMPLE_WEIGHTING:-uniform}

# Current best capacity setting from the LoRA ablations.
export LORA_R=${LORA_R:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

# Full run with checkpoints/reports so we can rescore intermediate epochs.
export NUM_EPOCHS=${NUM_EPOCHS:-10}
export CHECKPOINT_EVERY_EPOCH=${CHECKPOINT_EVERY_EPOCH:-true}
export EPOCH_REPORT_EVERY=${EPOCH_REPORT_EVERY:-1}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_5epoch.sh"
