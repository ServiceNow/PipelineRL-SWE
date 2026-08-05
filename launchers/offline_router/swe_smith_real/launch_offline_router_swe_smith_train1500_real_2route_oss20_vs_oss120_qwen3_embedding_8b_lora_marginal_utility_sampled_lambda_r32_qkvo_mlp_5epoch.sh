#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_marginal_utility_sampled_lambda_r32_qkvo_mlp_5epoch}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_marginal_utility_sampled_lambda_r32_qkvo_mlp_5epoch}

export TIMESTAMP
export JOB_NAME
export OUTPUT_ROOT
export TRAIN_OUTPUT_DIR

# Two-route comparative: original route 1 = OSS-20B, original route 3 = OSS-120B.
# Local route 0 is the cheap/default action; local route 1 is the expensive action.
export TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-1,3}
export ROUTE_OUTPUT_COST_WEIGHTS=${ROUTE_OUTPUT_COST_WEIGHTS:-1.299e-6,1.113e-5}
export SCOUT_ROUTE_IDX=${SCOUT_ROUTE_IDX:-0}
export REGRET_DEFAULT_ROUTE_IDX=${REGRET_DEFAULT_ROUTE_IDX:-0}

# Direct/prompt-only decision setting. No state expansion and no chaining.
export ATTEMPTED_STATE_MODE=${ATTEMPTED_STATE_MODE:-none}
export INCLUDE_BARE_STATE=${INCLUDE_BARE_STATE:-true}
export MAX_POLICY_STEPS=${MAX_POLICY_STEPS:-1}

# Keep economics out of the input. The model remains a route-success predictor;
# sampled lambda/costs only shape the auxiliary decision loss.
export INCLUDE_COSTS_IN_PROMPT=${INCLUDE_COSTS_IN_PROMPT:-false}

# Broad report grid, sampled decision supervision over the budget-sensitive band.
export UTILITY_LAMBDAS=${UTILITY_LAMBDAS:-0,1,2,3,4,5,7.5,10,12.5,15,20,25,30,40,50,75,100,150,200}
export DECISION_AUX_WEIGHT=${DECISION_AUX_WEIGHT:-1.0}
export DECISION_AUX_LAMBDAS=${DECISION_AUX_LAMBDAS:-10}
export DECISION_AUX_LAMBDA_SAMPLING=${DECISION_AUX_LAMBDA_SAMPLING:-log_uniform}
export DECISION_AUX_LAMBDA_SAMPLE_COUNT=${DECISION_AUX_LAMBDA_SAMPLE_COUNT:-2}
export DECISION_AUX_LAMBDA_MIN=${DECISION_AUX_LAMBDA_MIN:-1}
export DECISION_AUX_LAMBDA_MAX=${DECISION_AUX_LAMBDA_MAX:-150}
export DECISION_AUX_TEMPERATURE=${DECISION_AUX_TEMPERATURE:-0.1}
export DECISION_AUX_COST_MODE=${DECISION_AUX_COST_MODE:-actual}
export DECISION_AUX_STOP_TIE_BONUS=${DECISION_AUX_STOP_TIE_BONUS:-1.0e-4}
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

# Keep this small: one direct-state row per instance, five epochs, checkpoint/report every epoch.
export NUM_EPOCHS=${NUM_EPOCHS:-5}
export CHECKPOINT_EVERY_EPOCH=${CHECKPOINT_EVERY_EPOCH:-true}
export EPOCH_REPORT_EVERY=${EPOCH_REPORT_EVERY:-1}

exec bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_5epoch.sh"
