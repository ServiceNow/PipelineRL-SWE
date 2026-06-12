#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SUBMIT_SLEEP_SECS=${SUBMIT_SLEEP_SECS:-90}

submit_job() {
  local script_path="$1"
  echo "Submitting ${script_path##*/}"
  bash "${script_path}"
}

sleep_between_submissions() {
  echo "Sleeping ${SUBMIT_SLEEP_SECS}s before next submission"
  sleep "${SUBMIT_SLEEP_SECS}"
}

submit_job "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_2route_capacity_reward_parallel.sh"
sleep_between_submissions
submit_job "${SCRIPT_DIR}/launch_offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_expert_cost_only_normalized_r128_qkvo_mlp_input_only_10epoch.sh"
