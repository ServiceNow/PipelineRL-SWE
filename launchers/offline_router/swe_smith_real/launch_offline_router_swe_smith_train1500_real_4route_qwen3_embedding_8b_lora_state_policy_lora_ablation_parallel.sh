#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_TIMESTAMP=${TIMESTAMP:-$(date +%s)}

# Baseline already exists for r=16 q/k/v/o:
# offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_allsubsets_decision_aux_band_w03_10epoch
#
# These three jobs change only LoRA capacity/target modules relative to that setup.

TIMESTAMP=$((BASE_TIMESTAMP + 1)) bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_lora_r32_qkvo_decision_aux_band_10epoch.sh"
TIMESTAMP=$((BASE_TIMESTAMP + 2)) bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_lora_r16_qkvo_mlp_decision_aux_band_10epoch.sh"
TIMESTAMP=$((BASE_TIMESTAMP + 3)) bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_lora_r32_qkvo_mlp_decision_aux_band_10epoch.sh"
