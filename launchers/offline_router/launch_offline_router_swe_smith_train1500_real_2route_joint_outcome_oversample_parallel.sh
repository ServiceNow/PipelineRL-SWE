#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TIMESTAMP=${TIMESTAMP:-$(date +%s)}

bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_joint_outcome_oversample_r32_qkvo_mlp_input_only_10epoch.sh" &
bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_joint_outcome_oversample_r32_qkvo_mlp_semantic_late_fusion_input_only_10epoch.sh" &
wait
