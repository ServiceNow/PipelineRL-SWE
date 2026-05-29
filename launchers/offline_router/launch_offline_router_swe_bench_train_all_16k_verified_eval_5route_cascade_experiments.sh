#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
export TIMESTAMP

bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_7b_scout_qwen3_embedding_8b_lora_frugalgpt_cascade_5epoch.sh"
bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_4b_scout_qwen3_embedding_8b_lora_frugalgpt_cascade_5epoch.sh"
