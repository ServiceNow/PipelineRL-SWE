#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_train_all_16k_verified_eval_collect_5route_4b_scout_oss20_qwen30_oss120_gemini/collect}
JOB_NAME=${JOB_NAME:-offline_router_swe_bench_train_all_16k_verified_eval_5route_4b_scout_qwen3_embedding_8b_lora_frugalgpt_cascade_5epoch}
TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-0,1,2,3,4}
CASCADE_ORDER=${CASCADE_ORDER:-0,1,2,3,4}
CHECKPOINT_EVERY_EPOCH=${CHECKPOINT_EVERY_EPOCH:-true}
EPOCH_REPORT_EVERY=${EPOCH_REPORT_EVERY:-1}
COLLECT_OUTPUT_DIR="${COLLECT_OUTPUT_DIR}" JOB_NAME="${JOB_NAME}" TARGET_ROUTE_IDXS="${TARGET_ROUTE_IDXS}" CASCADE_ORDER="${CASCADE_ORDER}" CHECKPOINT_EVERY_EPOCH="${CHECKPOINT_EVERY_EPOCH}" EPOCH_REPORT_EVERY="${EPOCH_REPORT_EVERY}" \
  bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_3route_qwen3_embedding_8b_lora_frugalgpt_cascade_5epoch.sh"
