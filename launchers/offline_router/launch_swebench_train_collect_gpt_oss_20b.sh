#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:-openai/gpt-oss-20b}
MODEL_SLUG=${MODEL_SLUG:-openai_gpt-oss-20b}
JOB_NAME=${JOB_NAME:-offline_router_swebench_train_collect_gpt_oss_20b}
GPU_COUNT=${GPU_COUNT:-4}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MODEL="${MODEL}" MODEL_SLUG="${MODEL_SLUG}" JOB_NAME="${JOB_NAME}" GPU_COUNT="${GPU_COUNT}" TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" MAX_MODEL_LEN="${MAX_MODEL_LEN}"   bash "${SCRIPT_DIR}/launch_swebench_train_local_model_collect.sh"
