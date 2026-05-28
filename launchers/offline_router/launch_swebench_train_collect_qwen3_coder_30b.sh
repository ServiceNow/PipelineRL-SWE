#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
MODEL_SLUG=${MODEL_SLUG:-Qwen_Qwen3-Coder-30B-A3B-Instruct}
JOB_NAME=${JOB_NAME:-offline_router_swebench_train_collect_qwen3_coder_30b}
GPU_COUNT=${GPU_COUNT:-4}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MODEL="${MODEL}" MODEL_SLUG="${MODEL_SLUG}" JOB_NAME="${JOB_NAME}" GPU_COUNT="${GPU_COUNT}" TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" MAX_MODEL_LEN="${MAX_MODEL_LEN}"   bash "${SCRIPT_DIR}/launch_swebench_train_local_model_collect.sh"
