#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
JOB_NAME=${JOB_NAME:-offline_router_swe_bench_router_split_collect_gemini3_flash_openrouter}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${OUTPUT_ROOT}/collect}

SOURCE_COLLECTION_DIR=${SOURCE_COLLECTION_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_router_split_collect_1776749732/collect}

OPENROUTER_MODEL=${OPENROUTER_MODEL:-google/gemini-3-flash-preview}
OPENROUTER_BASE_URL=${OPENROUTER_BASE_URL:-https://openrouter.ai/api}
OPENROUTER_API_KEY_ENV=${OPENROUTER_API_KEY_ENV:-OPENROUTER_API_KEY}
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}
OPENROUTER_TITLE=${OPENROUTER_TITLE:-PipelineRL-SWE-offline-router}
OPENROUTER_REFERER=${OPENROUTER_REFERER:-}

TRAIN_DATASET_NAMES=${TRAIN_DATASET_NAMES:-swe_bench_train}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-/mnt/llmd/data/swebench/ds_train}
EVAL_DATASET_NAMES=${EVAL_DATASET_NAMES:-swebench_lite}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/mnt/llmd/data/swebench_lite/ds}

COLLECT_TRAIN=${COLLECT_TRAIN:-true}
COLLECT_EVAL=${COLLECT_EVAL:-true}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-4096}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-500}
MAX_CONCURRENT_PROBLEMS=${MAX_CONCURRENT_PROBLEMS:-8}
CONNECTOR_LIMIT=${CONNECTOR_LIMIT:-64}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
SHARD_SIZE=${SHARD_SIZE:-64}
MAX_TOKENS=${MAX_TOKENS:-15000}
TEMPERATURE=${TEMPERATURE:-0.7}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}
COLLECT_EXTRA_ARGS=${COLLECT_EXTRA_ARGS:-}

CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
COLLECTOR_ENV=${COLLECTOR_ENV:-pipeline-rl}
SNAPSHOT=${SNAPSHOT:-0}
LOCAL=${LOCAL:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -d "${SOURCE_COLLECTION_DIR}" ]]; then
    echo "Missing SOURCE_COLLECTION_DIR=${SOURCE_COLLECTION_DIR}" >&2
    exit 1
  fi
  if [[ "${COLLECT_TRAIN}" == "true" && ! -d "${TRAIN_DATASET_PATH}" ]]; then
    echo "Missing TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH}" >&2
    exit 1
  fi
  if [[ "${COLLECT_EVAL}" == "true" && ! -d "${EVAL_DATASET_PATH}" ]]; then
    echo "Missing EVAL_DATASET_PATH=${EVAL_DATASET_PATH}" >&2
    exit 1
  fi
  if [[ -z "${!OPENROUTER_API_KEY_ENV:-}" && ! -f "${OPENROUTER_API_KEY_FILE}" ]]; then
    echo "Missing OpenRouter key. Set ${OPENROUTER_API_KEY_ENV} for LOCAL=1, or write it to OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE} for remote jobs." >&2
    exit 1
  fi
fi

collect_train_flag="--collect-train"
if [[ "${COLLECT_TRAIN}" != "true" ]]; then
  collect_train_flag="--no-collect-train"
fi
collect_eval_flag="--collect-eval"
if [[ "${COLLECT_EVAL}" != "true" ]]; then
  collect_eval_flag="--no-collect-eval"
fi

referer_arg=""
if [[ -n "${OPENROUTER_REFERER}" ]]; then
  referer_arg="--openrouter-referer '${OPENROUTER_REFERER}'"
fi

make job \
  ENV="${COLLECTOR_ENV}" \
  CONDA_EXE="${CONDA_EXE}" \
  CONDA=1 \
  ACCELERATE=0 \
  DEEPSPEED=0 \
  NPROC=1 \
  CPU=8 \
  CPU_MEM=32 \
  GPU=0 \
  GPU_MEM=0 \
  SNAPSHOT="${SNAPSHOT}" \
  LOCAL="${LOCAL}" \
  DRY_RUN="${DRY_RUN}" \
  JOB_NAME="${JOB_NAME}" \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT}; set -o pipefail; python -m pipelinerl.swe.scripts.offline_router.collect_openrouter_expert_from_existing \
    --source-collection-dir '${SOURCE_COLLECTION_DIR}' \
    --output-dir '${COLLECT_OUTPUT_DIR}' \
    --model '${OPENROUTER_MODEL}' \
    --expert-label 'expert_0:${OPENROUTER_MODEL}' \
    --base-url '${OPENROUTER_BASE_URL}' \
    --api-key-env '${OPENROUTER_API_KEY_ENV}' \
    --api-key-file '${OPENROUTER_API_KEY_FILE}' \
    --train-dataset-names '${TRAIN_DATASET_NAMES}' \
    --train-dataset-path '${TRAIN_DATASET_PATH}' \
    --train-max-samples ${TRAIN_MAX_SAMPLES} \
    --eval-dataset-names '${EVAL_DATASET_NAMES}' \
    --eval-dataset-path '${EVAL_DATASET_PATH}' \
    --eval-max-samples ${EVAL_MAX_SAMPLES} \
    ${collect_train_flag} \
    ${collect_eval_flag} \
    --max-concurrent-problems ${MAX_CONCURRENT_PROBLEMS} \
    --connector-limit ${CONNECTOR_LIMIT} \
    --request-timeout ${REQUEST_TIMEOUT} \
    --shard-size ${SHARD_SIZE} \
    --max-tokens ${MAX_TOKENS} \
    --temperature ${TEMPERATURE} \
    --success-threshold ${SUCCESS_THRESHOLD} \
    --openrouter-title '${OPENROUTER_TITLE}' \
    ${referer_arg} \
    ${COLLECT_EXTRA_ARGS} \
    2>&1 | tee -a ${OUTPUT_ROOT}/launch.out"
