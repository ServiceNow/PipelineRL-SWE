#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
MODEL=${MODEL:?MODEL must be set, e.g. Qwen/Qwen3-Coder-30B-A3B-Instruct}
MODEL_SLUG=${MODEL_SLUG:-$(printf "%s" "${MODEL}" | sed -E 's#[^A-Za-z0-9._-]+#_#g; s#^[._-]+|[._-]+$##g')}
JOB_NAME=${JOB_NAME:-offline_router_swebench_train_local_collect_${MODEL_SLUG}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${OUTPUT_ROOT}/collect}

DATASET_PATH=${DATASET_PATH:-/mnt/llmd/data/swebench/all_16k/ds_train}
DATASET_NAME=${DATASET_NAME:-swe_bench_train}
INSTANCE_IDS_PATH=${INSTANCE_IDS_PATH:-}
LIMIT=${LIMIT:-0}

PORT=${PORT:-8390}
GPU_COUNT=${GPU_COUNT:-4}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-${GPU_COUNT}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
DTYPE=${DTYPE:-bfloat16}
MAX_CONCURRENT_PROBLEMS=${MAX_CONCURRENT_PROBLEMS:-8}
CONNECTOR_LIMIT=${CONNECTOR_LIMIT:-32}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
HEALTHCHECK_TIMEOUT=${HEALTHCHECK_TIMEOUT:-5400}
HEALTHCHECK_POLL=${HEALTHCHECK_POLL:-10}
MAX_TOKENS=${MAX_TOKENS:-15000}
MAX_TOKEN_FALLBACKS=${MAX_TOKEN_FALLBACKS:-8192,4096,2048}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-1.0}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}
SLEEP_AFTER_MODEL=${SLEEP_AFTER_MODEL:-5}
SEED=${SEED:-42}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

VLLM_PYTHON=${VLLM_PYTHON:-/home/toolkit/.conda/envs/vllm-env/bin/python}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
COLLECTOR_ENV=${COLLECTOR_ENV:-pipeline-rl}
SNAPSHOT=${SNAPSHOT:-1}
LOCAL=${LOCAL:-0}
DRY_RUN=${DRY_RUN:-0}
CPU=${CPU:-48}
CPU_MEM=${CPU_MEM:-512}
GPU_MEM=${GPU_MEM:-80}

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Missing DATASET_PATH=${DATASET_PATH}" >&2
    exit 1
  fi
  if [[ -n "${INSTANCE_IDS_PATH}" && ! -s "${INSTANCE_IDS_PATH}" ]]; then
    echo "Missing or empty INSTANCE_IDS_PATH=${INSTANCE_IDS_PATH}" >&2
    exit 1
  fi
  if [[ ! -x "${VLLM_PYTHON}" ]]; then
    echo "Missing executable VLLM_PYTHON=${VLLM_PYTHON}" >&2
    exit 1
  fi
fi

limit_arg=""
if [[ "${LIMIT}" != "0" ]]; then
  limit_arg="--limit ${LIMIT}"
fi

instance_ids_arg=""
if [[ -n "${INSTANCE_IDS_PATH}" ]]; then
  instance_ids_arg="--instance-ids-path '${INSTANCE_IDS_PATH}'"
fi

vllm_extra_arg=""
if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  vllm_extra_arg="--vllm-extra-arg '${VLLM_EXTRA_ARGS}'"
fi

make job   ENV="${COLLECTOR_ENV}"   CONDA_EXE="${CONDA_EXE}"   CONDA=1   ACCELERATE=0   DEEPSPEED=0   NPROC=1   CPU="${CPU}"   CPU_MEM="${CPU_MEM}"   GPU="${GPU_COUNT}"   GPU_MEM="${GPU_MEM}"   SNAPSHOT="${SNAPSHOT}"   LOCAL="${LOCAL}"   DRY_RUN="${DRY_RUN}"   JOB_NAME="${JOB_NAME}_${TIMESTAMP}"   COMMAND="cd ${REPO_ROOT}; mkdir -p '${OUTPUT_ROOT}'; set -o pipefail; python -u -m pipelinerl.swe.scripts.offline_router.collect_model_discovery_candidates     --output-dir '${COLLECT_OUTPUT_DIR}'     --dataset-path '${DATASET_PATH}'     --dataset-name '${DATASET_NAME}'     ${instance_ids_arg}     ${limit_arg}     --models '${MODEL}'     --port ${PORT}     --vllm-python '${VLLM_PYTHON}'     --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}     --max-model-len ${MAX_MODEL_LEN}     --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}     --dtype '${DTYPE}'     --healthcheck-timeout ${HEALTHCHECK_TIMEOUT}     --healthcheck-poll ${HEALTHCHECK_POLL}     --max-concurrent-problems ${MAX_CONCURRENT_PROBLEMS}     --connector-limit ${CONNECTOR_LIMIT}     --request-timeout ${REQUEST_TIMEOUT}     --max-tokens ${MAX_TOKENS}     --max-token-fallbacks '${MAX_TOKEN_FALLBACKS}'     --temperature ${TEMPERATURE}     --top-p ${TOP_P}     --success-threshold ${SUCCESS_THRESHOLD}     --sleep-after-model ${SLEEP_AFTER_MODEL}     --seed ${SEED}     ${vllm_extra_arg}     ${EXTRA_ARGS}     2>&1 | tee -a '${OUTPUT_ROOT}/launch.out'"
