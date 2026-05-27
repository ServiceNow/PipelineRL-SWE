#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
JOB_NAME=${JOB_NAME:-prepare_swe_bench_verified_full}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}

HF_DATASET=${HF_DATASET:-princeton-nlp/SWE-bench_Verified}
HF_SPLIT=${HF_SPLIT:-test}
DATASET_ROOT=${DATASET_ROOT:-/mnt/llmd/data/swebench_verified/full}
SINGLE_OUTPUT_PATH=${SINGLE_OUTPUT_PATH:-${DATASET_ROOT}/ds}
SUMMARY_JSON_PATH=${SUMMARY_JSON_PATH:-$(dirname "${SINGLE_OUTPUT_PATH}")/prepare_swe_bench_single_summary.json}
REPOS_BASE_DIR=${REPOS_BASE_DIR:-/mnt/llmd/data/swebench_verified/repos}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-0}
MAX_NORMALIZED_ROWS=${MAX_NORMALIZED_ROWS:-0}
ROW_SAMPLE_SEED=${ROW_SAMPLE_SEED:-42}
GIT_TIMEOUT_SECONDS=${GIT_TIMEOUT_SECONDS:-900}
GOLD_FILE_SOURCE=${GOLD_FILE_SOURCE:-raw-url}

CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
COLLECTOR_ENV=${COLLECTOR_ENV:-pipeline-rl}
SNAPSHOT=${SNAPSHOT:-1}
LOCAL=${LOCAL:-0}
DRY_RUN=${DRY_RUN:-0}

JOB_CPU=${JOB_CPU:-4}
JOB_CPU_MEM=${JOB_CPU_MEM:-32}

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ -e "${SINGLE_OUTPUT_PATH}" ]]; then
    echo "Refusing to overwrite existing SINGLE_OUTPUT_PATH=${SINGLE_OUTPUT_PATH}" >&2
    echo "Set SINGLE_OUTPUT_PATH/DATASET_ROOT to a new path, or remove the old dataset intentionally." >&2
    exit 1
  fi
fi

make job \
  ENV="${COLLECTOR_ENV}" \
  CONDA_EXE="${CONDA_EXE}" \
  CONDA=1 \
  ACCELERATE=0 \
  DEEPSPEED=0 \
  NPROC=1 \
  CPU="${JOB_CPU}" \
  CPU_MEM="${JOB_CPU_MEM}" \
  GPU=0 \
  GPU_MEM=0 \
  SNAPSHOT="${SNAPSHOT}" \
  LOCAL="${LOCAL}" \
  DRY_RUN="${DRY_RUN}" \
  JOB_NAME="${JOB_NAME}_${TIMESTAMP}" \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT} ${DATASET_ROOT}; set -euo pipefail; python pipelinerl/swe/scripts/new/prepare_swe_smith_dataset.py \
    --hf-dataset '${HF_DATASET}' \
    --hf-split '${HF_SPLIT}' \
    --single-output-path '${SINGLE_OUTPUT_PATH}' \
    --reconstruct-missing-gold-files \
    --gold-file-source '${GOLD_FILE_SOURCE}' \
    --repos-base-dir '${REPOS_BASE_DIR}' \
    --max-normalized-rows ${MAX_NORMALIZED_ROWS} \
    --row-sample-seed ${ROW_SAMPLE_SEED} \
    --max-total-tokens ${MAX_TOTAL_TOKENS} \
    --tokenizer-model '${TOKENIZER_MODEL}' \
    --git-timeout-seconds ${GIT_TIMEOUT_SECONDS} \
    2>&1 | tee -a ${OUTPUT_ROOT}/launch.out; \
    if [[ -f '${SUMMARY_JSON_PATH}' ]]; then cp '${SUMMARY_JSON_PATH}' '${OUTPUT_ROOT}/prepare_swe_bench_single_summary.json'; fi"
