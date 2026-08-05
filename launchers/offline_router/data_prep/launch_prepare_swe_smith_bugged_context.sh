#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
JOB_NAME=${JOB_NAME:-prepare_swe_smith_bugged_context}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
DATASET_ROOT=${DATASET_ROOT:-/mnt/llmd/data/swe_smith_bugged_context}
TRAIN_OUTPUT_PATH=${TRAIN_OUTPUT_PATH:-${DATASET_ROOT}/ds_train}
TEST_OUTPUT_PATH=${TEST_OUTPUT_PATH:-${DATASET_ROOT}/ds_test}
SUMMARY_JSON_PATH=${SUMMARY_JSON_PATH:-${DATASET_ROOT}/prepare_swe_smith_summary.json}
REPOS_BASE_DIR=${REPOS_BASE_DIR:-/mnt/llmd/data/swe_smith/repos}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-16000}
MAX_NORMALIZED_ROWS=${MAX_NORMALIZED_ROWS:-0}
ROW_SAMPLE_SEED=${ROW_SAMPLE_SEED:-42}
SPLIT_STRATEGY=${SPLIT_STRATEGY:-disjoint-repo}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.8}
SPLIT_SEED=${SPLIT_SEED:-42}
GIT_TIMEOUT_SECONDS=${GIT_TIMEOUT_SECONDS:-900}
GOLD_FILE_SOURCE=${GOLD_FILE_SOURCE:-git}
HF_DATASETS=${HF_DATASETS:-SWE-bench/SWE-smith-py SWE-bench/SWE-smith-go SWE-bench/SWE-smith-rs SWE-bench/SWE-smith-java}

CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
COLLECTOR_ENV=${COLLECTOR_ENV:-pipeline-rl}
SNAPSHOT=${SNAPSHOT:-1}
LOCAL=${LOCAL:-0}
DRY_RUN=${DRY_RUN:-0}
JOB_CPU=${JOB_CPU:-8}
JOB_CPU_MEM=${JOB_CPU_MEM:-64}

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ -e "${TRAIN_OUTPUT_PATH}" || -e "${TEST_OUTPUT_PATH}" ]]; then
    echo "Refusing to overwrite existing TRAIN_OUTPUT_PATH=${TRAIN_OUTPUT_PATH} or TEST_OUTPUT_PATH=${TEST_OUTPUT_PATH}" >&2
    echo "Set DATASET_ROOT/TRAIN_OUTPUT_PATH/TEST_OUTPUT_PATH to a new path, or remove the old dataset intentionally." >&2
    exit 1
  fi
fi

hf_args=""
for dataset in ${HF_DATASETS}; do
  hf_args+=" --hf-dataset '${dataset}'"
done

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
    ${hf_args} \
    --hf-split train \
    --train-output-path '${TRAIN_OUTPUT_PATH}' \
    --test-output-path '${TEST_OUTPUT_PATH}' \
    --split-strategy '${SPLIT_STRATEGY}' \
    --train-fraction ${TRAIN_FRACTION} \
    --split-seed ${SPLIT_SEED} \
    --reconstruct-missing-gold-files \
    --gold-file-source '${GOLD_FILE_SOURCE}' \
    --repos-base-dir '${REPOS_BASE_DIR}' \
    --max-normalized-rows ${MAX_NORMALIZED_ROWS} \
    --row-sample-seed ${ROW_SAMPLE_SEED} \
    --max-total-tokens ${MAX_TOTAL_TOKENS} \
    --tokenizer-model '${TOKENIZER_MODEL}' \
    --git-timeout-seconds ${GIT_TIMEOUT_SECONDS} \
    --swesmith-bugged-context \
    2>&1 | tee -a ${OUTPUT_ROOT}/launch.out; \
    if [[ -f '${SUMMARY_JSON_PATH}' ]]; then cp '${SUMMARY_JSON_PATH}' '${OUTPUT_ROOT}/prepare_swe_smith_summary.json'; fi"
