#!/usr/bin/env bash
# Collect Claude Opus outputs on SWE-bench train (all_16k) + Verified eval via OpenRouter.
#
# This adds Opus as the "call much stronger model instead of abstaining" route,
# to be used alongside the existing 5-route cascade
# (4B scout / oss-20b / qwen30b / oss-120b / gemini-flash).
#
# Optional env vars:
#   OPENROUTER_MODEL        -- OpenRouter model ID (default: anthropic/claude-opus-4-7)
#   TRAIN_SOURCE_COLLECTION_DIR / EVAL_SOURCE_COLLECTION_DIR -- existing collection for problem context
#   COLLECT_TRAIN / COLLECT_EVAL -- "true"/"false" to collect each split (default: both true)
#   TRAIN_MAX_SAMPLES / EVAL_MAX_SAMPLES -- 0 = all (default)
#   MAX_CONCURRENT_PROBLEMS -- concurrency (default: 4, Opus is expensive/rate-limited)
#   MAX_TOKENS              -- max output tokens (default: 16000)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
OPENROUTER_MODEL=${OPENROUTER_MODEL:-anthropic/claude-opus-5}
# Sanitize model name for use in job/dir names
MODEL_SLUG=$(echo "${OPENROUTER_MODEL}" | tr '/' '_' | tr '.' '_')

JOB_NAME=${JOB_NAME:-swebench_train_all_16k_verified_eval_collect_${MODEL_SLUG}_openrouter}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${OUTPUT_ROOT}/collect}

# Source collections (problem statements + existing route outputs for context)
TRAIN_SOURCE_COLLECTION_DIR=${TRAIN_SOURCE_COLLECTION_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_router_split_collect_all_16k_1777651552/collect}
EVAL_SOURCE_COLLECTION_DIR=${EVAL_SOURCE_COLLECTION_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_router_split_collect_verified_eval_only_1777683967/collect}

OPENROUTER_BASE_URL=${OPENROUTER_BASE_URL:-https://openrouter.ai/api}
OPENROUTER_API_KEY_ENV=${OPENROUTER_API_KEY_ENV:-OPENROUTER_API_KEY}
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}
OPENROUTER_TITLE=${OPENROUTER_TITLE:-PipelineRL-SWE-offline-router}
OPENROUTER_REFERER=${OPENROUTER_REFERER:-}

TRAIN_DATASET_NAMES=${TRAIN_DATASET_NAMES:-swe_bench_train}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-/mnt/llmd/data/swebench/all_16k/ds_train}
EVAL_DATASET_NAMES=${EVAL_DATASET_NAMES:-swe_bench_verified}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/mnt/llmd/data/swebench_verified/all_16k/ds}

COLLECT_TRAIN=${COLLECT_TRAIN:-true}
COLLECT_EVAL=${COLLECT_EVAL:-true}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-0}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-0}
# Opus is expensive — keep concurrency low to avoid rate limits and runaway costs
MAX_CONCURRENT_PROBLEMS=${MAX_CONCURRENT_PROBLEMS:-4}
CONNECTOR_LIMIT=${CONNECTOR_LIMIT:-32}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
SHARD_SIZE=${SHARD_SIZE:-64}
MAX_TOKENS=${MAX_TOKENS:-16000}
TEMPERATURE=${TEMPERATURE:-0.7}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}
COLLECT_EXTRA_ARGS=${COLLECT_EXTRA_ARGS:-}

CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
COLLECTOR_ENV=${COLLECTOR_ENV:-pipeline-rl}
JOB_NPROC=${JOB_NPROC:-1}
SNAPSHOT=${SNAPSHOT:-1}
LOCAL=${LOCAL:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ "${DRY_RUN}" != "1" ]]; then
  for required_dir in "${TRAIN_SOURCE_COLLECTION_DIR}" "${EVAL_SOURCE_COLLECTION_DIR}"; do
    if [[ ! -d "${required_dir}" ]]; then
      echo "Missing required directory: ${required_dir}" >&2
      exit 1
    fi
  done
  if [[ "${COLLECT_TRAIN}" == "true" && ! -d "${TRAIN_DATASET_PATH}" ]]; then
    echo "Missing TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH}" >&2
    exit 1
  fi
  if [[ "${COLLECT_EVAL}" == "true" && ! -d "${EVAL_DATASET_PATH}" ]]; then
    echo "Missing EVAL_DATASET_PATH=${EVAL_DATASET_PATH}" >&2
    exit 1
  fi
  if [[ "${LOCAL}" != "1" && ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
    echo "Missing or empty OpenRouter key file for remote job: OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE}" >&2
    echo "Remote EAI jobs do not inherit your local shell's ${OPENROUTER_API_KEY_ENV}; write the key to the file or override OPENROUTER_API_KEY_FILE." >&2
    exit 1
  fi
  if [[ "${LOCAL}" == "1" && -z "${!OPENROUTER_API_KEY_ENV:-}" && ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
    echo "Missing OpenRouter key. Set ${OPENROUTER_API_KEY_ENV}, or write it to OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE}." >&2
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

echo "=== Submitting Opus collection job: ${JOB_NAME}_${TIMESTAMP} ==="
echo "  Model:       ${OPENROUTER_MODEL}"
echo "  Train split: ${COLLECT_TRAIN} (source: ${TRAIN_SOURCE_COLLECTION_DIR})"
echo "  Eval split:  ${COLLECT_EVAL}  (source: ${EVAL_SOURCE_COLLECTION_DIR})"
echo "  Output:      ${OUTPUT_ROOT}"
echo ""

make -C "${REPO_ROOT}" job \
  ENV="${COLLECTOR_ENV}" \
  CONDA_EXE="${CONDA_EXE}" \
  CONDA=1 \
  ACCELERATE=0 \
  DEEPSPEED=0 \
  NPROC="${JOB_NPROC}" \
  CPU=8 \
  CPU_MEM=32 \
  GPU=0 \
  GPU_MEM=0 \
  SNAPSHOT="${SNAPSHOT}" \
  LOCAL="${LOCAL}" \
  DRY_RUN="${DRY_RUN}" \
  JOB_NAME="${JOB_NAME}_${TIMESTAMP}" \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT}; set -o pipefail; { python pipelinerl/swe/scripts/offline_router/collect_openrouter_expert_from_existing.py \
    --train-source-collection-dir '${TRAIN_SOURCE_COLLECTION_DIR}' \
    --eval-source-collection-dir '${EVAL_SOURCE_COLLECTION_DIR}' \
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
    ${COLLECT_EXTRA_ARGS}; \
    python pipelinerl/swe/scripts/offline_router/summarize_collected_dataset.py --dataset-dir '${COLLECT_OUTPUT_DIR}'; \
  } 2>&1 | tee -a ${OUTPUT_ROOT}/launch.out"
