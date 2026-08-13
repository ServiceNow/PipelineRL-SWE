#!/usr/bin/env bash
# Collect Claude Opus 5 outputs on SWE-bench Verified eval via OpenRouter.
#
# This produces the Opus 5 patches needed to run Daytona eval on Verified
# (cross-domain panel of the routing cost-savings chart).
#
# Runs collect_openrouter_expert_from_existing.py against the existing
# 5-route Verified collection, adding Opus 5 as the expert route.
# Expert output is stored at route_outputs[1] in the output parquets.
#
# After this completes, run launch_opus_verified_daytona_eval.sh to
# extract patches and run Daytona scoring.
#
# Optional env vars:
#   OPENROUTER_MODEL        -- OpenRouter model ID (default: anthropic/claude-opus-5)
#   SOURCE_COLLECTION_DIR   -- 5-route Verified collection (default below)
#   VERIFIED_DATASET_PATH   -- local SWE-bench Verified HF dataset (default below)
#   MAX_CONCURRENT_PROBLEMS -- concurrency (default: 4, Opus is expensive/rate-limited)
#   MAX_TOKENS              -- max output tokens (default: 16000)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
OPENROUTER_MODEL=${OPENROUTER_MODEL:-anthropic/claude-opus-5}
MODEL_SLUG=$(echo "${OPENROUTER_MODEL}" | tr '/.-' '___')

JOB_NAME=${JOB_NAME:-verified_collect_${MODEL_SLUG}_openrouter}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${OUTPUT_ROOT}/collect}

SOURCE_COLLECTION_DIR=${SOURCE_COLLECTION_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_train_all_16k_verified_eval_collect_5route_4b_scout_oss20_qwen30_oss120_gemini/collect}
VERIFIED_DATASET_PATH=${VERIFIED_DATASET_PATH:-/mnt/llmd/data/swebench_verified/all_16k/ds}

OPENROUTER_BASE_URL=${OPENROUTER_BASE_URL:-https://openrouter.ai/api}
OPENROUTER_API_KEY_ENV=${OPENROUTER_API_KEY_ENV:-OPENROUTER_API_KEY}
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}
OPENROUTER_TITLE=${OPENROUTER_TITLE:-PipelineRL-SWE-offline-router}

MAX_CONCURRENT_PROBLEMS=${MAX_CONCURRENT_PROBLEMS:-4}
CONNECTOR_LIMIT=${CONNECTOR_LIMIT:-32}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
SHARD_SIZE=${SHARD_SIZE:-64}
MAX_TOKENS=${MAX_TOKENS:-16000}
TEMPERATURE=${TEMPERATURE:-0.7}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}

CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
SNAPSHOT=${SNAPSHOT:-1}
LOCAL=${LOCAL:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -d "${SOURCE_COLLECTION_DIR}" ]]; then
    echo "Missing source collection: ${SOURCE_COLLECTION_DIR}" >&2
    exit 1
  fi
  if [[ ! -d "${VERIFIED_DATASET_PATH}" ]]; then
    echo "Missing VERIFIED_DATASET_PATH=${VERIFIED_DATASET_PATH}" >&2
    exit 1
  fi
  if [[ "${LOCAL}" != "1" && ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
    echo "Missing or empty OpenRouter key file: OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE}" >&2
    echo "Remote EAI jobs do not inherit your local shell's ${OPENROUTER_API_KEY_ENV}; write the key to the file." >&2
    exit 1
  fi
fi

echo "=== Submitting Verified Opus collection: ${JOB_NAME}_${TIMESTAMP} ==="
echo "  Model:           ${OPENROUTER_MODEL}"
echo "  Source:          ${SOURCE_COLLECTION_DIR}"
echo "  Dataset:         ${VERIFIED_DATASET_PATH}"
echo "  Output:          ${OUTPUT_ROOT}"
echo "  Concurrency:     ${MAX_CONCURRENT_PROBLEMS}"
echo ""

make -C "${REPO_ROOT}" job \
  ENV=pipeline-rl \
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
  JOB_NAME="${JOB_NAME}_${TIMESTAMP}" \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT}; set -o pipefail; { python pipelinerl/swe/scripts/offline_router/collect_openrouter_expert_from_existing.py \
    --source-collection-dir '${SOURCE_COLLECTION_DIR}' \
    --train-source-collection-dir '${SOURCE_COLLECTION_DIR}' \
    --eval-source-collection-dir  '${SOURCE_COLLECTION_DIR}' \
    --output-dir '${COLLECT_OUTPUT_DIR}' \
    --model '${OPENROUTER_MODEL}' \
    --expert-label 'expert_0:${OPENROUTER_MODEL}' \
    --base-url '${OPENROUTER_BASE_URL}' \
    --api-key-env '${OPENROUTER_API_KEY_ENV}' \
    --api-key-file '${OPENROUTER_API_KEY_FILE}' \
    --train-dataset-names 'swe_bench_verified' \
    --train-dataset-path '${VERIFIED_DATASET_PATH}' \
    --train-max-samples 0 \
    --eval-dataset-names 'swe_bench_verified' \
    --eval-dataset-path '${VERIFIED_DATASET_PATH}' \
    --eval-max-samples 0 \
    --no-collect-train \
    --collect-eval \
    --max-concurrent-problems ${MAX_CONCURRENT_PROBLEMS} \
    --connector-limit ${CONNECTOR_LIMIT} \
    --request-timeout ${REQUEST_TIMEOUT} \
    --shard-size ${SHARD_SIZE} \
    --max-tokens ${MAX_TOKENS} \
    --temperature ${TEMPERATURE} \
    --success-threshold ${SUCCESS_THRESHOLD} \
    --openrouter-title '${OPENROUTER_TITLE}'; \
    python pipelinerl/swe/scripts/offline_router/summarize_collected_dataset.py --dataset-dir '${COLLECT_OUTPUT_DIR}'; \
  } 2>&1 | tee -a ${OUTPUT_ROOT}/launch.out"

echo ""
echo "Collection output: ${OUTPUT_ROOT}"
echo "Next step (after job completes):"
echo "  SOURCE_COLLECTION_DIR=${COLLECT_OUTPUT_DIR}/eval \\"
echo "  bash launchers/abstention/launch_opus_verified_daytona_eval.sh"
