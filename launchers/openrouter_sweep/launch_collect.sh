#!/usr/bin/env bash
# Launch the OpenRouter diversity sweep collection as an EAI job.
# Runs 15 models against the 286-instance eval set via the OpenRouter API.
# Results are JSONL files in OUTPUT_DIR, one per model, ready for Daytona eval.
#
# Requires OPENROUTER_API_KEY to be set in your environment before launching.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=$(date +%s)
JOB_NAME=openrouter_sweep_collect_${TIMESTAMP}

# Path to the 286-instance eval parquet shards
EVAL_PARQUET_DIR=${EVAL_PARQUET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/eval}
# Path to the on-disk HuggingFace dataset (provides file_contents)
DATASET_PATH=${DATASET_PATH:-/mnt/llmd/data/swe_smith_bugged_context/ds_train}
# Output directory for per-model prediction JSONL files
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

# Concurrent API calls per model (OpenRouter rate limit friendly)
CONCURRENCY=${CONCURRENCY:-20}

# Load OPENROUTER_API_KEY from .env if not already set
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  ENV_FILE="${REPO_ROOT}/.env"
  if [[ -f "${ENV_FILE}" ]]; then
    OPENROUTER_API_KEY=$(grep -E '^OPENROUTER_API_KEY=' "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"'"'" )
  fi
fi
: "${OPENROUTER_API_KEY:?Need OPENROUTER_API_KEY — set it in .env or the environment}"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=1 \
  GPU=0 \
  CPU=8 \
  CPU_MEM=32 \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_DIR}; \
    OPENROUTER_API_KEY=${OPENROUTER_API_KEY} \
    python pipelinerl/swe/scripts/openrouter_sweep/collect_openrouter_sweep.py \
      --eval-parquet-dir ${EVAL_PARQUET_DIR} \
      --dataset-path ${DATASET_PATH} \
      --output-dir ${OUTPUT_DIR} \
      --concurrency ${CONCURRENCY} \
    2>&1 | tee ${OUTPUT_DIR}/collect.log"

echo "Output will be in: ${OUTPUT_DIR}"
echo "Run launch_daytona.sh with OUTPUT_DIR=${OUTPUT_DIR} once collection is done."
