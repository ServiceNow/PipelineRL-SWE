#!/usr/bin/env bash
set -euo pipefail

# Experiment B: kNN retrieval abstention.
# Encodes all train+eval tasks with the 8B encoder (base AND LoRA-tuned),
# then evaluates k-nearest-neighbour abstention (no learned head) vs the
# trained-head baseline.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
JOB_NAME=${JOB_NAME:-knn_retrieval_abstention}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
JOB_NPROC=${JOB_NPROC:-1}   # single GPU sufficient for inference
SKIP_BASE=${SKIP_BASE:-false}

SKIP_BASE_ARG=""
if [[ "${SKIP_BASE}" == "true" ]]; then
  SKIP_BASE_ARG="--skip-base"
fi

mkdir -p "${OUTPUT_ROOT}"

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT}; set -o pipefail; \
    python analysis/knn_retrieval_abstention.py ${SKIP_BASE_ARG} \
    2>&1 | tee -a ${OUTPUT_ROOT}/launch.out"
