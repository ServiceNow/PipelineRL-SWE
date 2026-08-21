#!/usr/bin/env bash
# Materialize corrected LCB tier outcomes, train a direct router, and score policies.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
: "${LCB_COLLECTION_DIR:?Set LCB_COLLECTION_DIR to the corrected 4B/120B collection}"
: "${LCB_OSS20_DIR:?Set LCB_OSS20_DIR to launch_lcb_full_router_collect_oss20.sh output}"

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
INPUT_ONLY=${INPUT_ONLY:-false}
INCLUDE_TEST_FEEDBACK=${INCLUDE_TEST_FEEDBACK:-true}
SNAPSHOT=${SNAPSHOT:-1}
JOB_NAME=${JOB_NAME:-lcb_full_router_train_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}
ROUTER_DATA_DIR=${ROUTER_DATA_DIR:-${OUTPUT_DIR}/router_data}

if [[ "${INPUT_ONLY}" == "true" ]]; then INPUT_ARG="--input-only"; else INPUT_ARG=""; fi
if [[ "${INCLUDE_TEST_FEEDBACK}" == "true" ]]; then FEEDBACK_ARG="--include-test-feedback"; else FEEDBACK_ARG=""; fi

COMMAND="cd ${REPO_ROOT} && python pipelinerl/swe/scripts/livecodebench/materialize_lcb_full_router.py --source-collection-dir ${LCB_COLLECTION_DIR} --expert-collection-dir ${LCB_OSS20_DIR} --output-dir ${ROUTER_DATA_DIR} && python pipelinerl/swe/scripts/livecodebench/train_lcb_full_router.py --router-data-dir ${ROUTER_DATA_DIR} --output-dir ${OUTPUT_DIR}/model ${INPUT_ARG} ${FEEDBACK_ARG} && python pipelinerl/swe/scripts/livecodebench/evaluate_lcb_full_router.py --predictions ${OUTPUT_DIR}/model/eval_predictions.jsonl --output-path ${OUTPUT_DIR}/policy_report.json"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" \
  NPROC=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=64 COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
