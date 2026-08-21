#!/usr/bin/env bash
# Submit the corrected gpt-oss-20b tier collection for LCB full routing.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
: "${LCB_COLLECTION_DIR:?Set LCB_COLLECTION_DIR to the corrected 4B/120B collection}"

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
MID_MODEL=${MID_MODEL:-openai/gpt-oss-20b}
ROUTE_LABEL=${ROUTE_LABEL:-oss20}
CONCURRENCY=${CONCURRENCY:-4}
MAX_TOKENS=${MAX_TOKENS:-4096}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
GEN_TIMEOUT=${GEN_TIMEOUT:-300}
SNAPSHOT=${SNAPSHOT:-1}
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}
JOB_NAME=${JOB_NAME:-lcb_full_router_${ROUTE_LABEL}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

[[ -s "${OPENROUTER_API_KEY_FILE}" ]] || { echo "Missing OpenRouter API key" >&2; exit 1; }
COMMAND="cd ${REPO_ROOT} && source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh && python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py --source-collection-dir ${LCB_COLLECTION_DIR} --output-dir ${OUTPUT_DIR} --route-label ${ROUTE_LABEL} --model ${MID_MODEL} --api-key-file ${OPENROUTER_API_KEY_FILE} --concurrency ${CONCURRENCY} --max-tokens ${MAX_TOKENS} --temperature 0.0 --eval-timeout ${EVAL_TIMEOUT} --gen-timeout ${GEN_TIMEOUT}"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" \
  NPROC=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=64 COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
