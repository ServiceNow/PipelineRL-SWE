#!/usr/bin/env bash
# Multi-draw stochastic collection for the thread-(a) MDP: correctness tensors
# (problem x model x draw) on the LCB temporal eval split.
#
# ONE job total: the official LCB evaluator forks a process per call and
# concurrent forks corrupt labels, so ALL grading must happen sequentially in a
# single process. This launcher loops models x draws inside one EAI job.
#
# Protocol (RoR-parity): temperature 0.2, k=10 draws, {scout, oss20, oss120},
# eval split. Plus scout-only T=0.6 sensitivity arm.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${LCB_COLLECTION_DIR:?Set LCB_COLLECTION_DIR to the corrected 4B/120B collection}"
: "${OPENROUTER_API_KEY_FILE:=/home/toolkit/.secrets/openrouter_api_key}"

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
NUM_DRAWS=${NUM_DRAWS:-10}
TEMP_PRIMARY=${TEMP_PRIMARY:-0.2}
CONCURRENCY=${CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-4096}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
GEN_TIMEOUT=${GEN_TIMEOUT:-600}
SNAPSHOT=${SNAPSHOT:-1}

JOB_NAME=${JOB_NAME:-lcb_multidraw_mdp_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

SCOUT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
OSS20_MODEL="openai/gpt-oss-20b"
OSS120_MODEL="openai/gpt-oss-120b"

COLLECT="python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py \
  --source-collection-dir ${LCB_COLLECTION_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --splits eval --concurrency ${CONCURRENCY} \
  --max-tokens ${MAX_TOKENS} --eval-timeout ${EVAL_TIMEOUT} --gen-timeout ${GEN_TIMEOUT} \
  --api-key-file ${OPENROUTER_API_KEY_FILE}"

# Inner loop runs inside one job process; every draw/model gets its own output
# file via --output-suffix so the resume logic works independently per file.
COMMAND="cd ${REPO_ROOT} && source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh && set -e"
for DRAW in $(seq 0 $((NUM_DRAWS-1))); do
  for PAIR in "scout:${SCOUT_MODEL}:${TEMP_PRIMARY}" "oss20:${OSS20_MODEL}:${TEMP_PRIMARY}" "oss120:${OSS120_MODEL}:${TEMP_PRIMARY}"; do
    ROUTE="${PAIR%%:*}"; REST="${PAIR#*:}"; MODEL="${REST%%:*}"; TEMP="${REST##*:}"
    COMMAND="${COMMAND} && echo '=== draw ${DRAW} route ${ROUTE} temp ${TEMP} ===' && ${COLLECT} --route-label ${ROUTE} --model '${MODEL}' --temperature ${TEMP} --output-suffix _d${DRAW}"
  done
done
# Sensitivity arm: scout at T=0.6
for DRAW in $(seq 0 $((NUM_DRAWS-1))); do
  COMMAND="${COMMAND} && echo '=== sensitivity: scout temp 0.6 draw ${DRAW} ===' && ${COLLECT} --route-label scout06 --model '${SCOUT_MODEL}' --temperature 0.6 --output-suffix _d${DRAW}"
done

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" \
  NPROC=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=32 COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
