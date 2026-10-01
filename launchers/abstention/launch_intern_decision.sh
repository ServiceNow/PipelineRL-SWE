#!/usr/bin/env bash
set -euo pipefail
INTERN_NAME=${INTERN_NAME:-intern_decision_20261001}
INTERN_OUT=${INTERN_OUT:-/mnt/llmd/results/exps/aristides/reason/${INTERN_NAME}}
INTERN_COMMAND="bash analysis/cost_headroom/run_intern_decision.sh ${INTERN_OUT}"
if [[ ${SUBMIT:-0} == 1 ]]; then
  make job JOB_NAME="${INTERN_NAME}" CONDA=0 GPU=1 GPU_MEM=32 CPU=8 CPU_MEM=48 SNAPSHOT=1 COMMAND="${INTERN_COMMAND}"
else
  printf '%s\n' "${INTERN_NAME}: ${INTERN_COMMAND}"
fi
