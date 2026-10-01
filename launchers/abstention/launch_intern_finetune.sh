#!/usr/bin/env bash
set -euo pipefail
INTERN_FT_NAME=${INTERN_FT_NAME:-intern_finetune_20261001}
INTERN_FT_OUT=${INTERN_FT_OUT:-/mnt/llmd/results/exps/aristides/reason/${INTERN_FT_NAME}}
for INTERN_FT_POOL in ${INTERN_FT_POOLS:-LCB Omni MMLU-Pro}; do
  INTERN_FT_LABEL=${INTERN_FT_POOL,,}
  INTERN_FT_LABEL=${INTERN_FT_LABEL//-/_}
  INTERN_FT_COMMAND="bash analysis/cost_headroom/run_intern_finetune.sh ${INTERN_FT_POOL} ${INTERN_FT_OUT}"
  if [[ ${SUBMIT:-0} == 1 ]]; then
    make job JOB_NAME="${INTERN_FT_NAME}_${INTERN_FT_LABEL}" CONDA=0 GPU=1 GPU_MEM=48 CPU=8 CPU_MEM=64 SNAPSHOT=1 COMMAND="${INTERN_FT_COMMAND}"
  else printf '%s\n' "${INTERN_FT_NAME}_${INTERN_FT_LABEL}: ${INTERN_FT_COMMAND}"; fi
done
