#!/usr/bin/env bash
# CARROT-style trained 137M encoder; no generation or embedding API calls.
set -euo pipefail
CARROT_JINA_NAME=${CARROT_JINA_NAME:-carrot_jina_20261001}
CARROT_JINA_OUT=${CARROT_JINA_OUT:-/mnt/llmd/results/exps/aristides/reason/${CARROT_JINA_NAME}}
for CARROT_JINA_POOL in ${CARROT_JINA_POOLS:-LCB Omni MMLU-Pro}; do
  CARROT_JINA_LABEL=${CARROT_JINA_POOL,,}
  CARROT_JINA_LABEL=${CARROT_JINA_LABEL//-/_}
  CARROT_JINA_COMMAND="bash analysis/cost_headroom/run_carrot_jina.sh ${CARROT_JINA_POOL} ${CARROT_JINA_OUT}"
  if [[ ${SUBMIT:-0} == 1 ]]; then
    make job JOB_NAME="${CARROT_JINA_NAME}_${CARROT_JINA_LABEL}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=1 GPU_MEM=32 CPU=8 CPU_MEM=32 SNAPSHOT=1 COMMAND="${CARROT_JINA_COMMAND}"
  else
    printf '%s\n' "${CARROT_JINA_NAME}_${CARROT_JINA_LABEL}: ${CARROT_JINA_COMMAND}"
  fi
done
