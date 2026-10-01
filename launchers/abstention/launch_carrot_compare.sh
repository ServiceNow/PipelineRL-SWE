#!/usr/bin/env bash
# CARROT's local SBERT/kNN variant: no API calls. Commit/push before snapshot launch.
set -euo pipefail
CARROT_NAME=${CARROT_NAME:-carrot_compare_20261001}
CARROT_OUT=${CARROT_OUT:-/mnt/llmd/results/exps/aristides/reason/${CARROT_NAME}}
for CARROT_POOL in ${CARROT_POOLS:-LCB Omni MMLU-Pro}; do
  CARROT_LABEL=${CARROT_POOL,,}
  CARROT_LABEL=${CARROT_LABEL//-/_}
  CARROT_COMMAND="python -m pip install --no-deps --target /tmp/carrot_python_deps sentence-transformers==3.4.1 && PYTHONPATH=/tmp/carrot_python_deps python -u analysis/cost_headroom/carrot_compare.py --pool ${CARROT_POOL} --out ${CARROT_OUT} --bootstrap 1000"
  if [[ ${SUBMIT:-0} == 1 ]]; then
    make job JOB_NAME="${CARROT_NAME}_${CARROT_LABEL}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=16 SNAPSHOT=1 COMMAND="${CARROT_COMMAND}"
  else
    printf '%s\n' "${CARROT_NAME}_${CARROT_LABEL}: ${CARROT_COMMAND}"
  fi
done
