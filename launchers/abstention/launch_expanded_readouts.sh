#!/usr/bin/env bash
set -euo pipefail
DATASETS=${DATASETS:-"omni500 mmlupro"}
for DATASET in ${DATASETS}; do
  COMMAND="bash analysis/cost_headroom/run_expanded_readouts.sh ${DATASET}"
  if [[ ${SUBMIT:-0} == 1 ]]; then
    make job JOB_NAME="expanded_eval_20261001_${DATASET}" ENV=pipeline-rl \
      CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 NPROC=1 GPU=0 GPU_MEM=0 \
      CPU=8 CPU_MEM=64 COMMAND="${COMMAND}"
  else
    printf '%s\n' "expanded_eval_20261001_${DATASET}: ${COMMAND}"
  fi
done
