#!/usr/bin/env bash
set -euo pipefail
PREFILL_NAME=${PREFILL_NAME:-expansion_prefills_20261001}
for PREFILL_DATASET in ${PREFILL_DATASETS:-mmlupro omni500}; do
  PREFILL_COMMAND="bash analysis/cost_headroom/run_expansion_prefill.sh ${PREFILL_DATASET}"
  if [[ ${SUBMIT:-0} == 1 ]]; then
    make job JOB_NAME="${PREFILL_NAME}_${PREFILL_DATASET}" ENV=vllm-env CONDA_EXE=/opt/conda/bin/conda \
      GPU=1 GPU_MEM=32 CPU=8 CPU_MEM=48 SNAPSHOT=1 COMMAND="${PREFILL_COMMAND}"
  else printf '%s\n' "${PREFILL_NAME}_${PREFILL_DATASET}: ${PREFILL_COMMAND}"; fi
done
