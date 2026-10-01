#!/usr/bin/env bash
# Frozen evaluation expansion. Commit/push first: SNAPSHOT=1 uses committed code.
set -euo pipefail
EXPANSION_NAME=${EXPANSION_NAME:-math_expand_20261001}
EXPANSION_OUT=${EXPANSION_OUT:-/mnt/llmd/results/exps/aristides/reason/${EXPANSION_NAME}}
for EXPANSION_DATASET in mmlupro omni500; do
  if [[ ${EXPANSION_DATASET} == mmlupro ]]; then EXPANSION_BUDGET=35; else EXPANSION_BUDGET=25; fi
  EXPANSION_COMMAND="python -u pipelinerl/swe/scripts/math_pool/collect_expansion.py --plan analysis/cost_headroom/expansion_20261001/plan.json --tasks analysis/cost_headroom/expansion_20261001/${EXPANSION_DATASET}_tasks.jsonl --dataset ${EXPANSION_DATASET} --out ${EXPANSION_OUT} --budget-usd ${EXPANSION_BUDGET} --concurrency 48"
  if [[ ${SUBMIT:-0} == 1 ]]; then
    make job JOB_NAME="${EXPANSION_NAME}_${EXPANSION_DATASET}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=4 CPU_MEM=16 SNAPSHOT=1 COMMAND="${EXPANSION_COMMAND}"
  else
    printf '%s\n' "${EXPANSION_NAME}_${EXPANSION_DATASET}: ${EXPANSION_COMMAND}"
  fi
done
