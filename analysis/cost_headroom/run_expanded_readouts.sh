#!/usr/bin/env bash
set -euo pipefail
READOUT_DATASET=$1
ROOT=/mnt/llmd/results/exps/aristides/reason
LOG_ROOT="${ROOT}/expanded_eval_20261001/${READOUT_DATASET}"
mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/run.log") 2>&1
if [[ ! -f "${ROOT}/math_expand_20261001/${READOUT_DATASET}/COMPLETE.json" ]]; then
  echo "Expansion collection is not complete for ${READOUT_DATASET}" >&2
  exit 2
fi
if [[ ! -f "${ROOT}/expansion_prefills_20261001/${READOUT_DATASET}_prefill.npz" ]]; then
  echo "Prefill extraction is not complete for ${READOUT_DATASET}" >&2
  exit 2
fi
python -u analysis/cost_headroom/run_expanded_readouts.py --dataset "${READOUT_DATASET}"
