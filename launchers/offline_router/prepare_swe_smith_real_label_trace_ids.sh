#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-42}
TRAIN_N=${TRAIN_N:-1500}
EVAL_N=${EVAL_N:-500}
ID_ROOT=${ID_ROOT:-/mnt/llmd/results/exps/aristides/reason/swe_smith_real_label_trace_ids_${TIMESTAMP}}
PYTHON_BIN=${PYTHON_BIN:-/home/toolkit/.conda/envs/pipeline-rl/bin/python}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=${PYTHON_BIN_FALLBACK:-python}
fi

"${PYTHON_BIN}" pipelinerl/swe/scripts/offline_router/sample_real_label_instance_ids.py \
  --output-dir "${ID_ROOT}" \
  --train-n "${TRAIN_N}" \
  --eval-n "${EVAL_N}" \
  --seed "${SEED}" \
  "$@"

echo "ID_ROOT=${ID_ROOT}"
echo "TRAIN_IDS=${ID_ROOT}/swe_smith_train_${TRAIN_N}_ids.txt"
echo "EVAL_IDS=${ID_ROOT}/swe_smith_eval_${EVAL_N}_ids.txt"
