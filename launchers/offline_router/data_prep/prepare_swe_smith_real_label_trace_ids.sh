#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-42}
TRAIN_N=${TRAIN_N:-1500}
EVAL_N=${EVAL_N:-500}
SWE_SMITH_DATA_ROOT=${SWE_SMITH_DATA_ROOT:-/mnt/llmd/data/swe_smith_bugged_context}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-${SWE_SMITH_DATA_ROOT}/ds_train}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-${SWE_SMITH_DATA_ROOT}/ds_test}
TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-swe_smith_train_bugged_context}
EVAL_DATASET_NAME=${EVAL_DATASET_NAME:-swe_smith_test_bugged_context}
ID_ROOT=${ID_ROOT:-/mnt/llmd/results/exps/aristides/reason/swe_smith_real_label_trace_ids_${TIMESTAMP}}
PYTHON_BIN=${PYTHON_BIN:-/home/toolkit/.conda/envs/pipeline-rl/bin/python}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=${PYTHON_BIN_FALLBACK:-python}
fi

EXCLUDE_ARGS=()
for path in ${EXCLUDE_IDS_PATHS:-}; do
  EXCLUDE_ARGS+=(--exclude-ids-path "${path}")
done
for path in ${EXCLUDE_TRAIN_IDS_PATHS:-}; do
  EXCLUDE_ARGS+=(--exclude-train-ids-path "${path}")
done
for path in ${EXCLUDE_EVAL_IDS_PATHS:-}; do
  EXCLUDE_ARGS+=(--exclude-eval-ids-path "${path}")
done

"${PYTHON_BIN}" pipelinerl/swe/scripts/offline_router/sample_real_label_instance_ids.py \
  --output-dir "${ID_ROOT}" \
  --train-dataset-path "${TRAIN_DATASET_PATH}" \
  --train-dataset-name "${TRAIN_DATASET_NAME}" \
  --eval-dataset-path "${EVAL_DATASET_PATH}" \
  --eval-dataset-name "${EVAL_DATASET_NAME}" \
  --train-n "${TRAIN_N}" \
  --eval-n "${EVAL_N}" \
  --seed "${SEED}" \
  "${EXCLUDE_ARGS[@]}" \
  "$@"

echo "ID_ROOT=${ID_ROOT}"
echo "TRAIN_IDS=${ID_ROOT}/swe_smith_train_${TRAIN_N}_ids.txt"
echo "EVAL_IDS=${ID_ROOT}/swe_smith_eval_${EVAL_N}_ids.txt"
