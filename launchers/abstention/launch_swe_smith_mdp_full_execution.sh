#!/usr/bin/env bash
# Prepared SWE-Smith adaptation of protocol v2. Safe by default: it prints the
# configuration and submits nothing unless SUBMIT=1 is set explicitly.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TRACE_ROOT=${TRACE_ROOT:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_multi_rollout_trace_collect_1781382734/eval150}
REPORT_ROOT=${REPORT_ROOT:-${REPO_ROOT}/router_analysis/uploaded_eval_full_20260617/swe_smith_multirollout_eval150_1781382734/logs/run_evaluation}
REPORT_SPLIT=${REPORT_SPLIT:-eval150}
DATASET_PATH=${DATASET_PATH:-/mnt/llmd/data/swe_smith_bugged_context/ds_test}
DATASET_NAME=${DATASET_NAME:-swe_smith_test_bugged_context}

# Once eval300 has real reports, set all four variables together. HELDOUT_IDS
# makes the no-overlap 300-instance collection the untouched test split.
EXTRA_TRACE_ROOT=${EXTRA_TRACE_ROOT:-}
EXTRA_REPORT_ROOT=${EXTRA_REPORT_ROOT:-}
EXTRA_REPORT_SPLIT=${EXTRA_REPORT_SPLIT:-eval300}
HELDOUT_IDS=${HELDOUT_IDS:-}

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-17}
NPROC=${NPROC:-4}
NUM_EPOCHS=${NUM_EPOCHS:-8}
NUM_ORDERINGS=${NUM_ORDERINGS:-10}
HISTORIES_PER_PROBLEM=${HISTORIES_PER_PROBLEM:-20}
EXECUTION_COST_USD=${EXECUTION_COST_USD:-0}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}
JOB_NAME=${JOB_NAME:-swe_smith_mdp_full_execution_seed${SEED}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

TENSORS_DIR=${OUTPUT_DIR}/tensors_v2
DATASET_DIR=${OUTPUT_DIR}/reachable_dataset
MODEL_DIR=${OUTPUT_DIR}/model
REPLAY_DIR=${OUTPUT_DIR}/replay

COLLECTION_ARGS="--trace-root ${TRACE_ROOT} --report-root ${REPORT_ROOT} --report-split ${REPORT_SPLIT}"
if [[ -n "${EXTRA_TRACE_ROOT}" || -n "${EXTRA_REPORT_ROOT}" || -n "${HELDOUT_IDS}" ]]; then
  : "${EXTRA_TRACE_ROOT:?Set EXTRA_TRACE_ROOT, EXTRA_REPORT_ROOT, and HELDOUT_IDS together}"
  : "${EXTRA_REPORT_ROOT:?Set EXTRA_TRACE_ROOT, EXTRA_REPORT_ROOT, and HELDOUT_IDS together}"
  : "${HELDOUT_IDS:?Set EXTRA_TRACE_ROOT, EXTRA_REPORT_ROOT, and HELDOUT_IDS together}"
  COLLECTION_ARGS+=" --trace-root ${EXTRA_TRACE_ROOT} --report-root ${EXTRA_REPORT_ROOT} --report-split ${EXTRA_REPORT_SPLIT} --heldout-ids ${HELDOUT_IDS}"
fi

if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_LAUNCH="python -m accelerate.commands.launch --multi_gpu --mixed_precision bf16 --num_processes ${NPROC} --config_file conf/accelerate/base_mp.yaml"
else
  TRAIN_LAUNCH="python"
fi

COMMAND="cd ${REPO_ROOT} && \
python pipelinerl/swe/scripts/offline_router/build_swe_smith_mdp_tensors_v2.py \
  ${COLLECTION_ARGS} --dataset-path ${DATASET_PATH} --dataset-name ${DATASET_NAME} \
  --output-dir ${TENSORS_DIR} --num-draws 3 --split-seed 0 && \
python pipelinerl/swe/scripts/livecodebench/build_mdp_reachable_dataset.py \
  --tensors-dir ${TENSORS_DIR} --output-dir ${DATASET_DIR} \
  --histories-per-problem ${HISTORIES_PER_PROBLEM} --seed 0 && \
${TRAIN_LAUNCH} pipelinerl/swe/scripts/livecodebench/train_mdp_reachable_policy.py \
  --dataset-dir ${DATASET_DIR} --output-dir ${MODEL_DIR} \
  --seed ${SEED} --num-epochs ${NUM_EPOCHS} --lr 1e-4 --max-seq-length 8192 && \
python pipelinerl/swe/scripts/livecodebench/replay_mdp_full_execution.py \
  --tensors-dir ${TENSORS_DIR} --output-dir ${REPLAY_DIR} \
  --sequential-model-dir ${MODEL_DIR} --num-orderings ${NUM_ORDERINGS} --cost-mode usd \
  --execution-cost-usd ${EXECUTION_COST_USD}"

echo "Prepared job: ${JOB_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Trace root: ${TRACE_ROOT}"
echo "Report root: ${REPORT_ROOT}"
if [[ "${SUBMIT}" != "1" ]]; then
  echo "Not submitted (SUBMIT=${SUBMIT}). Set SUBMIT=1 only after the LCB gate is reviewed."
  exit 0
fi

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC="${NPROC}" GPU=1 GPU_MEM=80 CPU=16 CPU_MEM=128 \
  COMMAND="${COMMAND}"
