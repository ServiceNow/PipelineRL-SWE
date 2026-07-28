#!/usr/bin/env bash
# Train the autoregressive Yes/No verifier on CoT trajectories + Daytona labels.
#
# Prerequisites:
#   - collect_cot_trajectories.py has run (produces trajectories_train.jsonl)
#   - Daytona eval has run on predictions_train.jsonl (produces report.json)
#
# Multi-GPU SFT with accelerate. Uses NPROC=4 GPUs by default.
#
# Usage:
#   TRAJECTORIES_DIR=/mnt/.../cot_trajectories_XYZ \
#   TRAIN_DAYTONA_REPORT=logs/run_evaluation/cot_train_XYZ/report.json \
#   bash launch_autoregressive_verifier_train.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=$(date +%s)
JOB_NAME=autoreg_verifier_train_${TIMESTAMP}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

# Trajectories JSONL from collect_cot_trajectories.py
TRAJECTORIES_DIR=${TRAJECTORIES_DIR:?Need TRAJECTORIES_DIR set}
TRAIN_TRAJECTORIES_PATH=${TRAIN_TRAJECTORIES_PATH:-${TRAJECTORIES_DIR}/trajectories_train.jsonl}
EVAL_TRAJECTORIES_PATH=${EVAL_TRAJECTORIES_PATH:-${TRAJECTORIES_DIR}/trajectories_eval.jsonl}

# Daytona summary reports (report.json from run_swesmith_eval_daytona.py)
TRAIN_DAYTONA_REPORT=${TRAIN_DAYTONA_REPORT:?Need TRAIN_DAYTONA_REPORT set to logs/run_evaluation/<run_id>/report.json}
EVAL_DAYTONA_REPORT=${EVAL_DAYTONA_REPORT:-}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}

NPROC=${NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}

TRAIN_CMD="python pipelinerl/swe/scripts/offline_router/train_autoregressive_verifier.py"
if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${NPROC} \
    --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
    pipelinerl/swe/scripts/offline_router/train_autoregressive_verifier.py"
fi

EVAL_ARGS=""
if [[ -f "${EVAL_TRAJECTORIES_PATH}" && -n "${EVAL_DAYTONA_REPORT}" ]]; then
  EVAL_ARGS="--eval-trajectories-path ${EVAL_TRAJECTORIES_PATH} --eval-daytona-report-path ${EVAL_DAYTONA_REPORT}"
fi

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${NPROC} \
  GPU=1 \
  GPU_MEM=80 \
  CPU=16 \
  CPU_MEM=128 \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_DIR}; set -o pipefail; { \
    ${TRAIN_CMD} \
      --trajectories-path ${TRAIN_TRAJECTORIES_PATH} \
      --daytona-report-path ${TRAIN_DAYTONA_REPORT} \
      ${EVAL_ARGS} \
      --output-dir ${OUTPUT_DIR} \
      --model-name ${MODEL_NAME} \
      --num-epochs ${NUM_EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --learning-rate ${LEARNING_RATE} \
      --max-seq-length ${MAX_SEQ_LENGTH} \
      --include-thinking; \
  } 2>&1 | tee ${OUTPUT_DIR}/train.log"

echo "Model will be saved to: ${OUTPUT_DIR}"
