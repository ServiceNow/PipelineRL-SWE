#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=$(date +%s)
JOB_NAME=${JOB_NAME:-offline_router_train}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}

# Reserve a full node by default; the trainer itself is single-process.
NPROC=${NPROC:-8}
DATASET_DIR=${DATASET_DIR:-}
MODEL_PATH=${MODEL_PATH:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ -z "${DATASET_DIR}" ]]; then
  echo "DATASET_DIR must be set to the collected offline router dataset directory." >&2
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must be set to the base or RL checkpoint to train from." >&2
  exit 1
fi

MODE=${MODE:-frozen_backbone}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-32000}

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${NPROC} \
  COMMAND="cd /home/toolkit/PipelineRL-SWE; python -m pipelinerl.swe.scripts.offline_router.train_router_offline \
    output_dir=${OUTPUT_DIR} \
    offline_router.train.dataset_dir=${DATASET_DIR} \
    offline_router.train.model_path=${MODEL_PATH} \
    offline_router.train.mode=${MODE} \
    offline_router.train.num_epochs=${NUM_EPOCHS} \
    offline_router.train.batch_size=${BATCH_SIZE} \
    offline_router.train.eval_batch_size=${EVAL_BATCH_SIZE} \
    offline_router.train.gradient_accumulation_steps=${GRAD_ACCUM} \
    offline_router.train.lr=${LR} \
    offline_router.train.weight_decay=${WEIGHT_DECAY} \
    offline_router.train.max_seq_length=${MAX_SEQ_LENGTH} \
    ${EXTRA_ARGS}"
