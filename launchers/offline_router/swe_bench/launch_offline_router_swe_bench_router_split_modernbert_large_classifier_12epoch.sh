#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_router_split_collect_1776749732}
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${SOURCE_RUN_DIR}/collect}

MODEL_NAME=${MODEL_NAME:-answerdotai/ModernBERT-large}
JOB_NAME=${JOB_NAME:-offline_router_swe_bench_router_split_modernbert_large_classifier_12epoch}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_modernbert_large_classifier_12epoch}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
NUM_EPOCHS=${NUM_EPOCHS:-12}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LR=${LR:-2.0e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-4096}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-500}
SEED=${SEED:-17}
SAVE_MODEL=${SAVE_MODEL:-false}

if [[ ! -f "${COLLECT_OUTPUT_DIR}/metadata.json" ]]; then
  echo "Missing offline-router collection metadata: ${COLLECT_OUTPUT_DIR}/metadata.json" >&2
  echo "Set SOURCE_RUN_DIR or COLLECT_OUTPUT_DIR to an existing collection run." >&2
  exit 1
fi

TRAIN_CMD="python pipelinerl/swe/scripts/offline_router/train_modernbert_router_baseline.py"
if [[ "${TRAIN_NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${TRAIN_NPROC} \
    --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
    pipelinerl/swe/scripts/offline_router/train_modernbert_router_baseline.py"
fi

SAVE_MODEL_ARG=""
if [[ "${SAVE_MODEL}" == "true" ]]; then
  SAVE_MODEL_ARG="--save-model"
fi

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${TRAIN_OUTPUT_DIR}; set -o pipefail; ${TRAIN_CMD} \
    --dataset-dir ${COLLECT_OUTPUT_DIR} \
    --output-dir ${TRAIN_OUTPUT_DIR} \
    --model-name ${MODEL_NAME} \
    --objective route_classifier \
    --max-seq-length ${MAX_SEQ_LENGTH} \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --eval-batch-size ${EVAL_BATCH_SIZE} \
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --warmup-ratio ${WARMUP_RATIO} \
    --max-train-rows ${MAX_TRAIN_ROWS} \
    --max-eval-rows ${MAX_EVAL_ROWS} \
    --seed ${SEED} \
    --gradient-checkpointing \
    ${SAVE_MODEL_ARG} \
    2>&1 | tee -a ${TRAIN_OUTPUT_DIR}/launch.out"
