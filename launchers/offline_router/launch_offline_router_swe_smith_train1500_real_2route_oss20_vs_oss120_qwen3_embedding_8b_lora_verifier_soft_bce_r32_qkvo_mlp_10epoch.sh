#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}
JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_verifier_soft_bce_r32_qkvo_mlp_10epoch}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_verifier_soft_bce_oss20_vs_oss120_r32_qkvo_mlp_10epoch}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
NUM_EPOCHS=${NUM_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LR=${LR:-2.0e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-0}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-0}
SEED=${SEED:-17}
TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-1,3}
CASCADE_ORDER=${CASCADE_ORDER:-0,1}
MAX_THRESHOLD_CANDIDATES=${MAX_THRESHOLD_CANDIDATES:-21}
LOSS_TYPE=${LOSS_TYPE:-soft_bce}
UTILITY_LAMBDAS=${UTILITY_LAMBDAS:-0.0,1.0e-5,2.0e-5,5.0e-5,1.0e-4,2.0e-4}
MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE:-1024}
DROPOUT=${DROPOUT:-0.1}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
CHECKPOINT_EVERY_EPOCH=${CHECKPOINT_EVERY_EPOCH:-true}
EPOCH_REPORT_EVERY=${EPOCH_REPORT_EVERY:-1}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
SAVE_MODEL=${SAVE_MODEL:-true}

TRAIN_CMD="python pipelinerl/swe/scripts/offline_router/train_qwen_embedding_cascade_baseline.py"
if [[ "${TRAIN_NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${TRAIN_NPROC} \
    --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
    pipelinerl/swe/scripts/offline_router/train_qwen_embedding_cascade_baseline.py"
fi

SAVE_MODEL_ARG=""
if [[ "${SAVE_MODEL}" == "true" ]]; then
  SAVE_MODEL_ARG="--save-model"
fi

CHECKPOINT_ARG=""
if [[ "${CHECKPOINT_EVERY_EPOCH}" == "true" ]]; then
  CHECKPOINT_ARG="--checkpoint-every-epoch"
fi

RESUME_ARG=""
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARG="--resume-from-checkpoint ${RESUME_FROM_CHECKPOINT}"
fi

ATTN_ARG=""
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ATTN_ARG="--attn-implementation ${ATTN_IMPLEMENTATION}"
fi

TARGET_ROUTE_ARG=""
if [[ -n "${TARGET_ROUTE_IDXS}" ]]; then
  TARGET_ROUTE_ARG="--target-route-idxs ${TARGET_ROUTE_IDXS}"
fi

if [[ ! -f "${REAL_LABEL_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing real-label router dataset: ${REAL_LABEL_DATASET_DIR}/metadata.json" >&2
  exit 1
fi

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT} ${TRAIN_OUTPUT_DIR}; set -o pipefail; \
    ${TRAIN_CMD} \
      --dataset-dir ${REAL_LABEL_DATASET_DIR} \
      --output-dir ${TRAIN_OUTPUT_DIR} \
      --model-name ${MODEL_NAME} \
      --cascade-order ${CASCADE_ORDER} \
      ${TARGET_ROUTE_ARG} \
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
      --dropout ${DROPOUT} \
      --mlp-hidden-size ${MLP_HIDDEN_SIZE} \
      --torch-dtype ${TORCH_DTYPE} \
      ${ATTN_ARG} \
      --no-encoder-frozen \
      --use-lora \
      --lora-r ${LORA_R} \
      --lora-alpha ${LORA_ALPHA} \
      --lora-dropout ${LORA_DROPOUT} \
      --lora-target-modules ${LORA_TARGET_MODULES} \
      --gradient-checkpointing \
      --max-threshold-candidates ${MAX_THRESHOLD_CANDIDATES} \
      --loss-type ${LOSS_TYPE} \
      --utility-lambdas ${UTILITY_LAMBDAS} \
      --epoch-report-every ${EPOCH_REPORT_EVERY} \
      ${CHECKPOINT_ARG} \
      ${RESUME_ARG} \
      ${SAVE_MODEL_ARG} \
      2>&1 | tee -a ${TRAIN_OUTPUT_DIR}/launch.out"
