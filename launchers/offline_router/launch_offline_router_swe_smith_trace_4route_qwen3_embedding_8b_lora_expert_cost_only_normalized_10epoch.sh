#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}
JOB_NAME=${JOB_NAME:-offline_router_swe_smith_trace_4route_qwen3_embedding_8b_lora_expert_cost_only_normalized_10epoch}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
TRACE_COST_DATASET_DIR=${TRACE_COST_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_cost_4route_real_eval_split_no_leak_1780461385/collect}
REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_expert_cost_only_normalized_10epoch}

# The materializer is cheap relative to training. Keep it in the job so relaunches
# are self-contained after pushing a new snapshot.
PREPARE_TRACE_DATASET=${PREPARE_TRACE_DATASET:-true}
PREPARE_OVERWRITE=${PREPARE_OVERWRITE:-false}
# all_non_eval uses every aligned train1500 trace except the held-out real-label eval ids.
# Set to reference_train for the stricter 1146-row train split used by reward training.
REFERENCE_TRAIN_MODE=${REFERENCE_TRAIN_MODE:-all_non_eval}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
INPUT_MODE=${INPUT_MODE:-post_primary}
INCLUDE_PRIMARY_OUTPUT_TOKEN_COUNT=${INCLUDE_PRIMARY_OUTPUT_TOKEN_COUNT:-true}
TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-0,1,2,3}
NUM_EPOCHS=${NUM_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LR=${LR:-2.0e-5}
COST_HEAD_LR=${COST_HEAD_LR:-5.0e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-0}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-0}
SEED=${SEED:-17}
# Post-scout routing already knows the scout output cost. Predict only the
# additional candidate-solver output costs.
COST_ROUTE_IDXS=${COST_ROUTE_IDXS:-1,2,3}
COST_MSE_WEIGHT=${COST_MSE_WEIGHT:-1.0}
COST_TARGET_NORMALIZATION=${COST_TARGET_NORMALIZATION:-per_route_standard}
MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE:-1024}
DROPOUT=${DROPOUT:-0.1}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}
CHECKPOINT_EVERY_EPOCH=${CHECKPOINT_EVERY_EPOCH:-true}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
SAVE_MODEL=${SAVE_MODEL:-false}

TRAIN_CMD="python pipelinerl/swe/scripts/offline_router/train_qwen_embedding_router_baseline.py"
if [[ "${TRAIN_NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${TRAIN_NPROC} \
    --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
    pipelinerl/swe/scripts/offline_router/train_qwen_embedding_router_baseline.py"
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

PRIMARY_OUTPUT_TOKEN_COUNT_ARG=""
if [[ "${INCLUDE_PRIMARY_OUTPUT_TOKEN_COUNT}" == "true" ]]; then
  PRIMARY_OUTPUT_TOKEN_COUNT_ARG="--include-primary-output-token-count"
fi

TARGET_ROUTE_IDXS_ARG=""
if [[ -n "${TARGET_ROUTE_IDXS}" ]]; then
  TARGET_ROUTE_IDXS_ARG="--target-route-idxs ${TARGET_ROUTE_IDXS}"
fi

PREPARE_OVERWRITE_ARG=""
if [[ "${PREPARE_OVERWRITE}" == "true" ]]; then
  PREPARE_OVERWRITE_ARG="--overwrite"
fi

PREPARE_CMD=""
if [[ "${PREPARE_TRACE_DATASET}" == "true" ]]; then
  PREPARE_CMD="if [[ ! -f ${TRACE_COST_DATASET_DIR}/metadata.json || ${PREPARE_OVERWRITE} == true ]]; then \
    python pipelinerl/swe/scripts/offline_router/materialize_trace_cost_router_dataset.py \
      --output-dir ${TRACE_COST_DATASET_DIR} \
      --split-reference-dataset-dir ${REAL_LABEL_DATASET_DIR} \
      --reference-train-mode ${REFERENCE_TRAIN_MODE} \
      --seed ${SEED} \
      ${PREPARE_OVERWRITE_ARG}; \
    fi;"
fi

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT} ${TRAIN_OUTPUT_DIR}; set -o pipefail; \
    ${PREPARE_CMD} \
    ${TRAIN_CMD} \
      --dataset-dir ${TRACE_COST_DATASET_DIR} \
      --output-dir ${TRAIN_OUTPUT_DIR} \
      --model-name ${MODEL_NAME} \
      --objective cost_mse \
      --max-seq-length ${MAX_SEQ_LENGTH} \
      --input-mode ${INPUT_MODE} \
      ${PRIMARY_OUTPUT_TOKEN_COUNT_ARG} \
      ${TARGET_ROUTE_IDXS_ARG} \
      --num-epochs ${NUM_EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --eval-batch-size ${EVAL_BATCH_SIZE} \
      --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
      --lr ${LR} \
      --cost-head-lr ${COST_HEAD_LR} \
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
      --predict-costs \
      --cost-route-idxs ${COST_ROUTE_IDXS} \
      --cost-gradient-mode joint \
      --cost-mse-weight ${COST_MSE_WEIGHT} \
      --cost-target-normalization ${COST_TARGET_NORMALIZATION} \
      ${CHECKPOINT_ARG} \
      ${RESUME_ARG} \
      ${SAVE_MODEL_ARG} \
      2>&1 | tee -a ${TRAIN_OUTPUT_DIR}/launch.out"
