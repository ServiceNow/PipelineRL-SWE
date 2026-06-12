#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}
JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_eval286_cost_score_r64_qkvo_mlp_input_only}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
SCORE_OUTPUT_DIR=${SCORE_OUTPUT_DIR:-${OUTPUT_ROOT}/eval286_cost_score}
REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
COST_MODEL_TRAIN_DIR=${COST_MODEL_TRAIN_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_expert_cost_only_normalized_r64_qkvo_mlp_input_only_10epoch_1781247691/train_qwen3_embedding_8b_lora_expert_cost_only_normalized_10epoch}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-${COST_MODEL_TRAIN_DIR}/checkpoints/epoch_0002}
COST_NORMALIZATION_CONFIG=${COST_NORMALIZATION_CONFIG:-${COST_MODEL_TRAIN_DIR}/train_config.json}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
INPUT_MODE=${INPUT_MODE:-input_only}
EMBEDDING_INPUT_LAYOUT=${EMBEDDING_INPUT_LAYOUT:-single}
TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-0,1,2,3}
COST_ROUTE_IDXS=${COST_ROUTE_IDXS:-0,1,2,3}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LR=${LR:-2.0e-5}
COST_HEAD_LR=${COST_HEAD_LR:-5.0e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
SEED=${SEED:-17}
COST_MSE_WEIGHT=${COST_MSE_WEIGHT:-1.0}
COST_TARGET_NORMALIZATION=${COST_TARGET_NORMALIZATION:-per_route_standard}
MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE:-1024}
DROPOUT=${DROPOUT:-0.1}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

if [[ ! -f "${REAL_LABEL_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing real-label dataset: ${REAL_LABEL_DATASET_DIR}/metadata.json" >&2
  exit 1
fi
if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "Missing checkpoint dir: ${RESUME_FROM_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${COST_NORMALIZATION_CONFIG}" ]]; then
  echo "Missing cost normalization config: ${COST_NORMALIZATION_CONFIG}" >&2
  exit 1
fi

TRAIN_CMD="python pipelinerl/swe/scripts/offline_router/train_qwen_embedding_router_baseline.py"
if [[ "${TRAIN_NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${TRAIN_NPROC} \
    --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
    pipelinerl/swe/scripts/offline_router/train_qwen_embedding_router_baseline.py"
fi

ATTN_ARG=""
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ATTN_ARG="--attn-implementation ${ATTN_IMPLEMENTATION}"
fi

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT} ${SCORE_OUTPUT_DIR}; set -o pipefail; \
    ${TRAIN_CMD} \
      --dataset-dir ${REAL_LABEL_DATASET_DIR} \
      --output-dir ${SCORE_OUTPUT_DIR} \
      --model-name ${MODEL_NAME} \
      --objective cost_mse \
      --max-seq-length ${MAX_SEQ_LENGTH} \
      --input-mode ${INPUT_MODE} \
      --embedding-input-layout ${EMBEDDING_INPUT_LAYOUT} \
      --target-route-idxs ${TARGET_ROUTE_IDXS} \
      --num-epochs 1 \
      --batch-size ${BATCH_SIZE} \
      --eval-batch-size ${EVAL_BATCH_SIZE} \
      --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
      --lr ${LR} \
      --cost-head-lr ${COST_HEAD_LR} \
      --weight-decay ${WEIGHT_DECAY} \
      --warmup-ratio ${WARMUP_RATIO} \
      --max-train-rows 0 \
      --max-eval-rows 0 \
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
      --cost-normalization-config ${COST_NORMALIZATION_CONFIG} \
      --resume-from-checkpoint ${RESUME_FROM_CHECKPOINT} \
      --eval-only \
      2>&1 | tee -a ${SCORE_OUTPUT_DIR}/launch.out"
