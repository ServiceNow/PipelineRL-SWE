#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_proxy_real_verifier_full}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}
PROXY_DATASET_DIR=${PROXY_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_cost_4route_expanded_1781073985/collect}
REAL_SOURCE_DATASET_DIR=${REAL_SOURCE_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
REAL_BUDGET_INSTANCES=${REAL_BUDGET_INSTANCES:-500}
REAL_SELECTION_STRATEGY=${REAL_SELECTION_STRATEGY:-random}
REAL_DATASET_DIR=${REAL_DATASET_DIR:-${OUTPUT_ROOT}/datasets/real_${REAL_SELECTION_STRATEGY}_${REAL_BUDGET_INSTANCES}_seed${SEED:-17}}

MULTIROLLOUT_SCORE_DATASET_DIR=${MULTIROLLOUT_SCORE_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_multirollout_eval150_proxy_verifier_rescore_dataset_1781735193/collect}
SCORE_OUTPUT_DIR=${SCORE_OUTPUT_DIR:-${OUTPUT_ROOT}/scores/multirollout_eval150}

PROXY_TRAIN_OUTPUT_DIR=${PROXY_TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_proxy_verifier_12k_attempts}
REAL_TRAIN_OUTPUT_DIR=${REAL_TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_proxy_init_real_verifier_${REAL_BUDGET_INSTANCES}instances}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
PROXY_NUM_EPOCHS=${PROXY_NUM_EPOCHS:-5}
REAL_NUM_EPOCHS=${REAL_NUM_EPOCHS:-5}
PROXY_LR=${PROXY_LR:-2.0e-5}
REAL_LR=${REAL_LR:-1.0e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-0}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-0}
SEED=${SEED:-17}
TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-}
CASCADE_ORDER=${CASCADE_ORDER:-0,1,2,3}
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
SAVE_MODEL=${SAVE_MODEL:-true}

if [[ "${PROXY_NUM_EPOCHS}" -lt 1 ]]; then
  echo "PROXY_NUM_EPOCHS must be >= 1" >&2
  exit 1
fi
PROXY_LAST_EPOCH=$(printf "%04d" "$((PROXY_NUM_EPOCHS - 1))")
INIT_FROM_PROXY_CHECKPOINT=${INIT_FROM_PROXY_CHECKPOINT:-${PROXY_TRAIN_OUTPUT_DIR}/checkpoints/epoch_${PROXY_LAST_EPOCH}}

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

ATTN_ARG=""
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ATTN_ARG="--attn-implementation ${ATTN_IMPLEMENTATION}"
fi

TARGET_ROUTE_ARG=""
if [[ -n "${TARGET_ROUTE_IDXS}" ]]; then
  TARGET_ROUTE_ARG="--target-route-idxs ${TARGET_ROUTE_IDXS}"
fi

if [[ ! -f "${PROXY_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing proxy verifier dataset: ${PROXY_DATASET_DIR}/metadata.json" >&2
  exit 1
fi
if [[ ! -f "${REAL_SOURCE_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing source real-label dataset: ${REAL_SOURCE_DATASET_DIR}/metadata.json" >&2
  exit 1
fi
if [[ ! -f "${MULTIROLLOUT_SCORE_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing multirollout score dataset: ${MULTIROLLOUT_SCORE_DATASET_DIR}/metadata.json" >&2
  exit 1
fi

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT} ${PROXY_TRAIN_OUTPUT_DIR} ${REAL_TRAIN_OUTPUT_DIR} ${SCORE_OUTPUT_DIR}; set -o pipefail; \
    python pipelinerl/swe/scripts/offline_router/materialize_active_real_label_verifier_subset.py \
      --source-dataset-dir ${REAL_SOURCE_DATASET_DIR} \
      --output-dir ${REAL_DATASET_DIR} \
      --strategy ${REAL_SELECTION_STRATEGY} \
      --budget-instances ${REAL_BUDGET_INSTANCES} \
      --seed ${SEED} \
      2>&1 | tee -a ${OUTPUT_ROOT}/materialize_real_subset.out; \
    ${TRAIN_CMD} \
      --dataset-dir ${PROXY_DATASET_DIR} \
      --output-dir ${PROXY_TRAIN_OUTPUT_DIR} \
      --model-name ${MODEL_NAME} \
      --cascade-order ${CASCADE_ORDER} \
      ${TARGET_ROUTE_ARG} \
      --max-seq-length ${MAX_SEQ_LENGTH} \
      --num-epochs ${PROXY_NUM_EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --eval-batch-size ${EVAL_BATCH_SIZE} \
      --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
      --lr ${PROXY_LR} \
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
      ${SAVE_MODEL_ARG} \
      2>&1 | tee -a ${PROXY_TRAIN_OUTPUT_DIR}/launch.out; \
    test -f ${INIT_FROM_PROXY_CHECKPOINT}/model.safetensors; \
    ${TRAIN_CMD} \
      --dataset-dir ${REAL_DATASET_DIR} \
      --output-dir ${REAL_TRAIN_OUTPUT_DIR} \
      --model-name ${MODEL_NAME} \
      --cascade-order ${CASCADE_ORDER} \
      ${TARGET_ROUTE_ARG} \
      --max-seq-length ${MAX_SEQ_LENGTH} \
      --num-epochs ${REAL_NUM_EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --eval-batch-size ${EVAL_BATCH_SIZE} \
      --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
      --lr ${REAL_LR} \
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
      --init-from-model-checkpoint ${INIT_FROM_PROXY_CHECKPOINT} \
      ${CHECKPOINT_ARG} \
      ${SAVE_MODEL_ARG} \
      2>&1 | tee -a ${REAL_TRAIN_OUTPUT_DIR}/launch.out; \
    python pipelinerl/swe/scripts/offline_router/score_qwen_embedding_cascade_verifier.py \
      --dataset-dir ${MULTIROLLOUT_SCORE_DATASET_DIR} \
      --checkpoint-dir ${REAL_TRAIN_OUTPUT_DIR} \
      --output-dir ${SCORE_OUTPUT_DIR} \
      --split eval \
      ${TARGET_ROUTE_ARG} \
      --route-order ${CASCADE_ORDER} \
      --max-seq-length ${MAX_SEQ_LENGTH} \
      --batch-size ${EVAL_BATCH_SIZE} \
      --loss-type ${LOSS_TYPE} \
      --device cuda \
      2>&1 | tee -a ${SCORE_OUTPUT_DIR}/launch.out"
