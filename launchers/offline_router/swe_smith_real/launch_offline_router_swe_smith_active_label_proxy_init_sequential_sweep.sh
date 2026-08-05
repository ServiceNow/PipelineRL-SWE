#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_real4route_proxyinit_active_label_seq_sweep}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
RUN_ROOT=${RUN_ROOT:-${OUTPUT_ROOT}}
DATASET_ROOT=${DATASET_ROOT:-${RUN_ROOT}/datasets}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}
SOURCE_DATASET_DIR=${SOURCE_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
SWEEP_SPECS=${SWEEP_SPECS:-random:128,top2_margin:128,mean_uncertainty:128,high_variance:128,high_score:128}
PROXY_PREDICTIONS=${PROXY_PREDICTIONS:-}
INIT_FROM_MODEL_CHECKPOINT=${INIT_FROM_MODEL_CHECKPOINT:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch_1781684950/train_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch/checkpoints/epoch_0004}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
NUM_EPOCHS=${NUM_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LR=${LR:-1.0e-5}
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
SUBRUN_SLEEP_SECS=${SUBRUN_SLEEP_SECS:-5}

if [[ ! -f "${SOURCE_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing source real-label dataset: ${SOURCE_DATASET_DIR}/metadata.json" >&2
  exit 1
fi
if [[ ! -f "${INIT_FROM_MODEL_CHECKPOINT}/model.safetensors" && ! -f "${INIT_FROM_MODEL_CHECKPOINT}" ]]; then
  echo "Missing init checkpoint model.safetensors: ${INIT_FROM_MODEL_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT}; set -o pipefail; \
    MODEL_NAME='${MODEL_NAME}' \
    RUN_ROOT='${RUN_ROOT}' \
    DATASET_ROOT='${DATASET_ROOT}' \
    SOURCE_DATASET_DIR='${SOURCE_DATASET_DIR}' \
    SWEEP_SPECS='${SWEEP_SPECS}' \
    PROXY_PREDICTIONS='${PROXY_PREDICTIONS}' \
    INIT_FROM_MODEL_CHECKPOINT='${INIT_FROM_MODEL_CHECKPOINT}' \
    TRAIN_NPROC='${TRAIN_NPROC}' \
    MIXED_PRECISION='${MIXED_PRECISION}' \
    ACCELERATE_CONFIG='${ACCELERATE_CONFIG}' \
    MAX_SEQ_LENGTH='${MAX_SEQ_LENGTH}' \
    NUM_EPOCHS='${NUM_EPOCHS}' \
    BATCH_SIZE='${BATCH_SIZE}' \
    EVAL_BATCH_SIZE='${EVAL_BATCH_SIZE}' \
    GRADIENT_ACCUMULATION_STEPS='${GRADIENT_ACCUMULATION_STEPS}' \
    LR='${LR}' \
    WEIGHT_DECAY='${WEIGHT_DECAY}' \
    WARMUP_RATIO='${WARMUP_RATIO}' \
    MAX_TRAIN_ROWS='${MAX_TRAIN_ROWS}' \
    MAX_EVAL_ROWS='${MAX_EVAL_ROWS}' \
    SEED='${SEED}' \
    TARGET_ROUTE_IDXS='${TARGET_ROUTE_IDXS}' \
    CASCADE_ORDER='${CASCADE_ORDER}' \
    MAX_THRESHOLD_CANDIDATES='${MAX_THRESHOLD_CANDIDATES}' \
    LOSS_TYPE='${LOSS_TYPE}' \
    UTILITY_LAMBDAS='${UTILITY_LAMBDAS}' \
    MLP_HIDDEN_SIZE='${MLP_HIDDEN_SIZE}' \
    DROPOUT='${DROPOUT}' \
    TORCH_DTYPE='${TORCH_DTYPE}' \
    ATTN_IMPLEMENTATION='${ATTN_IMPLEMENTATION}' \
    LORA_R='${LORA_R}' \
    LORA_ALPHA='${LORA_ALPHA}' \
    LORA_DROPOUT='${LORA_DROPOUT}' \
    LORA_TARGET_MODULES='${LORA_TARGET_MODULES}' \
    CHECKPOINT_EVERY_EPOCH='${CHECKPOINT_EVERY_EPOCH}' \
    EPOCH_REPORT_EVERY='${EPOCH_REPORT_EVERY}' \
    SAVE_MODEL='${SAVE_MODEL}' \
    SUBRUN_SLEEP_SECS='${SUBRUN_SLEEP_SECS}' \
    bash ${REPO_ROOT}/launchers/offline_router/run_offline_router_swe_smith_active_label_proxy_init_sequential_sweep_inner.sh \
    2>&1 | tee -a ${OUTPUT_ROOT}/launch.out"

echo "Submitted sequential sweep: ${OUTPUT_ROOT}"
echo "Sweep specs: ${SWEEP_SPECS}"
