#!/usr/bin/env bash
# Sample efficiency sweep: proxy-init cascade verifier fine-tuned with N real labels.
# For each N in SWEEP_BUDGETS, materializes a random real-label subset, fine-tunes
# from the proxy checkpoint, then scores the 286-task eval set so we can compute
# abstention AUC vs N.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_proxy_init_real_finetune_sample_efficiency}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}
REAL_SOURCE_DATASET_DIR=${REAL_SOURCE_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
INIT_FROM_MODEL_CHECKPOINT=${INIT_FROM_MODEL_CHECKPOINT:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch_1781684950/train_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch/checkpoints/epoch_0004}

# Comma-separated list of real-label budgets to sweep over
SWEEP_BUDGETS=${SWEEP_BUDGETS:-50,100,200,500}

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
SEED=${SEED:-17}
CASCADE_ORDER=${CASCADE_ORDER:-0,1,2,3}
LOSS_TYPE=${LOSS_TYPE:-soft_bce}
UTILITY_LAMBDAS=${UTILITY_LAMBDAS:-0.0,1.0e-5,2.0e-5,5.0e-5,1.0e-4,2.0e-4}
MAX_THRESHOLD_CANDIDATES=${MAX_THRESHOLD_CANDIDATES:-21}
MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE:-1024}
DROPOUT=${DROPOUT:-0.1}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

if [[ ! -f "${REAL_SOURCE_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing real-label dataset: ${REAL_SOURCE_DATASET_DIR}/metadata.json" >&2
  exit 1
fi
if [[ ! -f "${INIT_FROM_MODEL_CHECKPOINT}/model.safetensors" ]]; then
  echo "Missing proxy init checkpoint: ${INIT_FROM_MODEL_CHECKPOINT}/model.safetensors" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; \
    export SWEEP_BUDGETS=${SWEEP_BUDGETS}; \
    export OUTPUT_ROOT=${OUTPUT_ROOT}; \
    export REAL_SOURCE_DATASET_DIR=${REAL_SOURCE_DATASET_DIR}; \
    export INIT_FROM_MODEL_CHECKPOINT=${INIT_FROM_MODEL_CHECKPOINT}; \
    export MODEL_NAME=${MODEL_NAME}; \
    export CASCADE_ORDER=${CASCADE_ORDER}; \
    export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}; \
    export NUM_EPOCHS=${NUM_EPOCHS}; \
    export BATCH_SIZE=${BATCH_SIZE}; \
    export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}; \
    export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}; \
    export LR=${LR}; \
    export WEIGHT_DECAY=${WEIGHT_DECAY}; \
    export WARMUP_RATIO=${WARMUP_RATIO}; \
    export SEED=${SEED}; \
    export DROPOUT=${DROPOUT}; \
    export MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE}; \
    export TORCH_DTYPE=${TORCH_DTYPE}; \
    export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}; \
    export LORA_R=${LORA_R}; \
    export LORA_ALPHA=${LORA_ALPHA}; \
    export LORA_DROPOUT=${LORA_DROPOUT}; \
    export LORA_TARGET_MODULES=${LORA_TARGET_MODULES}; \
    export MAX_THRESHOLD_CANDIDATES=${MAX_THRESHOLD_CANDIDATES}; \
    export LOSS_TYPE=${LOSS_TYPE}; \
    export UTILITY_LAMBDAS=${UTILITY_LAMBDAS}; \
    export TRAIN_NPROC=${TRAIN_NPROC}; \
    export MIXED_PRECISION=${MIXED_PRECISION}; \
    export ACCELERATE_CONFIG=${ACCELERATE_CONFIG}; \
    bash launchers/offline_router/sample_efficiency_sweep_inner.sh \
    2>&1 | tee -a ${OUTPUT_ROOT}/launch.out"

echo "Submitted: ${OUTPUT_ROOT}"
echo "Sweeping N = ${SWEEP_BUDGETS}"
