#!/usr/bin/env bash
# Two-stage sample efficiency sweep for the POST-SCOUT ROUTER:
#   Stage 1: proxy pre-training on trace-expanded dataset (cheap, proxy rewards)
#   Stage 2: fine-tune from proxy checkpoint on N real labels for each N in SWEEP_BUDGETS
#
# Produces an abstention AUC vs N curve comparable to the cascade verifier sweep,
# but for the post-scout router architecture (train_qwen_embedding_router_baseline.py).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_proxy_init_real_finetune_router_sample_efficiency}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}
TRACE_DATASET_DIR=${TRACE_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_cost_4route_expanded_1781073985/collect}
REAL_SOURCE_DATASET_DIR=${REAL_SOURCE_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}

# Comma-separated list of real-label budgets for stage 2
SWEEP_BUDGETS=${SWEEP_BUDGETS:-50,100,200,500,1000}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}

# Stage 1: proxy pre-training
PROXY_NUM_EPOCHS=${PROXY_NUM_EPOCHS:-5}
PROXY_LR=${PROXY_LR:-2.0e-5}
PROXY_OBJECTIVE=${PROXY_OBJECTIVE:-reward_bce}

# Stage 2: real-label fine-tuning
REAL_NUM_EPOCHS=${REAL_NUM_EPOCHS:-10}
REAL_LR=${REAL_LR:-1.0e-5}
REAL_OBJECTIVE=${REAL_OBJECTIVE:-reward_bce_delta_aux}
DELTA_AUX_WEIGHT=${DELTA_AUX_WEIGHT:-1.0}

# Shared hyperparameters
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
SEED=${SEED:-17}
MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE:-1024}
DROPOUT=${DROPOUT:-0.1}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

if [[ ! -f "${TRACE_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing trace dataset: ${TRACE_DATASET_DIR}/metadata.json" >&2
  exit 1
fi
if [[ ! -f "${REAL_SOURCE_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing real-label dataset: ${REAL_SOURCE_DATASET_DIR}/metadata.json" >&2
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
    export TRACE_DATASET_DIR=${TRACE_DATASET_DIR}; \
    export REAL_SOURCE_DATASET_DIR=${REAL_SOURCE_DATASET_DIR}; \
    export MODEL_NAME=${MODEL_NAME}; \
    export TRAIN_NPROC=${TRAIN_NPROC}; \
    export MIXED_PRECISION=${MIXED_PRECISION}; \
    export ACCELERATE_CONFIG=${ACCELERATE_CONFIG}; \
    export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}; \
    export PROXY_NUM_EPOCHS=${PROXY_NUM_EPOCHS}; \
    export PROXY_LR=${PROXY_LR}; \
    export PROXY_OBJECTIVE=${PROXY_OBJECTIVE}; \
    export REAL_NUM_EPOCHS=${REAL_NUM_EPOCHS}; \
    export REAL_LR=${REAL_LR}; \
    export REAL_OBJECTIVE=${REAL_OBJECTIVE}; \
    export DELTA_AUX_WEIGHT=${DELTA_AUX_WEIGHT}; \
    export BATCH_SIZE=${BATCH_SIZE}; \
    export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}; \
    export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}; \
    export WEIGHT_DECAY=${WEIGHT_DECAY}; \
    export WARMUP_RATIO=${WARMUP_RATIO}; \
    export SEED=${SEED}; \
    export MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE}; \
    export DROPOUT=${DROPOUT}; \
    export TORCH_DTYPE=${TORCH_DTYPE}; \
    export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}; \
    export LORA_R=${LORA_R}; \
    export LORA_ALPHA=${LORA_ALPHA}; \
    export LORA_DROPOUT=${LORA_DROPOUT}; \
    export LORA_TARGET_MODULES=${LORA_TARGET_MODULES}; \
    bash launchers/offline_router/proxy_init_real_finetune_router_sweep_inner.sh \
    2>&1 | tee -a ${OUTPUT_ROOT}/launch.out"

echo "Submitted: ${OUTPUT_ROOT}"
echo "Sweeping N = ${SWEEP_BUDGETS}"
