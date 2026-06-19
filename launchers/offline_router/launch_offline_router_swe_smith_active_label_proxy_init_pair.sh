#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

LOCAL_PYTHON=${LOCAL_PYTHON:-/home/toolkit/.conda/envs/pipeline-rl/bin/python}
SOURCE_DATASET_DIR=${SOURCE_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
BUDGET_INSTANCES=${BUDGET_INSTANCES:-128}
ACTIVE_STRATEGY=${ACTIVE_STRATEGY:-uncertainty}
SEED=${SEED:-17}
SUBMIT_SLEEP_SECS=${SUBMIT_SLEEP_SECS:-20}
NUM_EPOCHS=${NUM_EPOCHS:-5}
LR=${LR:-1.0e-5}
RUN_ROOT=${RUN_ROOT:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_active_label_proxy_init_finetune_${TIMESTAMP}}
DATASET_ROOT=${DATASET_ROOT:-${RUN_ROOT}/datasets}
INIT_FROM_MODEL_CHECKPOINT=${INIT_FROM_MODEL_CHECKPOINT:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch_1781684950/train_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch/checkpoints/epoch_0004}

RANDOM_DATASET_DIR=${RANDOM_DATASET_DIR:-${DATASET_ROOT}/random_${BUDGET_INSTANCES}_seed${SEED}}
ACTIVE_DATASET_DIR=${ACTIVE_DATASET_DIR:-${DATASET_ROOT}/${ACTIVE_STRATEGY}_${BUDGET_INSTANCES}_seed${SEED}}

mkdir -p "${DATASET_ROOT}"

"${LOCAL_PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/materialize_active_real_label_verifier_subset.py"   --source-dataset-dir "${SOURCE_DATASET_DIR}"   --output-dir "${RANDOM_DATASET_DIR}"   --strategy random   --budget-instances "${BUDGET_INSTANCES}"   --seed "${SEED}"

"${LOCAL_PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/materialize_active_real_label_verifier_subset.py"   --source-dataset-dir "${SOURCE_DATASET_DIR}"   --output-dir "${ACTIVE_DATASET_DIR}"   --strategy "${ACTIVE_STRATEGY}"   --budget-instances "${BUDGET_INSTANCES}"   --seed "${SEED}"

REAL_LABEL_DATASET_DIR="${RANDOM_DATASET_DIR}" JOB_NAME="offline_router_swe_smith_real4route_proxyinit_random${BUDGET_INSTANCES}_${NUM_EPOCHS}epoch" OUTPUT_ROOT="${RUN_ROOT}/random_${BUDGET_INSTANCES}" TRAIN_OUTPUT_DIR="${RUN_ROOT}/random_${BUDGET_INSTANCES}/train_qwen3_embedding_8b_lora_verifier_proxy_init_real_random_${NUM_EPOCHS}epoch" INIT_FROM_MODEL_CHECKPOINT="${INIT_FROM_MODEL_CHECKPOINT}" NUM_EPOCHS="${NUM_EPOCHS}" LR="${LR}" SEED="${SEED}" bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_real_verifier_proxy_init_finetune.sh"

sleep "${SUBMIT_SLEEP_SECS}"

REAL_LABEL_DATASET_DIR="${ACTIVE_DATASET_DIR}" JOB_NAME="offline_router_swe_smith_real4route_proxyinit_${ACTIVE_STRATEGY}${BUDGET_INSTANCES}_${NUM_EPOCHS}epoch" OUTPUT_ROOT="${RUN_ROOT}/${ACTIVE_STRATEGY}_${BUDGET_INSTANCES}" TRAIN_OUTPUT_DIR="${RUN_ROOT}/${ACTIVE_STRATEGY}_${BUDGET_INSTANCES}/train_qwen3_embedding_8b_lora_verifier_proxy_init_real_${ACTIVE_STRATEGY}_${NUM_EPOCHS}epoch" INIT_FROM_MODEL_CHECKPOINT="${INIT_FROM_MODEL_CHECKPOINT}" NUM_EPOCHS="${NUM_EPOCHS}" LR="${LR}" SEED="${SEED}" bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_real_verifier_proxy_init_finetune.sh"

echo "Run root: ${RUN_ROOT}"
echo "Random dataset: ${RANDOM_DATASET_DIR}"
echo "Active dataset: ${ACTIVE_DATASET_DIR}"
