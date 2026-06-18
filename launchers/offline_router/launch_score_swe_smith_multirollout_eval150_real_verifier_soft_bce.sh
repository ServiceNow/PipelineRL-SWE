#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-score_swe_smith_multirollout_eval150_real_verifier_soft_bce}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
SCORE_OUTPUT_DIR=${SCORE_OUTPUT_DIR:-${OUTPUT_ROOT}/scores}
DATASET_DIR=${DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_multirollout_eval150_proxy_verifier_rescore_dataset_1781735193/collect}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_2route_oss20_vs_oss120_qwen3_embedding_8b_lora_verifier_soft_bce_r32_qkvo_mlp_10epoch_1781733167/train_qwen3_embedding_8b_lora_verifier_soft_bce_oss20_vs_oss120_r32_qkvo_mlp_10epoch}

JOB_NPROC=${JOB_NPROC:-1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
BATCH_SIZE=${BATCH_SIZE:-1}
TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-1,3}
ROUTE_ORDER=${ROUTE_ORDER:-0,1}
LOSS_TYPE=${LOSS_TYPE:-soft_bce}

if [[ ! -f "${DATASET_DIR}/metadata.json" ]]; then
  echo "Missing verifier rescore dataset: ${DATASET_DIR}/metadata.json" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_DIR}/scorer_head.pt" ]]; then
  echo "Missing verifier checkpoint scorer head: ${CHECKPOINT_DIR}/scorer_head.pt" >&2
  exit 1
fi

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT} ${SCORE_OUTPUT_DIR}; set -o pipefail; \
    python pipelinerl/swe/scripts/offline_router/score_qwen_embedding_cascade_verifier.py \
      --dataset-dir ${DATASET_DIR} \
      --checkpoint-dir ${CHECKPOINT_DIR} \
      --output-dir ${SCORE_OUTPUT_DIR} \
      --split eval \
      --target-route-idxs ${TARGET_ROUTE_IDXS} \
      --route-order ${ROUTE_ORDER} \
      --max-seq-length ${MAX_SEQ_LENGTH} \
      --batch-size ${BATCH_SIZE} \
      --loss-type ${LOSS_TYPE} \
      --device cuda \
      2>&1 | tee -a ${SCORE_OUTPUT_DIR}/launch.out"
