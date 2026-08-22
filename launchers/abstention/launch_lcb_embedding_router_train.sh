#!/usr/bin/env bash
# Train the repo-standard multi-route embedding router (one LoRA, N-dim output,
# objective=reward_bce) on LCB full-routing data, then score routing policies.
#
# Required env:
#   LCB_COLLECTION_DIR  -- corrected 4B/120B collection dir
#   LCB_OSS20_DIR       -- oss20 collection dir (launch_lcb_full_router_collect_oss20.sh output)
# Optional env:
#   INPUT_ONLY / INCLUDE_TEST_FEEDBACK / NUM_EPOCHS / NPROC / LORA_R / SEED
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${LCB_COLLECTION_DIR:?Set LCB_COLLECTION_DIR to the corrected 4B/120B collection}"
: "${LCB_OSS20_DIR:?Set LCB_OSS20_DIR to the oss20 collection output}"

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
INPUT_ONLY=${INPUT_ONLY:-false}
INCLUDE_TEST_FEEDBACK=${INCLUDE_TEST_FEEDBACK:-true}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LR=${LR:-1e-4}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
SEED=${SEED:-17}
NPROC=${NPROC:-4}
SNAPSHOT=${SNAPSHOT:-1}

if [[ "${INPUT_ONLY}" == "true" ]]; then
  VARIANT="inputonly"
  CONVERT_ARGS="--input-only"
  TRAIN_INPUT_MODE="--input-mode input_only"
else
  if [[ "${INCLUDE_TEST_FEEDBACK}" == "true" ]]; then
    VARIANT="postscout_testfb"
    CONVERT_ARGS="--include-test-feedback"
  else
    VARIANT="postscout"
    CONVERT_ARGS=""
  fi
  TRAIN_INPUT_MODE="--input-mode post_primary"
fi

JOB_NAME=${JOB_NAME:-lcb_embed_router_${VARIANT}_seed${SEED}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}
ROUTER_DATA_DIR=${OUTPUT_DIR}/router_data
DATASET_DIR=${OUTPUT_DIR}/router_dataset
MODEL_DIR=${OUTPUT_DIR}/model

if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_LAUNCH="python -m accelerate.commands.launch --multi_gpu --mixed_precision bf16 --num_processes ${NPROC} --config_file conf/accelerate/base_mp.yaml"
else
  TRAIN_LAUNCH="python"
fi

COMMAND="cd ${REPO_ROOT} && \
python pipelinerl/swe/scripts/livecodebench/materialize_lcb_full_router.py --source-collection-dir ${LCB_COLLECTION_DIR} --expert-collection-dir ${LCB_OSS20_DIR} --output-dir ${ROUTER_DATA_DIR} && \
python pipelinerl/swe/scripts/livecodebench/convert_lcb_router_to_dataset.py --router-data-dir ${ROUTER_DATA_DIR} --output-dir ${DATASET_DIR} ${CONVERT_ARGS} && \
${TRAIN_LAUNCH} pipelinerl/swe/scripts/offline_router/train_qwen_embedding_router_baseline.py \
  --dataset-dir ${DATASET_DIR} --output-dir ${MODEL_DIR} \
  --objective reward_bce --use-lora --no-encoder-frozen \
  --lora-r ${LORA_R} --lora-alpha ${LORA_ALPHA} \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lr ${LR} --num-epochs ${NUM_EPOCHS} --max-seq-length ${MAX_SEQ_LENGTH} \
  --seed ${SEED} --gradient-checkpointing --checkpoint-every-epoch \
  ${TRAIN_INPUT_MODE} && \
python pipelinerl/swe/scripts/livecodebench/adapt_baseline_preds_for_lcb_eval.py \
  --baseline-predictions ${MODEL_DIR}/eval_predictions.jsonl \
  --router-eval ${ROUTER_DATA_DIR}/router_eval.jsonl \
  --output ${MODEL_DIR}/eval_predictions_adapted.jsonl && \
python pipelinerl/swe/scripts/livecodebench/evaluate_lcb_full_router.py \
  --predictions ${MODEL_DIR}/eval_predictions_adapted.jsonl \
  --output-path ${OUTPUT_DIR}/policy_report.json"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" \
  NPROC=${NPROC} GPU=1 GPU_MEM=80 CPU=16 CPU_MEM=128 COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
