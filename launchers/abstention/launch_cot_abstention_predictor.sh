#!/usr/bin/env bash
# Train Qwen3-Embedding-8B with LoRA to predict strong-model success from
# the 4B CoT scout's thinking trace + patch (vs. patch-only baseline).
#
# Required env vars:
#   TRAJECTORIES_DIR  -- output dir from collect_cot_trajectories.py
#
# Optional:
#   REAL_LABEL_DATASET_DIR  -- parquet collection dir (default: existing 4-route run)
#   INCLUDE_THINKING        -- "true" (default) or "false" to ablate CoT
#   NUM_EPOCHS              -- default 10
#   LORA_R                  -- default 32
#
# Usage:
#   TRAJECTORIES_DIR=/mnt/.../cot_trajectories_XYZ \
#   bash launch_cot_abstention_predictor.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=$(date +%s)

TRAJECTORIES_DIR=${TRAJECTORIES_DIR:?Need TRAJECTORIES_DIR set to cot_trajectories output dir}
REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
TRAIN_PARQUET_DIR=${TRAIN_PARQUET_DIR:-${REAL_LABEL_DATASET_DIR}/train}
EVAL_PARQUET_DIR=${EVAL_PARQUET_DIR:-${REAL_LABEL_DATASET_DIR}/eval}

INCLUDE_THINKING=${INCLUDE_THINKING:-true}
INPUT_ONLY=${INPUT_ONLY:-false}
INCLUDE_TEST_FEEDBACK=${INCLUDE_TEST_FEEDBACK:-false}
TEST_FEEDBACK_FORMAT=${TEST_FEEDBACK_FORMAT:-full}  # full | names_only | count_only
MULTI_TASK_SCOUT=${MULTI_TASK_SCOUT:-false}
LABEL_ROUTE_IDX=${LABEL_ROUTE_IDX:-3}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
NPROC=${NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-2e-5}

if [[ "${INCLUDE_THINKING}" == "true" ]]; then
  THINKING_SUFFIX="cot"
  THINKING_ARG="--include-thinking"
else
  THINKING_SUFFIX="no_cot"
  THINKING_ARG="--no-include-thinking"
fi

if [[ "${MULTI_TASK_SCOUT}" == "true" ]]; then
  MULTITASK_SUFFIX="_mt"
  MULTITASK_ARG="--multi-task-scout"
else
  MULTITASK_SUFFIX=""
  MULTITASK_ARG=""
fi

if [[ "${INPUT_ONLY}" == "true" ]]; then
  INPUT_ONLY_SUFFIX="_input_only"
  INPUT_ONLY_ARG="--input-only"
else
  INPUT_ONLY_SUFFIX=""
  INPUT_ONLY_ARG=""
fi

if [[ "${INCLUDE_TEST_FEEDBACK}" == "true" ]]; then
  TEST_FB_SUFFIX="_testfb_${TEST_FEEDBACK_FORMAT}"
  TEST_FB_ARG="--include-test-feedback --test-feedback-format ${TEST_FEEDBACK_FORMAT}"
else
  TEST_FB_SUFFIX=""
  TEST_FB_ARG="--no-include-test-feedback"
fi

JOB_NAME=cot_abstention_qwen3_emb8b_lora_r${LORA_R}_${THINKING_SUFFIX}${MULTITASK_SUFFIX}${INPUT_ONLY_SUFFIX}${TEST_FB_SUFFIX}_route${LABEL_ROUTE_IDX}_${NUM_EPOCHS}epoch_${TIMESTAMP}
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}

RUNNER="${OUTPUT_DIR}/run_train.sh"
mkdir -p "${OUTPUT_DIR}"

TRAIN_CMD="python pipelinerl/swe/scripts/offline_router/train_cot_abstention_predictor.py"
if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${NPROC} \
    --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
    pipelinerl/swe/scripts/offline_router/train_cot_abstention_predictor.py"
fi

cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"

${TRAIN_CMD} \\
  --train-trajectories ${TRAJECTORIES_DIR}/trajectories_train.jsonl \\
  --eval-trajectories  ${TRAJECTORIES_DIR}/trajectories_eval.jsonl \\
  --train-parquet-dir  ${TRAIN_PARQUET_DIR} \\
  --eval-parquet-dir   ${EVAL_PARQUET_DIR} \\
  --output-dir         ${OUTPUT_DIR} \\
  --label-route-idx    ${LABEL_ROUTE_IDX} \\
  ${THINKING_ARG} \\
  ${MULTITASK_ARG} \\
  ${INPUT_ONLY_ARG} \\
  ${TEST_FB_ARG} \\
  --max-seq-length     ${MAX_SEQ_LENGTH} \\
  --num-epochs         ${NUM_EPOCHS} \\
  --batch-size         ${BATCH_SIZE} \\
  --eval-batch-size    ${BATCH_SIZE} \\
  --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \\
  --lr                 ${LR} \\
  --lora-r             ${LORA_R} \\
  --lora-alpha         ${LORA_ALPHA} \\
  --lora-target-modules ${LORA_TARGET_MODULES} \\
  --gradient-checkpointing \\
  --checkpoint-every-epoch \\
  2>&1 | tee ${OUTPUT_DIR}/train.log

echo "[done] Output: ${OUTPUT_DIR}"
SCRIPT_EOF
chmod +x "${RUNNER}"

echo "=== Submitting CoT abstention predictor: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${NPROC} \
  COMMAND="bash ${RUNNER}"

echo ""
echo "Output dir: ${OUTPUT_DIR}"
echo "Train log:  ${OUTPUT_DIR}/train.log"
echo "Eval preds: ${OUTPUT_DIR}/eval_predictions.jsonl"
echo ""
echo "To run the no-CoT ablation:"
echo "  INCLUDE_THINKING=false TRAJECTORIES_DIR=${TRAJECTORIES_DIR} bash $0"
