#!/usr/bin/env bash
# Train the autoregressive Yes/No verifier (LoRA) on CoT trajectories.
#
# Labels come from the existing real-label parquet (route_successes[LABEL_ROUTE_IDX]).
# Default: LABEL_ROUTE_IDX=3 = gpt-oss-120b.
#
# Prerequisites:
#   - collect_cot_trajectories.py has run (produces trajectories_{train,eval}.jsonl)
#   - Parquet dataset exists at REAL_LABEL_DATASET_DIR/train and .../eval
#
# Usage:
#   TRAJECTORIES_DIR=/mnt/.../cot_trajectories_XYZ \
#   bash launch_autoregressive_verifier_train.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=$(date +%s)
JOB_NAME=autoreg_verifier_train_${TIMESTAMP}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

TRAJECTORIES_DIR=${TRAJECTORIES_DIR:?Need TRAJECTORIES_DIR set to cot_trajectories output dir}
TRAIN_TRAJECTORIES_PATH=${TRAIN_TRAJECTORIES_PATH:-${TRAJECTORIES_DIR}/trajectories_train.jsonl}
EVAL_TRAJECTORIES_PATH=${EVAL_TRAJECTORIES_PATH:-${TRAJECTORIES_DIR}/trajectories_eval.jsonl}

REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
TRAIN_PARQUET_DIR=${TRAIN_PARQUET_DIR:-${REAL_LABEL_DATASET_DIR}/train}
EVAL_PARQUET_DIR=${EVAL_PARQUET_DIR:-${REAL_LABEL_DATASET_DIR}/eval}
LABEL_ROUTE_IDX=${LABEL_ROUTE_IDX:-3}  # 3 = gpt-oss-120b

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}

NPROC=${NPROC:-2}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

# --- Write runner script (avoids make $ expansion issues) ---
mkdir -p "${OUTPUT_DIR}"
RUNNER="${OUTPUT_DIR}/run_verifier_train.sh"

TRAIN_CMD="python pipelinerl/swe/scripts/offline_router/train_autoregressive_verifier.py"
if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${NPROC} \
    pipelinerl/swe/scripts/offline_router/train_autoregressive_verifier.py"
fi

EVAL_ARGS=""
if [[ -f "${EVAL_TRAJECTORIES_PATH}" ]]; then
  EVAL_ARGS="--eval-trajectories-path ${EVAL_TRAJECTORIES_PATH} --eval-labels-parquet-dir ${EVAL_PARQUET_DIR}"
fi

cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

${TRAIN_CMD} \\
  --trajectories-path ${TRAIN_TRAJECTORIES_PATH} \\
  --labels-parquet-dir ${TRAIN_PARQUET_DIR} \\
  --label-route-idx ${LABEL_ROUTE_IDX} \\
  ${EVAL_ARGS} \\
  --output-dir ${OUTPUT_DIR} \\
  --model-name ${MODEL_NAME} \\
  --lora-rank ${LORA_RANK} \\
  --lora-alpha ${LORA_ALPHA} \\
  --num-epochs ${NUM_EPOCHS} \\
  --batch-size ${BATCH_SIZE} \\
  --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \\
  --gradient-checkpointing \\
  --learning-rate ${LEARNING_RATE} \\
  --max-seq-length ${MAX_SEQ_LENGTH} \\
  --include-thinking \\
  2>&1 | tee ${OUTPUT_DIR}/train.log
SCRIPT_EOF
chmod +x "${RUNNER}"

echo "=== Submitting verifier training job: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${NPROC} \
  GPU=1 \
  GPU_MEM=80 \
  CPU=16 \
  CPU_MEM=128 \
  COMMAND="bash ${RUNNER}"

echo ""
echo "LoRA adapter will be saved to: ${OUTPUT_DIR}"
echo "Training log:                  ${OUTPUT_DIR}/train.log"
