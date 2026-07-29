#!/usr/bin/env bash
# Run the trained LoRA verifier on eval trajectories to produce per-instance P(Yes) scores.
#
# Prerequisites:
#   - train_autoregressive_verifier.py has completed (LORA_ADAPTER_DIR exists)
#   - collect_cot_trajectories.py has completed (TRAJECTORIES_DIR/trajectories_eval.jsonl exists)
#
# Usage:
#   LORA_ADAPTER_DIR=/mnt/.../autoreg_verifier_train_XYZ \
#   TRAJECTORIES_DIR=/mnt/.../cot_trajectories_YYY \
#   bash launch_score_autoregressive_verifier.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=$(date +%s)
JOB_NAME=autoreg_verifier_score_${TIMESTAMP}

LORA_ADAPTER_DIR=${LORA_ADAPTER_DIR:?Need LORA_ADAPTER_DIR set to trained LoRA adapter dir}
TRAJECTORIES_DIR=${TRAJECTORIES_DIR:?Need TRAJECTORIES_DIR set to cot_trajectories output dir}
EVAL_TRAJECTORIES_PATH=${EVAL_TRAJECTORIES_PATH:-${TRAJECTORIES_DIR}/trajectories_eval.jsonl}

REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
EVAL_PARQUET_DIR=${EVAL_PARQUET_DIR:-${REAL_LABEL_DATASET_DIR}/eval}
LABEL_ROUTE_IDX=${LABEL_ROUTE_IDX:-3}

OUTPUT_DIR=${OUTPUT_DIR:-${LORA_ADAPTER_DIR}}
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/eval_verifier_scores.jsonl}

BASE_MODEL_NAME=${BASE_MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}

# --- Write runner script ---
RUNNER="${OUTPUT_DIR}/run_verifier_score.sh"
mkdir -p "${OUTPUT_DIR}"

cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"

python pipelinerl/swe/scripts/offline_router/score_autoregressive_verifier.py \\
  --lora-adapter-dir ${LORA_ADAPTER_DIR} \\
  --base-model-name ${BASE_MODEL_NAME} \\
  --trajectories-path ${EVAL_TRAJECTORIES_PATH} \\
  --labels-parquet-dir ${EVAL_PARQUET_DIR} \\
  --label-route-idx ${LABEL_ROUTE_IDX} \\
  --output-path ${OUTPUT_PATH} \\
  --max-seq-length ${MAX_SEQ_LENGTH} \\
  --include-thinking \\
  2>&1 | tee ${OUTPUT_DIR}/score.log

echo "[done] Scores written to ${OUTPUT_PATH}"
SCRIPT_EOF
chmod +x "${RUNNER}"

echo "=== Submitting verifier scoring job: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=1 \
  GPU=1 \
  GPU_MEM=80 \
  CPU=8 \
  CPU_MEM=64 \
  COMMAND="bash ${RUNNER}"

echo ""
echo "Scores output: ${OUTPUT_PATH}"
echo "Score log:     ${OUTPUT_DIR}/score.log"
