#!/usr/bin/env bash
# Evaluate CoT scout predictions via Daytona.
#
# Two purposes:
#   1. Measure whether CoT traces improve the scout's own patch quality
#      (paper comparison: scout-no-think vs scout-CoT)
#   2. Provide per-instance scout success labels (optional — training labels
#      come from the parquet gpt-oss-120b column, not from here)
#
# Steps:
#   1. Convert search/replace text → git diffs in-place (local, fast)
#   2. Write runner script
#   3. Submit Daytona eval job
#
# Prerequisites:
#   - collect_cot_trajectories.py has completed (TRAJECTORIES_DIR exists)
#   - DAYTONA_API_KEY is set (or in .env)
#
# Usage:
#   TRAJECTORIES_DIR=/mnt/.../cot_trajectories_XYZ \
#   bash launch_daytona_cot_predictions.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3

TIMESTAMP=$(date +%s)

TRAJECTORIES_DIR=${TRAJECTORIES_DIR:?Need TRAJECTORIES_DIR set to cot_trajectories output dir}
DATASET_PATH=${DATASET_PATH:-/mnt/llmd/data/swe_smith_bugged_context/ds_train}
CONCURRENCY=${CONCURRENCY:-8}

# Load DAYTONA_API_KEY from .env if not already set
if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
  ENV_FILE="${REPO_ROOT}/.env"
  if [[ -f "${ENV_FILE}" ]]; then
    DAYTONA_API_KEY=$(grep -E '^DAYTONA_API_KEY=' "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"'"'" )
  fi
fi
: "${DAYTONA_API_KEY:?Need DAYTONA_API_KEY — set it in .env or the environment}"

# --- Step 1: convert search/replace text → git diffs (runs locally) ---
echo "=== Converting CoT predictions to git diffs ==="
for split in train eval; do
  PRED_FILE="${TRAJECTORIES_DIR}/predictions_${split}.jsonl"
  if [[ -f "${PRED_FILE}" ]]; then
    echo "  Converting ${split}..."
    "${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/openrouter_sweep/convert_text_to_patches.py" \
      --predictions-dir "${TRAJECTORIES_DIR}" \
      --dataset-path "${DATASET_PATH}"
    # convert_text_to_patches processes all JSONLs in the dir; break after first to avoid double-run
    break
  fi
done
echo ""

# --- Step 2: write runner script ---
JOB_NAME=cot_daytona_${TIMESTAMP}
RUNNER="${TRAJECTORIES_DIR}/run_daytona_eval.sh"

cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
SCRIPT_EOF

for split in train eval; do
  PRED_FILE="${TRAJECTORIES_DIR}/predictions_${split}.jsonl"
  if [[ -f "${PRED_FILE}" ]]; then
    RUN_ID="cot_${split}_${TIMESTAMP}"
    cat >> "${RUNNER}" << SCRIPT_EOF
echo "=== Evaluating CoT scout predictions (${split}) ==="
python ${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/run_swesmith_eval_daytona.py \
  --predictions_path ${PRED_FILE} \
  --run_id ${RUN_ID} \
  --concurrency ${CONCURRENCY} \
  --redo_existing \
  2>&1 | tee ${TRAJECTORIES_DIR}/daytona_${split}.log
SCRIPT_EOF
  fi
done

chmod +x "${RUNNER}"

# --- Step 3: submit job ---
echo "=== Submitting CoT Daytona eval job: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=1 \
  GPU=0 \
  GPU_MEM=0 \
  CPU=8 \
  CPU_MEM=32 \
  COMMAND="DAYTONA_API_KEY=${DAYTONA_API_KEY} bash ${RUNNER}"

echo ""
echo "Daytona results land at:"
echo "  logs/run_evaluation/cot_train_${TIMESTAMP}/"
echo "  logs/run_evaluation/cot_eval_${TIMESTAMP}/"
echo "Daytona logs:"
echo "  ${TRAJECTORIES_DIR}/daytona_train.log"
echo "  ${TRAJECTORIES_DIR}/daytona_eval.log"
