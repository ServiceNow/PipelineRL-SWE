#!/usr/bin/env bash
# Launch Daytona eval jobs for all model JSONL files produced by launch_collect.sh.
# Submits one EAI job per model; they run in parallel.
#
# Before submitting, filters all prediction files to the common instance-ID
# intersection (excluding laguna), so every model is evaluated on the same set.
#
# Daytona writes to logs/run_evaluation/<run_id>/report.json (relative to REPO_ROOT).
# This script uses deterministic run_ids (or_sweep_<slug>) so the analysis script
# can find them without a manifest file.
#
# Usage:
#   PREDICTIONS_DIR=/mnt/.../openrouter_sweep_collect_XYZ bash launch_daytona.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3

TIMESTAMP=$(date +%s)

# Directory containing per-model *.jsonl prediction files from launch_collect.sh
PREDICTIONS_DIR=${PREDICTIONS_DIR:?Need PREDICTIONS_DIR set to the collect output dir}
FILTERED_DIR="${PREDICTIONS_DIR}/filtered"

CONCURRENCY=${CONCURRENCY:-50}

# Daytona reports land at: logs/run_evaluation/<run_id>/ (relative to REPO_ROOT)
# We use deterministic run_ids so the analysis script can find them.
RUN_ID_PREFIX="or_sweep"

# --- Step 1: filter to common intersection (runs locally, not as a job) ---
echo "=== Filtering predictions to common intersection ==="
"${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/openrouter_sweep/filter_to_intersection.py" \
  --predictions-dir "${PREDICTIONS_DIR}" \
  --output-dir "${FILTERED_DIR}" \
  --exclude laguna

echo ""

# --- Step 2: submit one Daytona job per filtered model JSONL ---
echo "=== Submitting Daytona eval jobs ==="
for jsonl_file in "${FILTERED_DIR}"/*.jsonl; do
  [[ -f "${jsonl_file}" ]] || continue
  slug=$(basename "${jsonl_file}" .jsonl)
  run_id="${RUN_ID_PREFIX}_${slug}"
  job_name="or_daytona_${slug:0:35}_${TIMESTAMP}"

  echo "[submit] slug=${slug}  run_id=${run_id}"

  make -C "${REPO_ROOT}" job \
    JOB_NAME="${job_name}" \
    ENV=pipeline-rl \
    CONDA_EXE=/opt/conda/bin/conda \
    SNAPSHOT=1 \
    NPROC=1 \
    GPU=0 \
    GPU_MEM=0 \
    CPU=8 \
    CPU_MEM=32 \
    COMMAND="cd ${REPO_ROOT}; \
      python pipelinerl/swe/scripts/offline_router/run_swesmith_eval_daytona.py \
        --predictions_path ${jsonl_file} \
        --run_id ${run_id} \
        --concurrency ${CONCURRENCY} \
      2>&1 | tee ${FILTERED_DIR}/${slug}_daytona.log"

  sleep 2  # Stagger submissions slightly
done

echo ""
echo "Daytona jobs submitted. Reports land at:"
echo "  ${REPO_ROOT}/logs/run_evaluation/or_sweep_<slug>/report.json"
echo ""
echo "Once all jobs finish, run the analysis:"
echo "  python pipelinerl/swe/scripts/openrouter_sweep/analyze_openrouter_sweep.py \\"
echo "    --daytona-log-dir ${REPO_ROOT}/logs/run_evaluation \\"
echo "    --run-id-prefix or_sweep \\"
echo "    --existing-parquet-dir <eval parquet dir> \\"
echo "    --output-dir ${FILTERED_DIR}/analysis"
