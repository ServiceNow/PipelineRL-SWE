#!/usr/bin/env bash
# Launch a single EAI job that runs Daytona eval for all filtered model predictions.
# Filters to the common instance-ID intersection first (locally), then submits
# one job that evaluates each model sequentially (Daytona handles per-instance
# parallelism internally via async sandboxes).
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
RUN_ID_PREFIX="or_sweep"

# --- Step 1: filter to common intersection (runs locally, not as a job) ---
echo "=== Filtering predictions to common intersection ==="
"${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/openrouter_sweep/filter_to_intersection.py" \
  --predictions-dir "${PREDICTIONS_DIR}" \
  --output-dir "${FILTERED_DIR}" \
  --exclude laguna
echo ""

# --- Step 2: build sequential eval command for all models ---
EVAL_CMDS=""
for jsonl_file in "${FILTERED_DIR}"/*.jsonl; do
  [[ -f "${jsonl_file}" ]] || continue
  slug=$(basename "${jsonl_file}" .jsonl)
  run_id="${RUN_ID_PREFIX}_${slug}"
  EVAL_CMDS="${EVAL_CMDS}
echo '=== Evaluating ${slug} ==='; \
python pipelinerl/swe/scripts/offline_router/run_swesmith_eval_daytona.py \
  --predictions_path ${jsonl_file} \
  --run_id ${run_id} \
  --concurrency ${CONCURRENCY} \
  2>&1 | tee ${FILTERED_DIR}/${slug}_daytona.log;"
done

if [[ -z "${EVAL_CMDS}" ]]; then
  echo "No filtered JSONL files found in ${FILTERED_DIR}. Did filtering succeed?"
  exit 1
fi

# --- Step 3: submit single EAI job ---
JOB_NAME="or_sweep_daytona_${TIMESTAMP}"
echo "=== Submitting Daytona eval job: ${JOB_NAME} ==="

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=1 \
  GPU=0 \
  GPU_MEM=0 \
  CPU=8 \
  CPU_MEM=64 \
  COMMAND="cd ${REPO_ROOT}; ${EVAL_CMDS}"

echo ""
echo "Daytona job submitted: ${JOB_NAME}"
echo "Reports land at:"
echo "  ${REPO_ROOT}/logs/run_evaluation/or_sweep_<slug>/report.json"
echo ""
echo "Once the job finishes, run the analysis:"
echo "  python pipelinerl/swe/scripts/openrouter_sweep/analyze_openrouter_sweep.py \\"
echo "    --daytona-log-dir ${REPO_ROOT}/logs/run_evaluation \\"
echo "    --run-id-prefix or_sweep \\"
echo "    --existing-parquet-dir <eval parquet dir> \\"
echo "    --output-dir ${FILTERED_DIR}/analysis"
