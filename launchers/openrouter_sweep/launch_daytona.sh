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

CONCURRENCY=${CONCURRENCY:-15}
RUN_ID_PREFIX="or_sweep"

# Load DAYTONA_API_KEY from .env if not already set
if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
  ENV_FILE="${REPO_ROOT}/.env"
  if [[ -f "${ENV_FILE}" ]]; then
    DAYTONA_API_KEY=$(grep -E '^DAYTONA_API_KEY=' "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"'"'" )
  fi
fi
: "${DAYTONA_API_KEY:?Need DAYTONA_API_KEY — set it in .env or the environment}"

# --- Step 1: filter to common intersection (runs locally, not as a job) ---
echo "=== Filtering predictions to common intersection ==="
"${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/openrouter_sweep/filter_to_intersection.py" \
  --predictions-dir "${PREDICTIONS_DIR}" \
  --output-dir "${FILTERED_DIR}" \
  --exclude laguna
echo ""

# --- Step 2: write a runner script into the filtered dir ---
RUNNER="${FILTERED_DIR}/run_daytona_eval.sh"
cat > "${RUNNER}" << 'SCRIPT_EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SCRIPT_EOF

for jsonl_file in "${FILTERED_DIR}"/*.jsonl; do
  [[ -f "${jsonl_file}" ]] || continue
  slug=$(basename "${jsonl_file}" .jsonl)
  run_id="${RUN_ID_PREFIX}_${slug}"
  cat >> "${RUNNER}" << SCRIPT_EOF
echo "=== Evaluating ${slug} ==="
python ${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/run_swesmith_eval_daytona.py \
  --predictions_path ${jsonl_file} \
  --run_id ${run_id} \
  --concurrency ${CONCURRENCY} \
  --redo_existing \
  2>&1 | tee ${FILTERED_DIR}/${slug}_daytona.log
SCRIPT_EOF
done

chmod +x "${RUNNER}"
echo "Runner script written to: ${RUNNER}"
echo ""

# --- Step 3: submit single EAI job that runs the script ---
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
  COMMAND="DAYTONA_API_KEY=${DAYTONA_API_KEY} bash ${RUNNER}"

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
