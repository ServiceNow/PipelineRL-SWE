#!/usr/bin/env bash
# Collect LiveCodeBench scout + oracle trajectories for abstention predictor.
#
# Phase 1 (scout):  cheap 4B model generates solutions via OpenRouter
# Phase 2 (oracle): oss-120b generates solutions + evaluates locally
# Phase 3 (train):  abstention predictor trained on (problem + scout) -> oracle success
#
# Optional env vars:
#   SCOUT_MODEL          -- OpenRouter model ID for scout (default: openai/gpt-oss-4b)
#   ORACLE_MODEL         -- OpenRouter model ID for oracle (default: openai/gpt-oss-120b)
#   MAX_SAMPLES          -- max LCB problems to use (0 = all, default: 500)
#   MIN_DATE             -- only problems on/after this date (default: 2023-09-01)
#   CONCURRENCY          -- async concurrency (default: 16)
#   PHASE                -- scout | oracle | eval | all (default: all)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SCOUT_MODEL=${SCOUT_MODEL:-openai/gpt-oss-4b}
ORACLE_MODEL=${ORACLE_MODEL:-openai/gpt-oss-120b}
MAX_SAMPLES=${MAX_SAMPLES:-500}
MIN_DATE=${MIN_DATE:-2023-09-01}
CONCURRENCY=${CONCURRENCY:-16}
PHASE=${PHASE:-all}

SCOUT_SLUG=$(echo "${SCOUT_MODEL}" | tr '/.-' '___')
JOB_NAME=${JOB_NAME:-lcb_collect_${SCOUT_SLUG}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

if [[ ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
  echo "Missing OpenRouter key at OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

RUNNER="${OUTPUT_DIR}/run_collect.sh"
cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "=== LCB collection: phase=${PHASE}, scout=${SCOUT_MODEL}, oracle=${ORACLE_MODEL} ==="

python pipelinerl/swe/scripts/livecodebench/collect_lcb_trajectories.py \\
  --phase          ${PHASE} \\
  --output-dir     ${OUTPUT_DIR} \\
  --scout-model    '${SCOUT_MODEL}' \\
  --oracle-model   '${ORACLE_MODEL}' \\
  --api-key-file   '${OPENROUTER_API_KEY_FILE}' \\
  --max-samples    ${MAX_SAMPLES} \\
  --min-date       '${MIN_DATE}' \\
  --concurrency    ${CONCURRENCY} \\
  --use-private-tc \\
  2>&1 | tee "${OUTPUT_DIR}/collect.log"

echo "[done] Output: ${OUTPUT_DIR}"
SCRIPT_EOF
chmod +x "${RUNNER}"

echo "=== Submitting LCB collection job: ${JOB_NAME} ==="
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
  COMMAND="bash ${RUNNER}"

echo ""
echo "Job:        ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log:        ${OUTPUT_DIR}/collect.log"
