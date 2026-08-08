#!/usr/bin/env bash
# Re-collect LCB scout trajectories using local vLLM (phase=scout only).
#
# Use this when the OpenRouter LCB collection produced empty scout trajectories
# (Qwen3-4B models are not available on OpenRouter).  The oracle phase runs
# separately on OpenRouter; this job only covers the scout phase.
#
# If an EXISTING_OUTPUT_DIR is given, scout output is written there (re-using
# the oracle labels from a previous run).  Otherwise a new output dir is made.
#
# Required env vars:
#   SCOUT_MODEL       -- HuggingFace model ID (e.g. Qwen/Qwen3-4B-Instruct-2507)
#
# Optional env vars:
#   EXISTING_OUTPUT_DIR  -- reuse oracle data from a prior run; write scout there
#   MAX_SAMPLES          -- default: 500
#   MIN_DATE             -- default: 2023-09-01
#   CONCURRENCY          -- default: 16
#   VLLM_PORT            -- default: 8000
#   VLLM_GPU_UTIL        -- default: 0.90
#   MAX_MODEL_LEN        -- default: 32768
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

SCOUT_MODEL=${SCOUT_MODEL:?Need SCOUT_MODEL env var (e.g. Qwen/Qwen3-4B-Instruct-2507)}
MAX_SAMPLES=${MAX_SAMPLES:-500}
MIN_DATE=${MIN_DATE:-2023-09-01}
CONCURRENCY=${CONCURRENCY:-16}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SCOUT_SLUG=$(echo "${SCOUT_MODEL}" | tr '/.-' '___' | tr '[:upper:]' '[:lower:]')
JOB_NAME=${JOB_NAME:-lcb_scout_vllm_${SCOUT_SLUG}_${TIMESTAMP}}

if [[ -n "${EXISTING_OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="${EXISTING_OUTPUT_DIR}"
  echo "=== Reusing existing oracle data from ${OUTPUT_DIR} ==="
else
  OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}
fi

OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

mkdir -p "${OUTPUT_DIR}"

RUNNER="${OUTPUT_DIR}/run_scout_vllm.sh"
cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo '[vllm] Starting vLLM server for ${SCOUT_MODEL}...'
python -m vllm.entrypoints.openai.api_server \
  --model ${SCOUT_MODEL} \
  --port ${VLLM_PORT} \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization ${VLLM_GPU_UTIL} \
  --max-model-len ${MAX_MODEL_LEN} \
  --trust-remote-code \
  --served-model-name ${SCOUT_MODEL} \
  > ${OUTPUT_DIR}/vllm_server.log 2>&1 &
VLLM_PID=\$!

echo "[vllm] Waiting for server (PID=\${VLLM_PID})..."
for i in \$(seq 1 120); do
  if curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
    echo '[vllm] Server ready.'; break
  fi
  sleep 5
done

echo '[collect] Running scout phase via local vLLM...'
python pipelinerl/swe/scripts/livecodebench/collect_lcb_trajectories.py \
  --phase          scout \
  --output-dir     ${OUTPUT_DIR} \
  --scout-model    '${SCOUT_MODEL}' \
  --scout-base-url http://localhost:${VLLM_PORT} \
  --scout-api-key  local \
  --api-key-file   '${OPENROUTER_API_KEY_FILE}' \
  --max-samples    ${MAX_SAMPLES} \
  --min-date       '${MIN_DATE}' \
  --concurrency    ${CONCURRENCY} \
  2>&1 | tee "${OUTPUT_DIR}/scout_collect.log"

echo '[vllm] Shutting down...'
kill \${VLLM_PID} 2>/dev/null || true

echo "[done] Scout trajectories: ${OUTPUT_DIR}/trajectories_{train,eval}.jsonl"
SCRIPT_EOF
chmod +x "${RUNNER}"

echo "=== Submitting LCB scout (vLLM) job: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=1 \
  GPU=1 \
  GPU_MEM=24 \
  CPU=8 \
  CPU_MEM=32 \
  COMMAND="bash ${RUNNER}"

echo ""
echo "Job:        ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Scout log:  ${OUTPUT_DIR}/scout_collect.log"
