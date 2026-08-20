#!/usr/bin/env bash
# Collect a corrected LiveCodeBench dataset with a local 4B scout and an
# OpenRouter oracle. Both generations are graded with the pinned official LCB
# runner on public + private tests. The default split is temporal.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SCOUT_MODEL=${SCOUT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
ORACLE_MODEL=${ORACLE_MODEL:-openai/gpt-oss-120b}
RELEASE_VERSION=${RELEASE_VERSION:-release_v6}
MIN_DATE=${MIN_DATE:-2023-09-01}
TEMPORAL_CUTOFF=${TEMPORAL_CUTOFF:-2024-10-01}
MAX_SAMPLES=${MAX_SAMPLES:-0}
TEMPERATURE=${TEMPERATURE:-0.0}
CONCURRENCY=${CONCURRENCY:-16}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
SNAPSHOT=${SNAPSHOT:-1}
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

SCOUT_SLUG=$(echo "${SCOUT_MODEL}" | tr '/.-' '___' | tr '[:upper:]' '[:lower:]')
JOB_NAME=${JOB_NAME:-lcb_corrected_temporal_${SCOUT_SLUG}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

if [[ ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
  echo "Missing OpenRouter key at ${OPENROUTER_API_KEY_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
RUNNER="${OUTPUT_DIR}/run_collect.sh"
cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh

python -m vllm.entrypoints.openai.api_server \\
  --model '${SCOUT_MODEL}' \\
  --served-model-name '${SCOUT_MODEL}' \\
  --port ${VLLM_PORT} \\
  --tensor-parallel-size 1 \\
  --gpu-memory-utilization ${VLLM_GPU_UTIL} \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --trust-remote-code \\
  > '${OUTPUT_DIR}/vllm_server.log' 2>&1 &
VLLM_PID=\$!
trap 'kill \${VLLM_PID} 2>/dev/null || true' EXIT

ready=false
for _ in \$(seq 1 120); do
  if curl -sf 'http://localhost:${VLLM_PORT}/health' >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 \${VLLM_PID} 2>/dev/null; then
    echo "vLLM exited during startup; see ${OUTPUT_DIR}/vllm_server.log" >&2
    tail -100 '${OUTPUT_DIR}/vllm_server.log' >&2 || true
    exit 1
  fi
  sleep 5
done
if [[ "\${ready}" != "true" ]]; then
  echo "vLLM failed to become ready; see ${OUTPUT_DIR}/vllm_server.log" >&2
  exit 1
fi

python pipelinerl/swe/scripts/livecodebench/collect_lcb_trajectories.py \\
  --phase all \\
  --output-dir '${OUTPUT_DIR}' \\
  --scout-model '${SCOUT_MODEL}' \\
  --scout-base-url 'http://localhost:${VLLM_PORT}' \\
  --scout-api-key local \\
  --oracle-model '${ORACLE_MODEL}' \\
  --api-key-file '${OPENROUTER_API_KEY_FILE}' \\
  --release-version '${RELEASE_VERSION}' \\
  --min-date '${MIN_DATE}' \\
  --temporal-cutoff '${TEMPORAL_CUTOFF}' \\
  --max-samples ${MAX_SAMPLES} \\
  --temperature ${TEMPERATURE} \\
  --concurrency ${CONCURRENCY} \\
  --eval-timeout ${EVAL_TIMEOUT} \\
  --scout-feedback-tests public \\
  2>&1 | tee '${OUTPUT_DIR}/collect.log'
SCRIPT_EOF
chmod +x "${RUNNER}"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" \
  NPROC=1 \
  GPU=1 \
  GPU_MEM=24 \
  CPU=32 \
  CPU_MEM=64 \
  COMMAND="bash ${RUNNER}"

echo "Job:        ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
