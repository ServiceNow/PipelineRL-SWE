#!/usr/bin/env bash
# TACO viability gate: is there a routing problem left in contaminated data?
#
# LiveCodeBench is exhausted at 1055 problems and the fold analysis shows savings
# scaling with training-set size, so data is the binding constraint. TACO has 26,443
# problems but predates our models, so it is contaminated.
#
# Contamination is fatal for EVALUATION. It is not obviously fatal for ROUTER
# TRAINING: the router predicts which model succeeds, not the answer, so memorisation
# only breaks it if the relative difficulty ordering across routes is distorted rather
# than uniformly shifted.
#
# Two numbers decide whether a full collection is worth it:
#   1. scout solve rate -- if the 4B has memorised TACO the difficulty distribution has
#      collapsed and there is nothing left to learn from;
#   2. route complementarity -- problems oss120 solves that scout does not. That gap IS
#      the routing signal; without it the data is useless however large it is.
#
# One draw per route over 300 problems, ~600 generations, enough to size both. The
# generation, extraction and grading path is identical to LCB: TACO's input_output is
# already the shape LCB normalises to, so nothing about scoring changes.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SCOUT_MODEL=${SCOUT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
ORACLE_MODEL=${ORACLE_MODEL:-openai/gpt-oss-120b}
PROBLEMS_FILE=${PROBLEMS_FILE:?Set PROBLEMS_FILE to the built TACO problems JSONL}
# The gate needs solve rates, not a split, but the collector requires both sides to
# be non-empty. 2018-01-01 splits the 300-problem sample 132/168; the split itself is
# not used for anything here.
TEMPORAL_CUTOFF=${TEMPORAL_CUTOFF:-2018-01-01}
MAX_SAMPLES=${MAX_SAMPLES:-300}
TEMPERATURE=${TEMPERATURE:-0.0}
CONCURRENCY=${CONCURRENCY:-16}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

SCOUT_SLUG=$(echo "${SCOUT_MODEL}" | tr '/.-' '___' | tr '[:upper:]' '[:lower:]')
JOB_NAME=${JOB_NAME:-taco_gate_${SCOUT_SLUG}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

if [[ ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
  echo "Missing OpenRouter key at ${OPENROUTER_API_KEY_FILE}" >&2
  exit 1
fi
if [[ ! -s "${PROBLEMS_FILE}" ]]; then
  echo "Missing TACO problems file at ${PROBLEMS_FILE}; build it with build_taco_problems.py" >&2
  exit 1
fi

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
  problems: ${PROBLEMS_FILE} (max ${MAX_SAMPLES})
  scout: ${SCOUT_MODEL} (local vLLM)   oracle: ${ORACLE_MODEL} (OpenRouter)
  output: ${OUTPUT_DIR}
  decides: scout solve rate, and whether oss120 solves what scout cannot

Submit explicitly with:
  SUBMIT=1 PROBLEMS_FILE=${PROBLEMS_FILE} bash ${BASH_SOURCE[0]}
EOF
  exit 0
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
  --problems-file '${PROBLEMS_FILE}' \\
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
