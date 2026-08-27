#!/usr/bin/env bash
# Multi-draw stochastic collection for the thread-(a) MDP: correctness tensors
# (problem x model x draw) on the LCB temporal eval split.
#
# TWO separate jobs (submit each mode):
#   MODE=scout    GPU job: local vLLM serves the scout; k draws x {T=0.2 primary,
#                 T=0.6 sensitivity}. Qwen models are NOT on OpenRouter.
#   MODE=experts  CPU job: oss20 + oss120 via OpenRouter, k draws x T=0.2.
#
# ONE grading process per job: the official LCB evaluator forks a process per
# call and concurrent forks corrupt labels, so all grading stays sequential
# inside a single EAI job.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${LCB_COLLECTION_DIR:?Set LCB_COLLECTION_DIR to the corrected 4B/120B collection}"
MODE=${MODE:?Set MODE=scout or MODE=experts}
: "${OPENROUTER_API_KEY_FILE:=/home/toolkit/.secrets/openrouter_api_key}"

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
NUM_DRAWS=${NUM_DRAWS:-10}
# Asymmetric draw budget. Scout success is exactly zero past failure depth 5
# (0 positives in 1,097 held-out states), so draws 6-10 of the scout are dead
# spend; oss120 re-rolls are worth +14.4 points. Spend the difference on problems.
SCOUT_DRAWS=${SCOUT_DRAWS:-${NUM_DRAWS}}
EXPERT_DRAWS=${EXPERT_DRAWS:-${NUM_DRAWS}}
# SPLITS=eval reproduces the original 341-problem collection; SPLITS=train adds
# the 551 already-collected-for-scout problems that the MDP has never used.
SPLITS=${SPLITS:-eval}
# The T=0.6 scout sensitivity arm doubles scout cost; off by default for new splits.
SCOUT_T06=${SCOUT_T06:-1}
TEMP_PRIMARY=${TEMP_PRIMARY:-0.2}
CONCURRENCY=${CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-4096}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
GEN_TIMEOUT=${GEN_TIMEOUT:-600}
MAX_INVALID_FRAC=${MAX_INVALID_FRAC:-0.05}
SNAPSHOT=${SNAPSHOT:-1}
VLLM_PORT=${VLLM_PORT:-8000}

SCOUT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
JOB_NAME=${JOB_NAME:-lcb_multidraw_${MODE}_${SPLITS}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

COLLECT="python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py \
  --source-collection-dir ${LCB_COLLECTION_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --splits ${SPLITS} --concurrency ${CONCURRENCY} \
  --max-tokens ${MAX_TOKENS} --eval-timeout ${EVAL_TIMEOUT} --gen-timeout ${GEN_TIMEOUT} \
  --max-invalid-frac ${MAX_INVALID_FRAC}"

if [[ "${MODE}" == "scout" ]]; then
  RUNNER="${OUTPUT_DIR}/run_multidraw_scout.sh"
  mkdir -p "${OUTPUT_DIR}"

  # Build inner loop unrolled (dash-safe)
  INNER="echo local > ${OUTPUT_DIR}/local_key.txt && export OPENROUTER_API_KEY=local && export HF_HUB_DISABLE_IMPLICIT_TOKEN=1"
  LOCAL_KEY="${OUTPUT_DIR}/local_key.txt"
  for DRAW in $(seq 0 $((SCOUT_DRAWS-1))); do
    INNER="${INNER} && echo '=== scout T0.2 draw ${DRAW} ===' && ${COLLECT} --api-key-file ${LOCAL_KEY} --base-url http://localhost:${VLLM_PORT} --route-label scout --model '${SCOUT_MODEL}' --temperature 0.2 --output-suffix _d${DRAW}"
  done
  for DRAW in $(seq 0 $(( SCOUT_T06 == 1 ? SCOUT_DRAWS-1 : -1 ))); do
    INNER="${INNER} && echo '=== scout T0.6 draw ${DRAW} ===' && ${COLLECT} --api-key-file ${LOCAL_KEY} --base-url http://localhost:${VLLM_PORT} --route-label scout06 --model '${SCOUT_MODEL}' --temperature 0.6 --output-suffix _d${DRAW}"
  done

  cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

echo '[vllm] Starting vLLM server for ${SCOUT_MODEL}...'
python -m vllm.entrypoints.openai.api_server \\
  --model ${SCOUT_MODEL} --port ${VLLM_PORT} --tensor-parallel-size 1 \\
  --gpu-memory-utilization 0.90 --max-model-len 32768 --trust-remote-code \\
  --served-model-name ${SCOUT_MODEL} > ${OUTPUT_DIR}/vllm_server.log 2>&1 &
VLLM_PID=\$!
for i in \$(seq 1 120); do
  curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1 && break
  sleep 5
done
echo '[vllm] ready'

${INNER}

kill \${VLLM_PID} 2>/dev/null || true
echo '[done]'
SCRIPT_EOF
  chmod +x "${RUNNER}"
  make -C "${REPO_ROOT}" job JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
    SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=1 GPU_MEM=24 CPU=8 CPU_MEM=32 COMMAND="bash ${RUNNER}"

elif [[ "${MODE}" == "experts" ]]; then
  JOB_NAME=${JOB_NAME:-lcb_multidraw_experts_${SPLITS}_${TIMESTAMP}}
  OUTPUT_DIR=${OUTPUT_DIR}/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}
  OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}
  COMMAND="cd ${REPO_ROOT} && source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh && export OPENROUTER_API_KEY=\$(cat ${OPENROUTER_API_KEY_FILE})"
  for DRAW in $(seq 0 $((EXPERT_DRAWS-1))); do
    for PAIR in "oss20:openai/gpt-oss-20b" "oss120:openai/gpt-oss-120b"; do
      ROUTE="${PAIR%%:*}"; MODEL="${PAIR##*:}"
      COMMAND="${COMMAND} && echo '=== ${ROUTE} draw ${DRAW} ===' && ${COLLECT} --route-label ${ROUTE} --model '${MODEL}' --temperature ${TEMP_PRIMARY} --output-suffix _d${DRAW} --api-key-file ${OPENROUTER_API_KEY_FILE}"
    done
  done
  for DRAW in $(seq 0 $((EXPERT_DRAWS-1))); do
    for PAIR in "oss20:openai/gpt-oss-20b" "oss120:openai/gpt-oss-120b"; do
      ROUTE="${PAIR%%:*}"; MODEL="${PAIR##*:}"
      COMMAND="${COMMAND} && echo '=== ${ROUTE} draw ${DRAW} ===' && ${COLLECT} --route-label ${ROUTE} --model '${MODEL}' --temperature ${TEMP_PRIMARY} --output-suffix _d${DRAW} --api-key-file ${OPENROUTER_API_KEY_FILE}"
    done
  done
  # Retry pass: resume reuses complete rows, retries timeouts
  for DRAW in $(seq 0 $((EXPERT_DRAWS-1))); do
    for PAIR in "oss20:openai/gpt-oss-20b" "oss120:openai/gpt-oss-120b"; do
      ROUTE="${PAIR%%:*}"; MODEL="${PAIR##*:}"
      COMMAND="${COMMAND} && echo '=== ${ROUTE} draw ${DRAW} ===' && ${COLLECT} --route-label ${ROUTE} --model '${MODEL}' --temperature ${TEMP_PRIMARY} --output-suffix _d${DRAW} --api-key-file ${OPENROUTER_API_KEY_FILE}"
    done
  done
  make -C "${REPO_ROOT}" job JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
    SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=32 COMMAND="${COMMAND}"
else
  echo "Unknown MODE=${MODE}" >&2; exit 1
fi

echo "Job: ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
