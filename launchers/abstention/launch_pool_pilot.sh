#!/usr/bin/env bash
# POOL PILOT: which rungs should the recollected pool have?
#
# On the 3-route pool the fixed cascade scout -> 20B -> 120B lands on our frontier on TACO: with a
# verifier, "try each tier once, cheapest first" is already a crude adaptive policy, and with only
# three rungs it is close to right. A pool with more, geometrically priced rungs should make it
# wasteful. This pilot measures candidate rungs on 100 LCB problems (50 per split), 2 draws each,
# at each provider's RECOMMENDED sampling settings (the main pool used T=0.2 everywhere), through
# ONE prompt / extraction / grading path: the local 4B models are served by vLLM inside their job
# and collected with the same collect_lcb_expert.py as the OpenRouter routes.
#
# Keep a rung only if it buys >= ~5 points of pass@1 at >= 2x the price of the rung below; check
# the top/bottom price ratio is >= 50x; and compare "try each rung once" with a perfect-belief
# policy to see how much margin the pool leaves to win.
#
# One eai job per (route, draw), submitted in parallel.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
R=/mnt/llmd/results/exps/aristides/reason
SUBMIT=${SUBMIT:-0}
SNAPSHOT=${SNAPSHOT:-1}
DRAWS=${DRAWS:-2}
PER_SPLIT=${PER_SPLIT:-50}
MAX_TOKENS=${MAX_TOKENS:-65536}
KEYFILE=${KEYFILE:-/home/toolkit/.secrets/openrouter_api_key}
SRC=${SRC:-$R/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448}
BASE=${BASE:-$R/pool_pilot_lcb_$(date +%s)}
ONLY=${ONLY:-}          # optional space-separated route labels to (re)submit

# label | model | where | temperature | top_p | reasoning effort
ROUTES=(
  "scout4i|Qwen/Qwen3-4B-Instruct-2507|local|0.7|0.8|"
  "scout4t|Qwen/Qwen3-4B-Thinking-2507|local|0.6|0.95|"
  "oss20lo|openai/gpt-oss-20b|or|1.0|1.0|low"
  "oss20md|openai/gpt-oss-20b|or|1.0|1.0|medium"
  "oss20hi|openai/gpt-oss-20b|or|1.0|1.0|high"
  "q30t|qwen/qwen3-30b-a3b-thinking-2507|or|0.6|0.95|"
  "oss120lo|openai/gpt-oss-120b|or|1.0|1.0|low"
  "oss120md|openai/gpt-oss-120b|or|1.0|1.0|medium"
  "oss120hi|openai/gpt-oss-120b|or|1.0|1.0|high"
  "q235t|qwen/qwen3-235b-a22b-thinking-2507|or|0.6|0.95|"
)
mkdir -p "${BASE}"
echo "EMPTY" > "${BASE}/local_key"          # vLLM ignores it; never send the real key to localhost

for spec in "${ROUTES[@]}"; do
  IFS='|' read -r LABEL MODEL WHERE TEMP TOPP EFFORT <<< "${spec}"
  if [[ -n "${ONLY}" && " ${ONLY} " != *" ${LABEL} "* ]]; then continue; fi
  for DRAW in $(seq 0 $((DRAWS-1))); do
    RUNNER="${BASE}/run_${LABEL}_d${DRAW}.sh"
    COMMON="--source-collection-dir ${SRC} --output-dir ${BASE} --route-label ${LABEL} \
 --model '${MODEL}' --splits train,eval --max-problems ${PER_SPLIT} \
 --temperature ${TEMP} --top-p ${TOPP} --max-tokens ${MAX_TOKENS} --gen-timeout 3600 \
 --max-invalid-frac 0.10 --output-suffix _d${DRAW}${EFFORT:+ --reasoning-effort ${EFFORT}}"
    {
      echo '#!/usr/bin/env bash'; echo 'set -euo pipefail'
      echo "cd ${REPO_ROOT}"
      echo 'export HF_HUB_DISABLE_IMPLICIT_TOKEN=1'
      echo 'source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh'
      if [[ "${WHERE}" == "local" ]]; then
        echo "python -m vllm.entrypoints.openai.api_server --model ${MODEL} --port 8000 \\"
        echo "  --gpu-memory-utilization 0.90 --max-model-len 72000 --served-model-name ${MODEL} \\"
        echo "  > ${BASE}/vllm_${LABEL}_d${DRAW}.log 2>&1 &"
        echo 'for i in $(seq 1 180); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && break; sleep 5; done'
        echo "python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py ${COMMON} \\"
        echo "  --base-url http://localhost:8000 --api-key-file ${BASE}/local_key --concurrency 16"
      else
        echo "export OPENROUTER_API_KEY=\$(cat ${KEYFILE})"
        echo "python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py ${COMMON} \\"
        echo "  --api-key-file ${KEYFILE} --concurrency 8"
      fi
    } > "${RUNNER}"
    chmod +x "${RUNNER}"
    if [[ "${SUBMIT}" == "1" ]]; then
      if [[ "${WHERE}" == "local" ]]; then RES="GPU=1 GPU_MEM=80 CPU=8 CPU_MEM=64"; else RES="GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=32"; fi
      if out=$(make -C "${REPO_ROOT}" job JOB_NAME="pilot_${LABEL}_d${DRAW}_$(date +%s)" ENV=pipeline-rl \
          CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" NPROC=1 ${RES} \
          COMMAND="bash ${RUNNER}" 2>&1); then
        echo "ok   pilot_${LABEL}_d${DRAW}"
      else echo "FAIL pilot_${LABEL}_d${DRAW}: $(echo "$out" | tail -1)"; fi
      sleep 25
    fi
  done
done
echo "Output: ${BASE}"
[[ "${SUBMIT}" == "1" ]] || echo "Prepared, not submitted. SUBMIT=1 to launch."
