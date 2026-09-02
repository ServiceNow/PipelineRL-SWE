#!/usr/bin/env bash
# Re-collect LiveCodeBench at a 32k cap, sharded one job per draw.
#
# The truncation audit settled this: 39 of 124 pool-unsolved problems (31.5%) fall at 32768
# tokens, 73.1% of those draws exceeded the old 4096 cap, and none reached the new one. The
# impossible set moves 13.9% -> 9.5% from correcting gpt-oss-120b alone. Every stopping and
# headroom number computed on the old tensors is therefore measured against corrupted labels.
#
# Sharded by draw because grading is serialized inside a process (the official runner forks
# per call and concurrent forks corrupt labels), so the only way to parallelize is more jobs.
# One job per draw index, both expert routes, both splits.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
R=/mnt/llmd/results/exps/aristides/reason

: "${LCB_COLLECTION_DIR:=$R/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448}"
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
DRAWS=${DRAWS:-6}
MAX_TOKENS=${MAX_TOKENS:-32768}
TEMPERATURE=${TEMPERATURE:-0.2}
CONCURRENCY=${CONCURRENCY:-8}
GEN_TIMEOUT=${GEN_TIMEOUT:-1800}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
SPLITS=${SPLITS:-train,eval}
EXPERT_PAIRS=${EXPERT_PAIRS:-"oss20:openai/gpt-oss-20b oss120:openai/gpt-oss-120b"}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}
BASE=${BASE:-$R/lcb_recollect_32k_${TIMESTAMP}}
mkdir -p "${BASE}"

for DRAW in $(seq 0 $((DRAWS-1))); do
  RUNNER="${BASE}/run_d${DRAW}.sh"
  {
    echo '#!/usr/bin/env bash'; echo 'set -euo pipefail'
    echo 'export HF_HUB_DISABLE_IMPLICIT_TOKEN=1'
    echo 'source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh'
    echo "export OPENROUTER_API_KEY=\$(cat ${OPENROUTER_API_KEY_FILE})"
    for PASS in 1 2; do
      for PAIR in ${EXPERT_PAIRS}; do
        echo "echo '=== ${PAIR%%:*} draw ${DRAW} pass ${PASS} ==='"
        echo "python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py \\"
        echo "  --source-collection-dir ${LCB_COLLECTION_DIR} --output-dir ${BASE} \\"
        echo "  --route-label ${PAIR%%:*} --model '${PAIR##*:}' --api-key-file ${OPENROUTER_API_KEY_FILE} \\"
        echo "  --splits ${SPLITS} --temperature ${TEMPERATURE} --max-tokens ${MAX_TOKENS} \\"
        echo "  --concurrency ${CONCURRENCY} --eval-timeout ${EVAL_TIMEOUT} --gen-timeout ${GEN_TIMEOUT} \\"
        echo "  --max-invalid-frac 0.10 --output-suffix _d${DRAW}"
      done
    done
  } > "${RUNNER}"
  chmod +x "${RUNNER}"
  if [[ "${SUBMIT}" == "1" ]]; then
    make -C "${REPO_ROOT}" job JOB_NAME="lcb_recollect32k_d${DRAW}_${TIMESTAMP}" ENV=pipeline-rl \
      CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=32 \
      COMMAND="bash ${RUNNER}" >/dev/null 2>&1 && echo "submitted draw ${DRAW}"
    sleep 5
  fi
done
[[ "${SUBMIT}" == "1" ]] || echo "Prepared ${DRAWS} per-draw runners under ${BASE}. Submit with SUBMIT=1."
echo "Output dir: ${BASE}"
