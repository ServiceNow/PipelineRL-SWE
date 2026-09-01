#!/usr/bin/env bash
# Peer-pool screen: 50 problems x 4 candidate models, BEFORE committing to a full collection.
#
# WHY THIS EXISTS. `qwen/qwen3.5-plus-20260420` returned **336/341 empty outputs** in the
# gate-1 collection: a reasoning model whose entire 4096-token budget went to hidden
# reasoning, emitting nothing visible. We paid for 341 generations and got a 1.6% pass rate
# that meant nothing. Three of the four models below are reasoning models. This screen
# costs ~$1 and makes that failure impossible to repeat unnoticed.
#
# THE POOL, chosen on CodeRouterBench (9,999 code tasks x 8 models, real per-task costs)
# by maximising routable excess -- specialisation over a Rasch null -- subject to being a
# genuine peer band spanning distinct pretraining lineages:
#
#   qwen/qwen3-max          Alibaba   36.38%  plain
#   z-ai/glm-5              Zhipu     36.14%  reasoning
#   moonshotai/kimi-k2.5    Moonshot  34.43%  reasoning
#   minimax/minimax-m2.7    MiniMax   34.61%  reasoning
#
#   routable excess +4.98pt (z=24.9) | accuracy spread 1.95pt | disagreement rate 40.3%
#
# WHAT THE SCREEN DECIDES, per model:
#   1. does it emit extractable code at all (the q35p failure)?
#   2. what does a call ACTUALLY cost on competitive-programming problems? The $/call
#      estimates assume 1800 completion tokens; reasoning models will far exceed that, and
#      the full-collection budget depends on the real number.
#   3. is its solve rate in the same band as the others -- i.e. is this really a peer pool
#      on OUR problems, not just on CodeRouterBench's easier HumanEval-style tasks?
#
# A model that fails (1), or lands far outside the band on (3), is swapped before we spend
# on ~2,900 problems.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${LCB_COLLECTION_DIR:?Set LCB_COLLECTION_DIR to an existing scout collection (supplies the problem ids)}"
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
MAX_PROBLEMS=${MAX_PROBLEMS:-50}
SPLITS=${SPLITS:-eval}
TEMPERATURE=${TEMPERATURE:-0.2}
# 32768, not the 4096 default: that default is exactly what silently broke q35p.
MAX_TOKENS=${MAX_TOKENS:-32768}
CONCURRENCY=${CONCURRENCY:-4}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
GEN_TIMEOUT=${GEN_TIMEOUT:-900}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}

# route_label:openrouter_model
PEER_PAIRS=${PEER_PAIRS:-"qmax:qwen/qwen3-max glm5:z-ai/glm-5 kimi:moonshotai/kimi-k2.5 minimax:minimax/minimax-m2.7"}

JOB_NAME=${JOB_NAME:-peer_pool_screen_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

if [[ ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
  echo "Missing OpenRouter key at ${OPENROUTER_API_KEY_FILE}" >&2
  exit 1
fi

N_MODELS=$(echo "${PEER_PAIRS}" | wc -w)
EST=$(awk "BEGIN{printf \"%.2f\", ${N_MODELS}*${MAX_PROBLEMS}*0.006}")

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
  problems: first ${MAX_PROBLEMS} of the '${SPLITS}' split, by sorted id (same set for every model)
  models (${N_MODELS}):
$(for p in ${PEER_PAIRS}; do echo "    ${p%%:*}  ->  ${p#*:}"; done)
  max_tokens: ${MAX_TOKENS}   temperature: ${TEMPERATURE}
  output: ${OUTPUT_DIR}
  rough ceiling on spend: ~\$${EST}

  Reads out: empty-output rate, code-extraction rate, REAL tokens and cost per call,
  reasoning-token share, and solve rate per model.

Submit explicitly with:
  SUBMIT=1 LCB_COLLECTION_DIR=${LCB_COLLECTION_DIR} bash ${BASH_SOURCE[0]}
EOF
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
RUNNER="${OUTPUT_DIR}/run_screen.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo 'export HF_HUB_DISABLE_IMPLICIT_TOKEN=1'
  echo 'source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh'
  for pair in ${PEER_PAIRS}; do
    label="${pair%%:*}"; model="${pair#*:}"
    cat <<EOF
echo "=== screening ${label} (${model}) ==="
python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py \\
  --source-collection-dir '${LCB_COLLECTION_DIR}' \\
  --output-dir '${OUTPUT_DIR}' \\
  --route-label '${label}' \\
  --model '${model}' \\
  --api-key-file '${OPENROUTER_API_KEY_FILE}' \\
  --splits '${SPLITS}' \\
  --max-problems ${MAX_PROBLEMS} \\
  --max-tokens ${MAX_TOKENS} \\
  --temperature ${TEMPERATURE} \\
  --concurrency ${CONCURRENCY} \\
  --eval-timeout ${EVAL_TIMEOUT} \\
  --gen-timeout ${GEN_TIMEOUT} \\
  --max-invalid-frac 1.0 \\
  --output-suffix _screen || echo "!! ${label} FAILED -- that is itself a screen result"
EOF
  done
  echo "python pipelinerl/swe/scripts/livecodebench/analyze_peer_screen.py --screen-dir '${OUTPUT_DIR}' | tee '${OUTPUT_DIR}/screen_report.txt'"
} > "${RUNNER}"
chmod +x "${RUNNER}"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=0 CPU=8 CPU_MEM=32 \
  COMMAND="bash ${RUNNER}"

echo "Job:        ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Report:     ${OUTPUT_DIR}/screen_report.txt"
