#!/usr/bin/env bash
# Collect the BigCodeBench pool -- the second dataset, and the one where the decisions are live.
#
# Why BCB matters beyond "generalisation": gpt-oss-20b-medium scores ~28% here against 73-86% on
# LiveCodeBench. LCB's mean per-draw success is ~0.80, so 44% of its problems are solved on EVERY
# draw and there is nothing to decide on half the pool. BCB puts q in the band where both the
# stop decision and the escalation decision actually bite.
#
# Only tasks whose own reference solution passed 3/3 in THIS environment are collected
# (bcb_keep.json); the rest would measure our sandbox. Draws mirror the LCB pool so the two are
# comparable, and BCB is ~0.3x the cost per draw (254-token prompts, ~944-token answers), so the
# whole pool is ~$9.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
R=/mnt/llmd/results/exps/aristides/reason
SUBMIT=${SUBMIT:-0}
SNAPSHOT=${SNAPSHOT:-0}
KEYFILE=${KEYFILE:-/home/toolkit/.secrets/openrouter_api_key}
KEEP=${KEEP:-$R/bcb_validation/bcb_keep.json}
# Tasks are staged on /mnt/llmd, not loaded from the Hub: compute nodes cannot reach it, so
# load_dataset dies there with DatasetNotFoundError while working fine on the login node. That
# failure mode cost 28 jobs.
TASKS=${TASKS:-$R/bcb_tasks_v014.jsonl}
BASE=${BASE:-$R/pool_v2_bcb}
CONCURRENCY=${CONCURRENCY:-8}
ONLY=${ONLY:-}
START_DRAW=${START_DRAW:-0}

# label | model | temp | top_p | effort | draws | max_tokens
ROUTES=(
  "oss20lo|openai/gpt-oss-20b|1.0|1.0|low|16|117964"
  "oss20md|openai/gpt-oss-20b|1.0|1.0|medium|12|117964"
  "dsv4f|deepseek/deepseek-v4-flash|0.7|0.95|on|8|128000"
  "oss120md|openai/gpt-oss-120b|1.0|1.0|medium|6|115264"
  "oss120hi|openai/gpt-oss-120b|1.0|1.0|high|3|115264"
)
mkdir -p "${BASE}"
for spec in "${ROUTES[@]}"; do
  IFS='|' read -r LABEL MODEL TEMP TOPP EFFORT NDRAW MAXTOK <<< "${spec}"
  if [[ -n "${ONLY}" && " ${ONLY} " != *" ${LABEL} "* ]]; then continue; fi
  case "${EFFORT}" in
    "")  REASON_ARG="" ;;
    on)  REASON_ARG=" --reasoning-enabled --require-parameters" ;;
    *)   REASON_ARG=" --reasoning-effort ${EFFORT}" ;;
  esac
  for DRAW in $(seq ${START_DRAW} $((START_DRAW + NDRAW - 1))); do
    RUNNER="${BASE}/run_${LABEL}_d${DRAW}.sh"
    {
      echo '#!/usr/bin/env bash'; echo 'set -euo pipefail'
      echo "cd ${REPO_ROOT}"
      echo "export OPENROUTER_API_KEY=\$(cat ${KEYFILE})"
      echo "python pipelinerl/swe/scripts/bigcodebench/collect_bcb_expert.py \\"
      echo "  --keep-file ${KEEP} --tasks-file ${TASKS} --output-dir ${BASE} --route-label ${LABEL} \\"
      echo "  --model '${MODEL}' --splits train,eval --output-suffix _d${DRAW} \\"
      echo "  --temperature ${TEMP} --top-p ${TOPP} --max-tokens ${MAXTOK} \\"
      echo "  --api-key-file ${KEYFILE} --concurrency ${CONCURRENCY}${REASON_ARG}"
    } > "${RUNNER}"
    chmod +x "${RUNNER}"
    if [[ "${SUBMIT}" == "1" ]]; then
      if out=$(make -C "${REPO_ROOT}" job JOB_NAME="bcb_${LABEL}_d${DRAW}_$(date +%s)" ENV=pipeline-rl \
          CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" NPROC=1 \
          GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=32 COMMAND="bash ${RUNNER}" 2>&1); then
        echo "  ok   bcb_${LABEL}_d${DRAW}"
      else echo "  FAIL bcb_${LABEL}_d${DRAW}: $(echo "$out" | tail -1)"; fi
      sleep 15
    fi
  done
done
echo "Output: ${BASE}"
[[ "${SUBMIT}" == "1" ]] || echo "Prepared, not submitted. SUBMIT=1 to launch."
