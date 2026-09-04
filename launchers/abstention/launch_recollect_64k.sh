#!/usr/bin/env bash
# Re-collect at a 65536-token cap. Experts via OpenRouter (fast, and the key now has headroom);
# the scout stays local because Qwen3-4B is not served on OpenRouter -- and it is the model whose
# activations are the probe, so we want it under our own control regardless.
#
# Purpose is LABEL QUALITY, not cost estimation. Raising the cap is measured to make cost
# prediction *worse* (LCB 4096 -> 32768 lowered dollar-space R2 on every route), because a low
# cap hides a stochastic tail. What a bigger cap buys is fewer truncated outcomes, i.e. solve
# labels that reflect the model rather than the budget.
#
# KNOWN RISK: OpenRouter produced 22.6% EmptyGeneration on gpt-oss-20b at the 32k cap -- reasoning
# returned with an empty answer channel. collect_lcb_trajectories now retries an empty answer
# unless finish_reason == "length", but that fix has never run against a live key. Check the
# EmptyGeneration rate on the first completed draw before trusting the rest.
#
# CONFOUND to carry: the current primary data is local-32k, so cap and serving path move together
# here. This collection replaces that data rather than being comparable to it.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
R=/mnt/llmd/results/exps/aristides/reason

DATASET=${DATASET:?Need DATASET=lcb|taco}
KEYFILE=${KEYFILE:-/home/toolkit/.secrets/openrouter_api_key}
MAX_TOKENS=${MAX_TOKENS:-65536}
DRAWS=${DRAWS:-6}
TEMPERATURE=${TEMPERATURE:-0.2}
CONCURRENCY=${CONCURRENCY:-8}
GEN_TIMEOUT=${GEN_TIMEOUT:-3600}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}
BASE=${BASE:-$R/recollect64k_${DATASET}_$(date +%s)}

if [[ "${DATASET}" == "lcb" ]]; then
  SRC=$R/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448
  EXTRA=""
else
  SRC=$R/taco_mdp_base_2000_1788249997
  EXTRA="--problems-file $R/taco_mdp_base_2000_1788249997/taco_problems.jsonl \
 --problem-ids-file $R/taco_medhard_ids.txt --temporal-cutoff 2018-01-01"
fi
mkdir -p "${BASE}"

for DRAW in $(seq 0 $((DRAWS-1))); do
  RUNNER="${BASE}/run_d${DRAW}.sh"
  {
    echo '#!/usr/bin/env bash'; echo 'set -euo pipefail'
    echo "cd ${REPO_ROOT}"
    echo 'export HF_HUB_DISABLE_IMPLICIT_TOKEN=1'
    echo 'source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh'
    echo "export OPENROUTER_API_KEY=\$(cat ${KEYFILE})"
    for PAIR in "oss20:openai/gpt-oss-20b" "oss120:openai/gpt-oss-120b"; do
      echo "echo '=== ${PAIR%%:*} draw ${DRAW} ==='"
      echo "python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py \\"
      echo "  --source-collection-dir ${SRC} --output-dir ${BASE} \\"
      echo "  --route-label ${PAIR%%:*} --model '${PAIR##*:}' --api-key-file ${KEYFILE} \\"
      echo "  --splits train,eval --temperature ${TEMPERATURE} --max-tokens ${MAX_TOKENS} \\"
      echo "  --concurrency ${CONCURRENCY} --gen-timeout ${GEN_TIMEOUT} \\"
      echo "  --max-invalid-frac 0.10 --output-suffix _d${DRAW} ${EXTRA}"
    done
  } > "${RUNNER}"
  chmod +x "${RUNNER}"
  if [[ "${SUBMIT}" == "1" ]]; then
    U=$(date +%s%N | tail -c 8)
    if out=$(make -C "${REPO_ROOT}" job JOB_NAME="rc64_${DATASET}_d${DRAW}_${U}" ENV=pipeline-rl \
        CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=0 GPU_MEM=0 \
        CPU=8 CPU_MEM=32 COMMAND="bash ${RUNNER}" 2>&1); then
      echo "ok   rc64_${DATASET}_d${DRAW}_${U}"
    else echo "FAIL rc64_${DATASET}_d${DRAW}: $(echo "$out"|tail -1)"; fi
    sleep 20
  fi
done
echo "Output: ${BASE}"
[[ "${SUBMIT}" == "1" ]] || echo "Prepared, not submitted. SUBMIT=1 to launch."
