#!/usr/bin/env bash
# Collect the whole LiveCodeBench pool locally, retiring OpenRouter.
#
# Built for the probe baseline, but self-hosting turns out to fix four things at once:
#
#   1. The blocker. The OpenRouter key sits at its $200 ceiling and every call 403s.
#   2. The EmptyGeneration artifact. 0.0% on locally served vLLM against 16-43% via
#      OpenRouter -- it tracked the serving path, not the model, and it was recording
#      "the provider returned no answer" as "gpt-oss-20b got it wrong" on the middle rung
#      of the ladder the whole cascade argument rests on.
#   3. A real cost basis. Tokens/sec over GPU-hour is measurable here; OpenRouter list
#      prices are subsidised and the scout has no market price at all.
#   4. Reproducibility. Pinned weights and pinned quantization, versus OpenRouter routing
#      to an unnamed provider whose serving configuration we cannot see or cite.
#
# Everything is recollected rather than topped up: locally served MXFP4 and whatever a
# provider served are not interchangeable, so mixing draws for the same nominal model would
# be unsound. The OpenRouter data is kept only as the serving-path comparison that evidences (2).
#
# Two environments in one job. vLLM 0.15 (gpt-oss capable) lives in vllm-env; the collector
# and the LiveCodeBench grader need pipeline-rl. They are separate processes talking over
# HTTP, so each runs under the interpreter that supports it.
#
# One job per route, all draws behind a single server load. Generation is batched and fast;
# grading is the bottleneck and is deliberately serialised, because the official runner forks
# a worker per call and concurrent forks corrupt labels.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
R=/mnt/llmd/results/exps/aristides/reason

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
BASE=${BASE:-$R/lcb_local_pool_${TIMESTAMP}}
: "${LCB_COLLECTION_DIR:=$R/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448}"
VLLM_PY=${VLLM_PY:-/home/toolkit/.conda/envs/vllm-env/bin/python}
DRAWS=${DRAWS:-6}
TEMPERATURE=${TEMPERATURE:-0.2}
MAX_TOKENS=${MAX_TOKENS:-32768}
# Must exceed MAX_TOKENS plus the prompt, or vLLM 400s every request -- the exact bug that
# made the previous scout re-collection 100% invalid.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
CONCURRENCY=${CONCURRENCY:-16}
GEN_TIMEOUT=${GEN_TIMEOUT:-1800}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
MAX_INVALID_FRAC=${MAX_INVALID_FRAC:-0.05}
SPLITS=${SPLITS:-train,eval}
PORT=${PORT:-8000}
# Set these to run the identical pipeline over TACO instead of LiveCodeBench. PROBLEMS_FILE
# bypasses the LCB download; PROBLEM_IDS_FILE narrows to a subset (e.g. medium+hard only);
# TEMPORAL_CUTOFF must match the cutoff the base collection was split on, or the source-ID
# check fails before any spend.
PROBLEMS_FILE=${PROBLEMS_FILE:-}
PROBLEM_IDS_FILE=${PROBLEM_IDS_FILE:-}
TEMPORAL_CUTOFF=${TEMPORAL_CUTOFF:-}
EXTRA=""
[[ -n "${PROBLEMS_FILE}" ]] && EXTRA="${EXTRA} --problems-file ${PROBLEMS_FILE}"
[[ -n "${PROBLEM_IDS_FILE}" ]] && EXTRA="${EXTRA} --problem-ids-file ${PROBLEM_IDS_FILE}"
[[ -n "${TEMPORAL_CUTOFF}" ]] && EXTRA="${EXTRA} --temporal-cutoff ${TEMPORAL_CUTOFF}"
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}

# label:model:tensor_parallel:gpus  -- gpt-oss is MXFP4 under vLLM, so 120b fits two cards
# comfortably at a 40k context; the scout and 20b take one each. Four GPUs in total.
ROUTES=${ROUTES:-"scout:Qwen/Qwen3-4B-Instruct-2507:1:1 oss20:openai/gpt-oss-20b:1:1 oss120:openai/gpt-oss-120b:2:2"}

mkdir -p "${BASE}"
echo local > "${BASE}/local_key.txt"

for SPEC in ${ROUTES}; do
  LABEL=${SPEC%%:*}; REST=${SPEC#*:}
  MODEL=${REST%%:*}; REST2=${REST#*:}
  TP=${REST2%%:*}; GPUS=${REST2##*:}
  RUNNER="${BASE}/run_${LABEL}.sh"

  COLLECT="python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py \
 --source-collection-dir ${LCB_COLLECTION_DIR} --output-dir ${BASE} \
 --splits ${SPLITS} --concurrency ${CONCURRENCY} --temperature ${TEMPERATURE} \
 --max-tokens ${MAX_TOKENS} --eval-timeout ${EVAL_TIMEOUT} --gen-timeout ${GEN_TIMEOUT} \
 --max-invalid-frac ${MAX_INVALID_FRAC} --api-key-file ${BASE}/local_key.txt \
 --base-url http://localhost:${PORT} --route-label ${LABEL} --model '${MODEL}'${EXTRA}"

  INNER=""
  for DRAW in $(seq 0 $((DRAWS-1))); do
    INNER="${INNER}echo \"=== ${LABEL} draw ${DRAW} \$(date -Is) ===\"
${COLLECT} --output-suffix _d${DRAW}
"
  done

  cat > "${RUNNER}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OPENROUTER_API_KEY=local

# Timestamps bracket the run so tokens/sec -- and therefore a real serving cost -- can be
# recovered afterwards from the output files and the GPU count.
echo "[serving] route=${LABEL} model=${MODEL} tp=${TP} gpus=${GPUS} start=\$(date -Is)"
"${VLLM_PY}" -m vllm.entrypoints.openai.api_server \\
  --model ${MODEL} --port ${PORT} --tensor-parallel-size ${TP} \\
  --gpu-memory-utilization 0.90 --max-model-len ${MAX_MODEL_LEN} --trust-remote-code \\
  --served-model-name ${MODEL} > ${BASE}/vllm_${LABEL}.log 2>&1 &
VLLM_PID=\$!

ready=0
for i in \$(seq 1 240); do
  if curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 \${VLLM_PID} 2>/dev/null; then
    echo "[serving] vLLM died during startup; last 40 lines:"; tail -40 ${BASE}/vllm_${LABEL}.log
    exit 1
  fi
  sleep 5
done
[ "\${ready}" = "1" ] || { echo "[serving] timed out waiting for health"; tail -40 ${BASE}/vllm_${LABEL}.log; exit 1; }
echo "[serving] ready at \$(date -Is)"

${INNER}
echo "[serving] collection finished at \$(date -Is)"
kill \${VLLM_PID} 2>/dev/null || true
echo "[done]"
SCRIPT
  chmod +x "${RUNNER}"

  if [[ "${SUBMIT}" == "1" ]]; then
    make -C "${REPO_ROOT}" job JOB_NAME="lcb_local_${LABEL}_${TIMESTAMP}" ENV=pipeline-rl \
      CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU="${GPUS}" GPU_MEM=80 \
      CPU=16 CPU_MEM=96 COMMAND="bash ${RUNNER}" \
      && echo "submitted ${LABEL} (${MODEL}, tp=${TP}, ${GPUS} GPU, ${DRAWS} draws)" \
      || { echo "FAILED to submit ${LABEL}"; exit 1; }
    sleep 30
  fi
done

echo "Output: ${BASE}"
[[ "${SUBMIT}" == "1" ]] || echo "Prepared, not submitted. Submit with SUBMIT=1."
