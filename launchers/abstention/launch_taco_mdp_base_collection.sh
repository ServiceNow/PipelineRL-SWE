#!/usr/bin/env bash
# TACO base collection: stage 1 of the full sequential MDP on external data.
#
# WHY. Savings scale with training-set size, measured two independent ways:
#   half vs full   275 problems -> +11.4% savings | 551 -> +17.1%  (+5.7pt, 6/6 accuracy levels)
#   rolling folds  177 -> ~null | 357 -> mixed | 536 -> clearly positive | 551 -> +17%
# LiveCodeBench is exhausted at 1055 problems, so more data has to come from elsewhere.
# TACO carries 26,443 competitive-programming problems with hidden tests.
#
# NOT a variance fix. The half-data experiment showed seed CV does NOT shrink with data
# (11.8% at 551 vs 7.7% at 275), so this buys mean savings, not stability. Do not sell it
# as the latter.
#
# CONTAMINATION. TACO predates our models, so it is unusable for EVALUATION and this
# collection is training data only -- the router is still evaluated on untouched LCB.
# Contamination breaks router training only if it distorts the relative difficulty
# ORDERING across routes rather than shifting all routes together, which is the
# hypothesis this data tests.
#
# DEPTH 6, NOT 10. oss120 pass@k saturates completely at k=6 on the test split
# (57.89% at k=1 -> 71.93% at k=6, then 0.00pt for draws 7, 8, 9 and 10), and the union
# ceiling is flat from k=6. Collecting 10 draws would spend ~40% more for a measured
# zero. Stage 2 below therefore uses 6.
#
# STAGE 1 (this script): build the TACO problem file and collect one scout + one oss120
# draw, which establishes the problem set, splits and grading that stage 2 extends.
# STAGE 2 (after this finishes -- a real dependency, not a convenience chain):
#
#   LCB_COLLECTION_DIR=<this OUTPUT_DIR> SCOUT_DRAWS=6 EXPERT_DRAWS=6 SCOUT_T06=0 \
#     MODE=scout   SUBMIT=1 bash launchers/abstention/launch_lcb_multidraw_mdp_collect.sh
#   LCB_COLLECTION_DIR=<this OUTPUT_DIR> SCOUT_DRAWS=6 EXPERT_DRAWS=6 SCOUT_T06=0 \
#     MODE=experts SUBMIT=1 bash launchers/abstention/launch_lcb_multidraw_mdp_collect.sh
#
# The two stage-2 jobs are independent of each other and must be launched in parallel.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SCOUT_MODEL=${SCOUT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
ORACLE_MODEL=${ORACLE_MODEL:-openai/gpt-oss-120b}
MAX_SAMPLES=${MAX_SAMPLES:-2000}
TACO_SPLIT=${TACO_SPLIT:-train}
MIN_TESTS=${MIN_TESTS:-2}
# TACO is training data, so nearly everything should land on the train side. The
# collector requires both sides non-empty, and TACO rows mostly carry no date (the
# builder writes 1970-01-01), so this cutoff keeps a small dated tail as the eval side.
TEMPORAL_CUTOFF=${TEMPORAL_CUTOFF:-2018-01-01}
TEMPERATURE=${TEMPERATURE:-0.2}
CONCURRENCY=${CONCURRENCY:-16}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

JOB_NAME=${JOB_NAME:-taco_mdp_base_${MAX_SAMPLES}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}
PROBLEMS_FILE=${PROBLEMS_FILE:-${OUTPUT_DIR}/taco_problems.jsonl}

if [[ ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
  echo "Missing OpenRouter key at ${OPENROUTER_API_KEY_FILE}" >&2
  exit 1
fi

# Cost note: scout runs on a local vLLM and is free; oss120 is $0.029725 per expected
# call. Stage 1 is one oss120 draw per problem; stage 2 experts add 6 oss20 + 6 oss120.
STAGE1_USD=$(awk "BEGIN{printf \"%.0f\", ${MAX_SAMPLES}*0.029725}")
STAGE2_USD=$(awk "BEGIN{printf \"%.0f\", ${MAX_SAMPLES}*(6*0.029725+6*0.003963)}")

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
  problems: TACO ${TACO_SPLIT} split, ${MAX_SAMPLES} sampled, min ${MIN_TESTS} tests
  built to: ${PROBLEMS_FILE}
  scout: ${SCOUT_MODEL} (local vLLM, free)   oracle: ${ORACLE_MODEL} (OpenRouter)
  output: ${OUTPUT_DIR}

  estimated spend  stage 1 (this job): ~\$${STAGE1_USD}
                   stage 2 (6 draws):  ~\$${STAGE2_USD}
  depth capped at 6: oss120 draws 7-10 are measured at 0.00 accuracy points.

Submit explicitly with:
  SUBMIT=1 bash ${BASH_SOURCE[0]}
EOF
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
RUNNER="${OUTPUT_DIR}/run_taco_base.sh"
cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh

python pipelinerl/swe/scripts/livecodebench/build_taco_problems.py \\
  --output '${PROBLEMS_FILE}' \\
  --split '${TACO_SPLIT}' \\
  --max-problems ${MAX_SAMPLES} \\
  --min-tests ${MIN_TESTS} \\
  2>&1 | tee '${OUTPUT_DIR}/build_problems.log'

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
echo
echo "When this finishes, launch stage 2 as TWO PARALLEL jobs:"
echo "  LCB_COLLECTION_DIR=${OUTPUT_DIR} SCOUT_DRAWS=6 EXPERT_DRAWS=6 SCOUT_T06=0 MODE=scout   SUBMIT=1 bash launchers/abstention/launch_lcb_multidraw_mdp_collect.sh"
echo "  LCB_COLLECTION_DIR=${OUTPUT_DIR} SCOUT_DRAWS=6 EXPERT_DRAWS=6 SCOUT_T06=0 MODE=experts SUBMIT=1 bash launchers/abstention/launch_lcb_multidraw_mdp_collect.sh"
