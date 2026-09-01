#!/usr/bin/env bash
# Plan-then-execute on LiveCodeBench: does "large model plans, small model implements"
# create a new rung on the cost ladder, or a new dimension?
#
# Stage 1 (this launcher) is inference-only and precedes any RL. Two jobs:
#
#   MODE=plan     CPU job. The planner (default gpt-oss-120b via OpenRouter) writes a short
#                 natural-language plan per problem, no code. Records real prompt and
#                 completion tokens including hidden reasoning, since that is the bill.
#
#   MODE=execute  GPU job. Local vLLM serves the scout, which implements each plan with
#                 EXEC_DRAWS draws at T=0.2, graded on public and full suites. Also runs two
#                 controls in the same job:
#                   * self-plan: the scout writes its own plans, then executes them (the
#                     COPE stage-1 analogue; isolates "planning as a format" from "the
#                     large model's plan content");
#                   * null-plan: a content-free scaffold (the 2604.01029 null-draft control).
#                 Finishes by running analyze_plan_execute.py against the existing tensor.
#
# Why LCB eval (341 problems) first: it is the split where oss120 and scout draws already
# exist in tensors_v3, so c_inf, hazard and cost per solved problem come out of one job.
# The 551-problem train split is only needed once the composite justifies RL training.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

MODE=${MODE:?Set MODE=plan or MODE=execute}
: "${LCB_COLLECTION_DIR:=/mnt/llmd/results/exps/aristides/reason/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448}"
: "${TENSORS_DIR:=/mnt/llmd/results/exps/aristides/reason/lcb_mdp_temporal_551_341_prepared_v1/tensors_v3}"
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SPLITS=${SPLITS:-eval}
MAX_PROBLEMS=${MAX_PROBLEMS:-0}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
GEN_TIMEOUT=${GEN_TIMEOUT:-600}
MAX_INVALID_FRAC=${MAX_INVALID_FRAC:-0.05}

# planner
PLANNER_LABEL=${PLANNER_LABEL:-plan120}
PLANNER_MODEL=${PLANNER_MODEL:-openai/gpt-oss-120b}
PLAN_MAX_TOKENS=${PLAN_MAX_TOKENS:-8192}   # visible plan is capped by the prompt (<300 words); this bounds hidden reasoning
PLAN_TEMPERATURE=${PLAN_TEMPERATURE:-0.2}
PLAN_CONCURRENCY=${PLAN_CONCURRENCY:-4}

# executor
SCOUT_MODEL=${SCOUT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
EXEC_DRAWS=${EXEC_DRAWS:-4}
EXEC_TEMPERATURE=${EXEC_TEMPERATURE:-0.2}
EXEC_MAX_TOKENS=${EXEC_MAX_TOKENS:-4096}
EXEC_CONCURRENCY=${EXEC_CONCURRENCY:-8}
VLLM_PORT=${VLLM_PORT:-8000}
SELF_PLAN=${SELF_PLAN:-1}
NULL_PLAN=${NULL_PLAN:-1}
: "${PLAN_DIR:=}"   # execute: directory holding ${PLANNER_LABEL}_{split}.jsonl from the plan job

JOB_NAME=${JOB_NAME:-lcb_plan_execute_${MODE}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

COLLECT="python pipelinerl/swe/scripts/livecodebench/collect_lcb_plan_execute.py \
  --source-collection-dir ${LCB_COLLECTION_DIR} --output-dir ${OUTPUT_DIR} \
  --splits ${SPLITS} --eval-timeout ${EVAL_TIMEOUT} --gen-timeout ${GEN_TIMEOUT} \
  --max-invalid-frac ${MAX_INVALID_FRAC} --max-problems ${MAX_PROBLEMS}"

if [[ "${MODE}" == "plan" ]]; then
  [[ -s "${OPENROUTER_API_KEY_FILE}" ]] || { echo "Missing OpenRouter key at ${OPENROUTER_API_KEY_FILE}" >&2; exit 1; }
  RUNNER_BODY="source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh && export OPENROUTER_API_KEY=\$(cat ${OPENROUTER_API_KEY_FILE})"
  # Two passes: the second reuses complete rows and retries API failures.
  for PASS in 1 2; do
    RUNNER_BODY="${RUNNER_BODY} && echo '=== plan ${PLANNER_LABEL} pass ${PASS} ===' && ${COLLECT} --phase plan \
      --route-label ${PLANNER_LABEL} --model '${PLANNER_MODEL}' --api-key-file ${OPENROUTER_API_KEY_FILE} \
      --max-tokens ${PLAN_MAX_TOKENS} --temperature ${PLAN_TEMPERATURE} --concurrency ${PLAN_CONCURRENCY}"
  done
  RESOURCES="GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=32"
  SUMMARY="planner ${PLANNER_MODEL} -> ${OUTPUT_DIR}/${PLANNER_LABEL}_<split>.jsonl"

elif [[ "${MODE}" == "execute" ]]; then
  : "${PLAN_DIR:?Set PLAN_DIR to the output dir of the plan job}"
  for S in ${SPLITS//,/ }; do
    [[ -s "${PLAN_DIR}/${PLANNER_LABEL}_${S}.jsonl" ]] || { echo "Missing ${PLAN_DIR}/${PLANNER_LABEL}_${S}.jsonl" >&2; exit 1; }
  done
  mkdir -p "${OUTPUT_DIR}"
  LOCAL_KEY="${OUTPUT_DIR}/local_key.txt"
  LOCAL="--api-key-file ${LOCAL_KEY} --base-url http://localhost:${VLLM_PORT} --model '${SCOUT_MODEL}'"
  EXEC_ARGS="--max-tokens ${EXEC_MAX_TOKENS} --temperature ${EXEC_TEMPERATURE} --concurrency ${EXEC_CONCURRENCY}"
  RUNNER_BODY="echo local > ${LOCAL_KEY} && export OPENROUTER_API_KEY=local && export HF_HUB_DISABLE_IMPLICIT_TOKEN=1"
  RUNNER_BODY="${RUNNER_BODY} && source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh"
  RUNNER_BODY="${RUNNER_BODY} && python -m vllm.entrypoints.openai.api_server --model ${SCOUT_MODEL} --port ${VLLM_PORT} \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 --max-model-len 32768 --trust-remote-code \
    --served-model-name ${SCOUT_MODEL} > ${OUTPUT_DIR}/vllm_server.log 2>&1 & VLLM_PID=\$!"
  RUNNER_BODY="${RUNNER_BODY} && for i in \$(seq 1 120); do curl -sf http://localhost:${VLLM_PORT}/health >/dev/null 2>&1 && break; sleep 5; done && echo '[vllm] ready'"

  # main arm: scout executes the planner's plans
  for DRAW in $(seq 0 $((EXEC_DRAWS-1))); do
    RUNNER_BODY="${RUNNER_BODY} && echo '=== ${PLANNER_LABEL}_scout draw ${DRAW} ===' && ${COLLECT} --phase execute \
      --route-label ${PLANNER_LABEL}_scout --plan-file '${PLAN_DIR}/${PLANNER_LABEL}_{split}.jsonl' ${LOCAL} ${EXEC_ARGS} --output-suffix _d${DRAW}"
  done
  # control 1: scout plans for itself, then executes its own plans
  if [[ "${SELF_PLAN}" == "1" ]]; then
    RUNNER_BODY="${RUNNER_BODY} && echo '=== self-plan: scout writes plans ===' && ${COLLECT} --phase plan \
      --route-label planscout ${LOCAL} --max-tokens 2048 --temperature ${PLAN_TEMPERATURE} --concurrency ${EXEC_CONCURRENCY}"
    for DRAW in $(seq 0 $((EXEC_DRAWS-1))); do
      RUNNER_BODY="${RUNNER_BODY} && echo '=== planscout_scout draw ${DRAW} ===' && ${COLLECT} --phase execute \
        --route-label planscout_scout --plan-file '${OUTPUT_DIR}/planscout_{split}.jsonl' ${LOCAL} ${EXEC_ARGS} --output-suffix _d${DRAW}"
    done
  fi
  # control 2: content-free scaffold
  if [[ "${NULL_PLAN}" == "1" ]]; then
    for DRAW in $(seq 0 $((EXEC_DRAWS-1))); do
      RUNNER_BODY="${RUNNER_BODY} && echo '=== nullplan_scout draw ${DRAW} ===' && ${COLLECT} --phase execute \
        --route-label nullplan_scout ${LOCAL} ${EXEC_ARGS} --output-suffix _d${DRAW}"
    done
  fi
  RUNNER_BODY="${RUNNER_BODY} && kill \${VLLM_PID} 2>/dev/null || true"
  # analysis against the existing tensor (oss120 + scout draws), one report per arm
  for ARM in ${PLANNER_LABEL}_scout $([[ "${SELF_PLAN}" == "1" ]] && echo planscout_scout) $([[ "${NULL_PLAN}" == "1" ]] && echo nullplan_scout); do
    for S in ${SPLITS//,/ }; do
      RUNNER_BODY="${RUNNER_BODY} && python pipelinerl/swe/scripts/livecodebench/analyze_plan_execute.py \
        --tensors-dir ${TENSORS_DIR} --exec-dir ${OUTPUT_DIR} --route-label ${ARM} --split ${S} | tee -a ${OUTPUT_DIR}/analysis_report.txt"
    done
  done
  RESOURCES="GPU=1 GPU_MEM=24 CPU=8 CPU_MEM=32"
  SUMMARY="scout ${SCOUT_MODEL} executes ${PLAN_DIR}/${PLANNER_LABEL}_<split>.jsonl, ${EXEC_DRAWS} draws; self-plan=${SELF_PLAN} null-plan=${NULL_PLAN}"
else
  echo "Unknown MODE=${MODE}" >&2; exit 1
fi

mkdir -p "${OUTPUT_DIR}"
RUNNER="${OUTPUT_DIR}/run_${MODE}.sh"
printf '#!/usr/bin/env bash\nset -euo pipefail\n%s\necho "[done]"\n' "${RUNNER_BODY}" > "${RUNNER}"
chmod +x "${RUNNER}"

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job:      ${JOB_NAME}
  mode:     ${MODE}   splits: ${SPLITS}   max_problems: ${MAX_PROBLEMS:-all}
  ${SUMMARY}
  runner:   ${RUNNER}
  output:   ${OUTPUT_DIR}

Submit with:
  SUBMIT=1 MODE=${MODE}$([[ -n "${PLAN_DIR}" ]] && echo " PLAN_DIR=${PLAN_DIR}") bash ${BASH_SOURCE[0]}
EOF
  exit 0
fi

# shellcheck disable=SC2086
make -C "${REPO_ROOT}" job JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC=1 ${RESOURCES} COMMAND="bash ${RUNNER}"

echo "Job:        ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
