#!/usr/bin/env bash
# Re-grade saved LCB generations one test at a time, so failure PATTERNS exist.
#
# The pinned grader returns at the first failing test, so every result_codes array in the
# collection is a prefix and the tests after the first failure were never run. Two analyses
# in this project were computed on that degenerate encoding and both need redoing:
#
#   * "fraction of tests passed" is a first-failure index, not partial credit. Its measured
#     signal content was nil (AUC 0.53 for pool-solvability) -- but that was the index.
#   * cross-model failure agreement could only ask "same stopping index", not "same failed
#     tests". 59.1% of failed draws stop at index 0, collapsing most of the population into
#     one bucket, and the resulting statistics came out at 0.50-0.60 AUC.
#
# Grading each test as its own single-test suite removes the short circuit without patching
# the pinned runner: with one test there is nothing to stop early for, so the semantics are
# preserved exactly and the full pattern falls out.
#
# No generation is re-run. Every program is read from the saved draw files, so this costs
# CPU only and no API spend. Grading is serialized because the official runner forks a
# worker per call and concurrent forks corrupt labels.
#
# Scope here is draw 0 of all three ladder routes over all 892 problems, which is the
# cheapest set that answers both questions: real partial credit for every first attempt, and
# failure-set agreement across models on the subset where all three fail.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
R=/mnt/llmd/results/exps/aristides/reason

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
DRAW=${DRAW:-0}
MAX_TESTS=${MAX_TESTS:-30}       # median suite is 16; caps the tail so the job is bounded
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
ONLY_ALL_FAIL=${ONLY_ALL_FAIL:-0}
LIMIT=${LIMIT:-0}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}

SCOUT_DRAWS=${SCOUT_DRAWS:-${R}/lcb_multidraw_scout_1787547502/scout_eval_d${DRAW}.jsonl}
SCOUT_TRAIN=${SCOUT_TRAIN:-${R}/lcb_multidraw_scout_train_1787866682/scout_train_d${DRAW}.jsonl}
OSS20_EVAL=${OSS20_EVAL:-${R}/lcb_multidraw_experts_1787502237/oss20_eval_d${DRAW}.jsonl}
OSS20_TRAIN=${OSS20_TRAIN:-${R}/lcb_multidraw_experts_train_1787866632/oss20_train_d${DRAW}.jsonl}
OSS120_EVAL=${OSS120_EVAL:-${R}/lcb_multidraw_experts_1787502237/oss120_eval_d${DRAW}.jsonl}
OSS120_TRAIN=${OSS120_TRAIN:-${R}/lcb_multidraw_experts_train_1787866632/oss120_train_d${DRAW}.jsonl}

JOB_NAME=${JOB_NAME:-lcb_full_pattern_regrade_d${DRAW}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-${R}/${JOB_NAME}}
mkdir -p "${OUTPUT_DIR}"

ARGS="--output ${OUTPUT_DIR}/patterns_d${DRAW}.jsonl --max-tests ${MAX_TESTS} --eval-timeout ${EVAL_TIMEOUT}"
[[ "${ONLY_ALL_FAIL}" == "1" ]] && ARGS="${ARGS} --only-all-fail"
[[ "${LIMIT}" != "0" ]] && ARGS="${ARGS} --limit ${LIMIT}"
for SPEC in "scout=${SCOUT_DRAWS}" "scout=${SCOUT_TRAIN}" "oss20=${OSS20_EVAL}" "oss20=${OSS20_TRAIN}" \
            "oss120=${OSS120_EVAL}" "oss120=${OSS120_TRAIN}"; do
  [[ -s "${SPEC#*=}" ]] || { echo "Missing draw file: ${SPEC#*=}" >&2; exit 1; }
  ARGS="${ARGS} --draw-file ${SPEC}"
done

RUNNER="${OUTPUT_DIR}/run_regrade.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo 'export HF_HUB_DISABLE_IMPLICIT_TOKEN=1'
  echo 'source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh'
  # Two passes: the script resumes from its own output, so a crash mid-suite costs one row.
  for PASS in 1 2; do
    echo "echo '=== regrade pass ${PASS} ==='"
    echo "python pipelinerl/swe/scripts/livecodebench/regrade_full_patterns.py ${ARGS}"
  done
} > "${RUNNER}"
chmod +x "${RUNNER}"

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job:     ${JOB_NAME}
  scope:   draw ${DRAW}, routes scout/oss20/oss120, $([[ "${ONLY_ALL_FAIL}" == "1" ]] && echo "all-fail problems only" || echo "all problems")
  tests:   up to ${MAX_TESTS} per problem, graded individually (no short circuit)
  output:  ${OUTPUT_DIR}/patterns_d${DRAW}.jsonl
  cost:    CPU only, no API spend

Submit with:
  SUBMIT=1 bash ${BASH_SOURCE[0]}
EOF
  exit 0
fi

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=32 \
  COMMAND="bash ${RUNNER}"

echo "Job:        ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
