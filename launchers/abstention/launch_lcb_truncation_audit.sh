#!/usr/bin/env bash
# Truncation audit: is the pool-unsolved set real, or an artifact of the 4096-token cap?
#
# The multi-draw LCB collection ran at --max-tokens 4096. On the resulting tensor, 62.7% of
# gpt-oss-120b's failed draws end within 50 tokens of that cap, and on the 124 problems no
# route solves at any depth it is 73.0%, at a mean of 3,604 completion tokens. A truncated
# generation yields no extractable program and is graded as a failure whatever the model was
# about to write.
#
# This is not bookkeeping. The pool-unsolved set carries the project's largest measured
# quantity: those problems consume ~82% of realized spend, and declining to start on them is
# ~63% of all oracle headroom. If a large share of the set is capped rather than unsolvable,
# the stopping result is a property of the collection rather than of the models, and several
# headline numbers need re-deriving before anything is written.
#
# The test: re-run gpt-oss-120b on exactly those 124 problems at a 32768-token cap, three
# draws, same temperature and same official grader, and count how many fall. The peer screen
# already showed these models using 8k-18k completion tokens when allowed to, so the cap is
# known to bind in this regime.
#
# Reads out via audit_truncation.py --compare:
#   * how many previously-unsolvable problems are now solved
#   * the corrected impossible fraction
#   * what share of draws exceed the old 4096 cap, i.e. how much was being cut off
#
# Nothing is overwritten. The re-collection lands in its own directory and is compared
# against the existing labels; tensors_v3 is untouched.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${LCB_COLLECTION_DIR:=/mnt/llmd/results/exps/aristides/reason/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448}"
: "${TENSORS_DIR:=/mnt/llmd/results/exps/aristides/reason/lcb_mdp_temporal_551_341_prepared_v1/tensors_v3}"
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
ROUTE_LABEL=${ROUTE_LABEL:-oss120_32k}
MODEL=${MODEL:-openai/gpt-oss-120b}
NEW_CAP=${NEW_CAP:-32768}
OLD_CAP=${OLD_CAP:-4096}
DRAWS=${DRAWS:-3}
TEMPERATURE=${TEMPERATURE:-0.2}      # matches the original multidraw collection
CONCURRENCY=${CONCURRENCY:-4}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
GEN_TIMEOUT=${GEN_TIMEOUT:-1800}     # 32k tokens of reasoning takes a while
SPLITS=${SPLITS:-train,eval}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}

JOB_NAME=${JOB_NAME:-lcb_truncation_audit_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}
IDS_FILE="${OUTPUT_DIR}/pool_unsolved_ids.txt"

[[ -s "${OPENROUTER_API_KEY_FILE}" ]] || { echo "Missing OpenRouter key at ${OPENROUTER_API_KEY_FILE}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"

# Materialize the id list before submitting, so the set under test is frozen, inspectable,
# and identical no matter when the job actually starts. This runs on the SUBMITTING host,
# which is outside the job's conda env, so it needs an explicit interpreter.
HOST_PYTHON=${HOST_PYTHON:-/home/toolkit/.conda/envs/pipeline-rl/bin/python3}
[[ -x "${HOST_PYTHON}" ]] || HOST_PYTHON=$(command -v python3)
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${HOST_PYTHON}" \
  "${REPO_ROOT}/pipelinerl/swe/scripts/livecodebench/audit_truncation.py" \
  --tensors-dir "${TENSORS_DIR}" --dump-unsolved --ids-file "${IDS_FILE}" --old-cap "${OLD_CAP}" \
  > "${OUTPUT_DIR}/dump_unsolved.log"
N_IDS=$(wc -l < "${IDS_FILE}")

COLLECT="python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py \
  --source-collection-dir ${LCB_COLLECTION_DIR} --output-dir ${OUTPUT_DIR} \
  --route-label ${ROUTE_LABEL} --model '${MODEL}' --api-key-file ${OPENROUTER_API_KEY_FILE} \
  --problem-ids-file ${IDS_FILE} --splits ${SPLITS} --temperature ${TEMPERATURE} \
  --max-tokens ${NEW_CAP} --concurrency ${CONCURRENCY} --eval-timeout ${EVAL_TIMEOUT} \
  --gen-timeout ${GEN_TIMEOUT} --max-invalid-frac 1.0"

RUNNER="${OUTPUT_DIR}/run_audit.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo 'export HF_HUB_DISABLE_IMPLICIT_TOKEN=1'
  echo 'source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh'
  echo "export OPENROUTER_API_KEY=\$(cat ${OPENROUTER_API_KEY_FILE})"
  # Two passes over each draw: the second reuses complete rows and retries API failures,
  # which matters here because 32k-token generations time out more often than 4k ones.
  for PASS in 1 2; do
    for DRAW in $(seq 0 $((DRAWS-1))); do
      echo "echo '=== ${ROUTE_LABEL} draw ${DRAW} (pass ${PASS}) ==='"
      echo "${COLLECT} --output-suffix _d${DRAW}"
    done
  done
  echo "python pipelinerl/swe/scripts/livecodebench/audit_truncation.py \
  --tensors-dir ${TENSORS_DIR} --compare --ids-file ${IDS_FILE} \
  --recollect-dir ${OUTPUT_DIR} --route-label ${ROUTE_LABEL} --new-cap ${NEW_CAP} \
  | tee ${OUTPUT_DIR}/audit_report.txt"
} > "${RUNNER}"
chmod +x "${RUNNER}"

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job:       ${JOB_NAME}
  problems:  ${N_IDS} pool-unsolved ids (frozen at ${IDS_FILE})
  model:     ${MODEL} at ${NEW_CAP} tokens (was ${OLD_CAP}), ${DRAWS} draws, T=${TEMPERATURE}
  calls:     $(( N_IDS * DRAWS ))
  output:    ${OUTPUT_DIR}
  report:    ${OUTPUT_DIR}/audit_report.txt

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
echo "Report:     ${OUTPUT_DIR}/audit_report.txt"
