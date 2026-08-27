#!/usr/bin/env bash
# Standalone replay over an existing MDP artifact (tensors_v2 + trained model).
#
# One protocol per job. Do NOT chain protocols in a single job: they are
# independent, and chaining scout_first + free_start previously turned two ~55
# minute runs into a 1h48m serial wait.
#
# STATE_LAYOUT must match the layout the model was TRAINED on, or the scorer is
# fed state text it has never seen. Artifacts built before 2026-08-27 used
# problem_first; counts_last is the current default in the builder.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${ARTIFACT_DIR:?Set ARTIFACT_DIR to a run containing tensors_v2/ and model/}"
: "${STATE_LAYOUT:?Set STATE_LAYOUT to the layout the model was trained on (problem_first|counts_last)}"

START_PROTOCOL=${START_PROTOCOL:-scout_first}
NUM_ORDERINGS=${NUM_ORDERINGS:-5}
COST_MODE=${COST_MODE:-usd}
CALIBRATION_MAP=${CALIBRATION_MAP:-}
REPLAY_TAG=${REPLAY_TAG:-replay_${START_PROTOCOL}}
SNAPSHOT=${SNAPSHOT:-1}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
JOB_NAME=${JOB_NAME:-lcb_replay_${START_PROTOCOL}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-${ARTIFACT_DIR}/${REPLAY_TAG}}

COMMAND="cd ${REPO_ROOT} && \
python pipelinerl/swe/scripts/livecodebench/replay_mdp_full_execution.py \
  --tensors-dir ${ARTIFACT_DIR}/tensors_v2 \
  --output-dir ${OUTPUT_DIR} \
  --sequential-model-dir ${ARTIFACT_DIR}/model \
  --num-orderings ${NUM_ORDERINGS} \
  --start-protocol ${START_PROTOCOL} \
  --state-layout ${STATE_LAYOUT} \
  --cost-mode ${COST_MODE} \
  ${CALIBRATION_MAP:+--calibration-map ${CALIBRATION_MAP}}"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=1 GPU_MEM=40 CPU=8 CPU_MEM=48 \
  COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
