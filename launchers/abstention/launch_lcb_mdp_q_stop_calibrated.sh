#!/usr/bin/env bash
# Experiment A: is q(s)'s failure representational, or only calibration?
#
# Uncalibrated q(s) never drops below 0.30, so every threshold at or under that
# abstains on nothing, and where it does abstain it loses to the plain value rule.
# But 83.8% of depth-10 states are genuinely doomed, and the head's ranking is
# usable (AUC 0.733). A compressed range with usable ranking is the signature of a
# calibration fault, not a representational one: BCE against a base rate swinging
# 0.170 -> 0.838 across depth, with 43% of the data at depths 1-2, drags every
# prediction toward the shallow base rate.
#
# Per-(head, depth) Platt scaling injects the missing level while preserving the
# ranking the model does supply. If that restores q-stopping, the head was fine and
# the loss was wrong. If it does not, the defect is representational and experiment
# B (factorized difficulty) is the justified next step.
#
# Two stages in one job because they are strictly dependent: the calibration map
# must exist before the replay can consume it. This is not the forbidden pattern of
# chaining independent runs.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
ARTIFACT_DIR=${ARTIFACT_DIR:-/mnt/llmd/results/exps/aristides/reason/lcb_mdp_temporal_551_341_counts_last_unweighted_1787939458b}
TENSORS_DIR=${TENSORS_DIR:-${ARTIFACT_DIR}/tensors_v3}
DATASET_DIR=${DATASET_DIR:-${ARTIFACT_DIR}/reachable_dataset}
MODEL_DIR=${MODEL_DIR:-${ARTIFACT_DIR}/model}
REPLAY_TAG=${REPLAY_TAG:-replay_q_stop_calibrated_v1}
OUTPUT_DIR=${OUTPUT_DIR:-${ARTIFACT_DIR}/${REPLAY_TAG}}
CALIBRATION_MAP=${CALIBRATION_MAP:-${ARTIFACT_DIR}/depth_calibration_v1.json}
JOB_NAME=${JOB_NAME:-lcb_q_stop_calibrated_${TIMESTAMP}}
# Extends past the previous grid, which stopped at 0.50 and reached only 18.6%
# abstention against the value rule's 29.6%, leaving the comparison unmatched.
Q_STOP_GRID=${Q_STOP_GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70}
RETENTION_GRID=${RETENTION_GRID:-0.95}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-2000}
SNAPSHOT=${SNAPSHOT:-1}

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
  stage 1: fit per-(head, depth) Platt scaling on the CALIBRATION split
           -> ${CALIBRATION_MAP}
  stage 2: q-stopping replay consuming that map
           -> ${OUTPUT_DIR}
  q grid: ${Q_STOP_GRID}
  compare against: replay_q_stop_v1 (uncalibrated) and replay_oracle_decomp_v1

Submit explicitly with:
  SUBMIT=1 bash ${BASH_SOURCE[0]}
EOF
  exit 0
fi

# No cd into the live tree: SNAPSHOT=1 already points --workdir and PYTHONPATH at
# the snapshot, and stage 2 begins long after launch.
COMMAND="python pipelinerl/swe/scripts/livecodebench/fit_depth_calibration.py \
  --dataset-dir ${DATASET_DIR} \
  --model-dir ${MODEL_DIR} \
  --output ${CALIBRATION_MAP} \
  --split calibration && \
python pipelinerl/swe/scripts/livecodebench/replay_mdp_full_execution.py \
  --tensors-dir ${TENSORS_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --sequential-model-dir ${MODEL_DIR} \
  --calibration-map ${CALIBRATION_MAP} \
  --num-orderings 5 \
  --start-protocol scout_first \
  --state-layout counts_last \
  --cost-mode usd \
  --bellman-horizons 2 \
  --retention-grid ${RETENTION_GRID} \
  --q-stop-family sequential_decay --q-stop-horizon 2 --q-stop-grid ${Q_STOP_GRID} \
  --bootstrap-samples ${BOOTSTRAP_SAMPLES}"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=1 GPU_MEM=40 CPU=8 CPU_MEM=48 \
  COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
