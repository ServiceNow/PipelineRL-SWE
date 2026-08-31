#!/usr/bin/env bash
# Experiment B: learn the decay instead of imposing it.
#
# The state-conditioned model reads counts and predicts a probability per state. It
# demonstrably fails to represent the decay: flat for scout (-0.0041 across its own
# failures), and its `nothing` head never falls below 0.30 when 83.8% of depth-10
# states are doomed. It also overfits before finishing one epoch on 551 problems
# (best_calibration_epoch = 0).
#
# The factorized model reads the problem statement ALONE and emits six numbers -- a
# difficulty theta_m and a learned persistence s_m per route -- from which every
# depth follows as theta_m * s_m/(s_m + n_m). So:
#   * the decay is a model output, not the hand-set pseudo-count of 2.0;
#   * it is monotone in n by construction, so the flat/compressed curves cannot occur;
#   * `nothing` is DERIVED from the route beliefs rather than trained against them,
#     making the two consistent where today they can contradict each other;
#   * capacity drops from 4 outputs x 10,665 states to 6 outputs x 551 problems, with
#     every state of a problem now informing the same six parameters -- the right
#     medicine for a model whose calibration loss peaks at epoch 0.
#
# It also makes the Bellman lattice exact at any horizon from a single forward pass,
# since future beliefs are closed-form rather than needing a query per successor.
#
# The `sequential` family is the correct one here, NOT `sequential_decay`: a factorized
# scorer already returns theta_m * s_m/(s_m + n_m) with a learned s_m, so applying the
# hand-set analytic decay on top compounds the constant the model replaces. The replay
# now raises rather than silently double-decaying.
#
# Stages are strictly dependent (dataset -> train -> replay), not independent runs.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-17}
SOURCE_ARTIFACT=${SOURCE_ARTIFACT:-/mnt/llmd/results/exps/aristides/reason/lcb_mdp_temporal_551_341_counts_last_unweighted_1787939458b}
TENSORS_DIR=${TENSORS_DIR:-${SOURCE_ARTIFACT}/tensors_v3}
JOB_NAME=${JOB_NAME:-lcb_mdp_factorized_seed${SEED}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}
DATASET_DIR=${DATASET_DIR:-${OUTPUT_DIR}/reachable_dataset}
MODEL_DIR=${MODEL_DIR:-${OUTPUT_DIR}/model}
REPLAY_DIR=${REPLAY_DIR:-${OUTPUT_DIR}/replay}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LR=${LR:-2e-5}
NPROC=${NPROC:-4}
NUM_ORDERINGS=${NUM_ORDERINGS:-5}
HISTORIES_PER_PROBLEM=${HISTORIES_PER_PROBLEM:-10}
SNAPSHOT=${SNAPSHOT:-1}

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
  tensors (reused): ${TENSORS_DIR}
  stage 1: rebuild reachable dataset (adds problem_text + raw counts)
  stage 2: train factorized policy -- 6 numbers per problem, learned decay
  stage 3: replay, Bellman H=2, against the same RoR baseline
  output: ${OUTPUT_DIR}
  compare against: lcb_mdp_temporal_551_341_counts_last_unweighted_1787939458b

Submit explicitly with:
  SUBMIT=1 bash ${BASH_SOURCE[0]}
EOF
  exit 0
fi

if [[ "${NPROC}" -gt 1 ]]; then
  # Explicit --config_file: a bare `accelerate launch` picks up the user default,
  # which points at a deepspeed config that does not exist in the job image.
  TRAIN_LAUNCH="python -m accelerate.commands.launch --multi_gpu --mixed_precision bf16 --num_processes ${NPROC} --config_file conf/accelerate/base_mp.yaml"
else
  TRAIN_LAUNCH="python"
fi

# No cd into the live tree: SNAPSHOT=1 already points --workdir and PYTHONPATH at the
# snapshot, and stage 3 begins hours after launch.
COMMAND="python pipelinerl/swe/scripts/livecodebench/build_mdp_reachable_dataset.py \
  --tensors-dir ${TENSORS_DIR} --output-dir ${DATASET_DIR} \
  --histories-per-problem ${HISTORIES_PER_PROBLEM} --start-protocol scout_first \
  --state-layout counts_last --seed ${SEED} && \
${TRAIN_LAUNCH} pipelinerl/swe/scripts/livecodebench/train_mdp_reachable_policy.py \
  --dataset-dir ${DATASET_DIR} --output-dir ${MODEL_DIR} --seed ${SEED} \
  --num-epochs ${NUM_EPOCHS} --lr ${LR} --max-seq-length 8192 --factorized && \
python pipelinerl/swe/scripts/livecodebench/replay_mdp_full_execution.py \
  --tensors-dir ${TENSORS_DIR} --output-dir ${REPLAY_DIR} \
  --sequential-model-dir ${MODEL_DIR} \
  --num-orderings ${NUM_ORDERINGS} --start-protocol scout_first \
  --state-layout counts_last --cost-mode usd \
  --bellman-horizons 2 --retention-grid 0.90,0.95,0.98,1.0 \
  --q-stop-family sequential --q-stop-horizon 2 \
  --oracle-stopping-family sequential --oracle-stopping-horizon 2"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC="${NPROC}" GPU=1 GPU_MEM=80 CPU=16 CPU_MEM=128 \
  COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output: ${OUTPUT_DIR}"
