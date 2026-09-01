#!/usr/bin/env bash
# Decision-focused training: fix the predict-then-optimize mismatch we measured.
#
# The factorized run is a clean demonstration of the mismatch. It beat the baseline on
# every route head's AUC (.831/.845/.785 vs .817/.827/.753) AND on calibration
# (ECE .087/.135/.122 vs .172/.213/.231), and produced a WORSE policy (-3% to -11% cost
# at matched accuracy). Better predictions, worse decisions.
#
# The mechanism is measurable. The value rule acts on sign(p_m*R - c_m + ...), so a state
# whose p_m is far from c_m/R cannot change the action however wrong it is. At the
# reported operating points 66-79% of the flat BCE signal sits on such states:
#
#   R        scout      oss20     oss120    any head near a threshold
#   0.0155   43.0%      38.1%      0.0%     63.8%
#   0.1240   25.1%       0.0%     31.9%     33.6%
#   0.6560   21.4%       0.0%      0.0%     21.4%
#
# So reweight each (state, head) BCE term by proximity to c_m/R over the swept R grid,
# floored rather than zeroed so far states still supply calibration. The `nothing` head
# keeps weight 1.0 -- it has no threshold in the value rule and only the q-stop arms
# read it.
#
# Three seeds, because the headline metric carries an 11.8% cost CV and a single seed
# cannot distinguish a real gain from seed noise.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-17}
SOURCE_ARTIFACT=${SOURCE_ARTIFACT:-/mnt/llmd/results/exps/aristides/reason/lcb_mdp_temporal_551_341_counts_last_unweighted_1787939458b}
TENSORS_DIR=${TENSORS_DIR:-${SOURCE_ARTIFACT}/tensors_v3}
DATASET_DIR=${DATASET_DIR:-${SOURCE_ARTIFACT}/reachable_dataset}
JOB_NAME=${JOB_NAME:-lcb_mdp_decisionfocus_seed${SEED}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}
MODEL_DIR=${MODEL_DIR:-${OUTPUT_DIR}/model}
REPLAY_DIR=${REPLAY_DIR:-${OUTPUT_DIR}/replay}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LR=${LR:-2e-5}
NPROC=${NPROC:-4}
NUM_ORDERINGS=${NUM_ORDERINGS:-5}
SNAPSHOT=${SNAPSHOT:-1}
DECISION_R_GRID=${DECISION_R_GRID:-0.0155,0.0546,0.124,0.285,0.656}
DECISION_SIGMA=${DECISION_SIGMA:-0.10}
DECISION_FLOOR=${DECISION_FLOOR:-0.1}

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
  dataset (reused, no collection): ${DATASET_DIR}
  stage 1: train with --decision-weighting boundary
           R grid ${DECISION_R_GRID} | sigma ${DECISION_SIGMA} | floor ${DECISION_FLOOR}
  stage 2: replay, Bellman H=2, same arms as the baseline
  output: ${OUTPUT_DIR}
  compare against: ${SOURCE_ARTIFACT}/replay_tier1_v1
    baseline learned-vs-counts savings: +17.1% (55%), +29.8% (60%), +14.5% (68%)

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

# No cd into the live tree; SNAPSHOT=1 already points --workdir and PYTHONPATH at the snapshot.
COMMAND="${TRAIN_LAUNCH} pipelinerl/swe/scripts/livecodebench/train_mdp_reachable_policy.py \
  --dataset-dir ${DATASET_DIR} --output-dir ${MODEL_DIR} --seed ${SEED} \
  --num-epochs ${NUM_EPOCHS} --lr ${LR} --max-seq-length 8192 \
  --decision-weighting boundary --decision-r-grid ${DECISION_R_GRID} \
  --decision-sigma ${DECISION_SIGMA} --decision-floor ${DECISION_FLOOR} && \
python pipelinerl/swe/scripts/livecodebench/replay_mdp_full_execution.py \
  --tensors-dir ${TENSORS_DIR} --output-dir ${REPLAY_DIR} \
  --sequential-model-dir ${MODEL_DIR} \
  --num-orderings ${NUM_ORDERINGS} --start-protocol scout_first \
  --state-layout counts_last --cost-mode usd \
  --bellman-horizons 2 --retention-grid 0.90,0.95,0.98,1.0"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC="${NPROC}" GPU=1 GPU_MEM=80 CPU=16 CPU_MEM=128 \
  COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output: ${OUTPUT_DIR}"
